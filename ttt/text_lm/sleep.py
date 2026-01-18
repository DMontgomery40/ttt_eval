from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ttt.optim.muon import make_muon_optimizer
from ttt.text_lm.corpus import sample_next_token_batch
from ttt.text_lm.model import TinyLm, TinyLmConfig
from ttt.text_lm.session_store import TextSessionStore
from ttt.text_lm.store import TextModelStore, _atomic_write_json, _read_json
from ttt.tokenization.bpe import BpeTokenizer


def _device_from_arg(device: str) -> torch.device:
    d = device.strip().lower()
    if d == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if d == "mps" and not torch.backends.mps.is_available():
        raise ValueError("device=mps requested but MPS not available")
    return torch.device(d)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    return out


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / max(1, len(xs))) if xs else 0.0


def _load_core_texts(core_path: str) -> List[str]:
    if not core_path or not os.path.exists(core_path):
        return []
    texts: List[str] = []
    if core_path.endswith(".jsonl"):
        for rec in _read_jsonl(core_path):
            if isinstance(rec.get("text"), str):
                texts.append(rec["text"])
    else:
        with open(core_path, "r", encoding="utf-8", errors="replace") as f:
            texts.append(f.read())
    return [t for t in texts if t and t.strip()]


def _encode_texts(tok: BpeTokenizer, texts: Sequence[str]) -> List[int]:
    ids: List[int] = []
    for t in texts:
        ids.extend(tok.encode(t, add_bos=True, add_eos=True))
    return ids


@torch.no_grad()
def _estimate_loss(model: TinyLm, token_ids: List[int], device: torch.device, seq_len: int, batches: int, seed: int) -> float:
    if len(token_ids) < seq_len + 2:
        return float("nan")
    model.eval()
    rng = random.Random(int(seed))
    vals: List[float] = []
    for _ in range(int(batches)):
        batch = sample_next_token_batch(token_ids, batch_size=16, seq_len=int(seq_len), device=device, rng=rng)
        logits = model(batch.x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.y.reshape(-1))
        vals.append(float(loss.item()))
    return _mean(vals)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Sleep consolidation: replay chat traces into base TinyLm")
    p.add_argument("--artifacts_root", type=str, default="artifacts")
    p.add_argument("--base_model_id", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--train_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.00001)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--ns_steps", type=int, default=5)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--sessions_limit", type=int, default=200)
    p.add_argument("--max_memories", type=int, default=200)
    p.add_argument("--core_path", type=str, default="")
    p.add_argument("--core_ratio", type=float, default=0.8)
    p.add_argument("--eval_batches", type=int, default=20)
    args = p.parse_args(list(argv) if argv is not None else None)

    store = TextModelStore(args.artifacts_root)
    store.ensure()

    base_id = args.base_model_id.strip() or store.latest_model_id()
    if not base_id:
        raise SystemExit("ERROR: no base model found")

    dev = _device_from_arg(args.device)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    tok = BpeTokenizer.load(store.paths.tokenizer_json(base_id))
    cfg_raw = _read_json(store.paths.config_json(base_id))
    if not isinstance(cfg_raw, dict):
        raise SystemExit("ERROR: base config.json invalid")

    cfg = TinyLmConfig(
        vocab_size=int(cfg_raw.get("vocab_size", tok.vocab_size)),
        d_model=int(cfg_raw.get("d_model", 256)),
        backbone=str(cfg_raw.get("backbone", "ssm")),
    )

    ckpt = torch.load(store.paths.checkpoint_pt(base_id), map_location="cpu")
    model = TinyLm(cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(dev)

    # Harvest memories from text_sessions/*/trace.jsonl
    sess_store = TextSessionStore(args.artifacts_root)
    sess_store.ensure()
    sessions = sess_store.list_sessions(limit=int(args.sessions_limit))

    memories: List[str] = []
    for s in sessions:
        sid = str((s or {}).get("session_id") or "").strip()
        if not sid:
            continue
        try:
            meta = sess_store.load_meta(sid)
        except Exception:
            continue
        if str(meta.get("model_id") or "").strip() != base_id:
            continue

        trace_path = sess_store.paths.trace_jsonl(sid)
        if not os.path.exists(trace_path):
            continue

        for tr in _read_jsonl(trace_path):
            if len(memories) >= int(args.max_memories):
                break
            prompt = tr.get("prompt")
            completion = tr.get("completion")
            if not isinstance(prompt, str) or not isinstance(completion, str):
                continue
            memories.append(prompt.rstrip() + "\n" + completion.rstrip() + "\n")

    core_texts = _load_core_texts(args.core_path)

    mix: List[str] = []
    cr = max(0.0, min(1.0, float(args.core_ratio)))
    if core_texts and memories:
        n = max(1, len(core_texts) + len(memories))
        n_core = int(round(cr * n))
        n_mem = max(0, n - n_core)
        core_pool = list(core_texts)
        mem_pool = list(memories)
        random.shuffle(core_pool)
        random.shuffle(mem_pool)
        while len(core_pool) < n_core:
            core_pool.extend(core_texts)
        while len(mem_pool) < n_mem:
            mem_pool.extend(memories)
        mix = core_pool[:n_core] + mem_pool[:n_mem]
        random.shuffle(mix)
    else:
        mix = core_texts + memories

    token_ids = _encode_texts(tok, mix)
    if len(token_ids) < int(args.seq_len) + 2:
        raise SystemExit(f"ERROR: sleep corpus too small: tokens={len(token_ids)} (need >= seq_len+2)")

    pre_core = _estimate_loss(model, _encode_texts(tok, core_texts), dev, int(args.seq_len), int(args.eval_batches), int(args.seed)) if core_texts else None
    pre_mix = _estimate_loss(model, token_ids, dev, int(args.seq_len), int(args.eval_batches), int(args.seed))

    # Train backbone + ln only
    for prm in model.parameters():
        prm.requires_grad = False
    for prm in model.backbone.parameters():
        prm.requires_grad = True
    for prm in model.ln.parameters():
        prm.requires_grad = True

    opt = make_muon_optimizer(
        [prm for prm in model.parameters() if prm.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        momentum=float(args.momentum),
        nesterov=True,
        ns_steps=int(args.ns_steps),
        adjust_lr_fn=None,
    )

    model.train()
    t0 = time.time()
    rng = random.Random(int(args.seed))
    for step in range(1, int(args.train_steps) + 1):
        batch = sample_next_token_batch(token_ids, batch_size=16, seq_len=int(args.seq_len), device=dev, rng=rng)
        logits = model(batch.x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, int(args.log_every)) == 0 or step == int(args.train_steps):
            print(json.dumps({"step": step, "loss": float(loss.item()), "seconds": float(time.time() - t0)}))

    post_core = _estimate_loss(model, _encode_texts(tok, core_texts), dev, int(args.seq_len), int(args.eval_batches), int(args.seed)) if core_texts else None
    post_mix = _estimate_loss(model, token_ids, dev, int(args.seq_len), int(args.eval_batches), int(args.seed))

    out_id = store.new_model_id(prefix="sleep")
    out_dir = store.paths.model_dir(out_id)
    os.makedirs(out_dir, exist_ok=False)

    shutil.copy2(store.paths.tokenizer_json(base_id), store.paths.tokenizer_json(out_id))
    shutil.copy2(store.paths.config_json(base_id), store.paths.config_json(out_id))

    ckpt_out = {
        "model_id": out_id,
        "step": int(args.train_steps),
        "config": model.config_dict(),
        "model_state": model.state_dict(),
        "sleep": {"base_model_id": base_id, "created_at_unix": int(time.time())},
    }
    torch.save(ckpt_out, store.paths.checkpoint_pt(out_id))

    manifest = {
        "schema_version": 1,
        "created_at_unix": int(time.time()),
        "base_model_id": base_id,
        "out_model_id": out_id,
        "device": str(dev),
        "seed": int(args.seed),
        "seq_len": int(args.seq_len),
        "train_steps": int(args.train_steps),
        "core_path": str(args.core_path),
        "core_ratio": float(cr),
        "memories_count": int(len(memories)),
        "eval": {"pre_core_loss": pre_core, "post_core_loss": post_core, "pre_mix_loss": pre_mix, "post_mix_loss": post_mix},
    }
    _atomic_write_json(os.path.join(out_dir, "sleep_manifest.json"), manifest)

    store.register_model(out_id, {
        "model_id": out_id,
        "created_at_unix": int(time.time()),
        "status": "sleep_candidate",
        "parent_model_id": base_id,
        "type": "sleep_consolidation",
    })

    print(f"sleep_candidate_model_id={out_id}")


if __name__ == "__main__":
    main()
