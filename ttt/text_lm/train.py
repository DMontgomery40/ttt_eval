from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from ttt.optim.muon import make_muon_optimizer
from ttt.text_lm.corpus import encode_corpus_files, sample_next_token_batch
from ttt.text_lm.model import TinyLm, TinyLmConfig
from ttt.text_lm.store import TextModelStore, _atomic_write_json
from ttt.tokenization.bpe import BpeTokenizer, train_bpe_from_files


def _device_from_arg(device: str) -> torch.device:
    d = device.strip().lower()
    if d == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if d == "mps" and not torch.backends.mps.is_available():
        raise ValueError("device=mps requested, but torch.backends.mps.is_available() is False")
    return torch.device(d)


def _collect_corpus_files(
    *,
    corpus_paths: Sequence[str],
    corpus_dirs: Sequence[str],
    allowed_exts: Sequence[str] = (".txt", ".md", ".text", ".tex", ".rst"),
) -> list[str]:
    def _is_lfs_pointer(path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                first = f.readline().strip()
            return first == "version https://git-lfs.github.com/spec/v1"
        except Exception:
            return False

    files: list[str] = []

    for p in corpus_paths:
        p = str(p).strip()
        if not p:
            continue
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Corpus file not found: {p}")
        files.append(p)

    for d in corpus_dirs:
        d = str(d).strip()
        if not d:
            continue
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Corpus dir not found: {d}")
        for root, dirs, names in os.walk(d):
            # Avoid accidentally ingesting git metadata or other hidden dirs.
            dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("__pycache__", "node_modules")]
            for name in names:
                if name.startswith("."):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext not in allowed_exts:
                    continue
                candidate = os.path.join(root, name)
                if _is_lfs_pointer(candidate):
                    continue
                files.append(candidate)

    out = sorted(set(os.path.abspath(f) for f in files))
    if not out:
        raise ValueError("No corpus files found (provide --corpus and/or --corpus_dir)")
    return out


def _append_jsonl(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def train(
    *,
    artifacts_root: str,
    model_id: Optional[str],
    corpus_paths: Sequence[str],
    corpus_dirs: Sequence[str],
    tokenizer_path: Optional[str],
    tokenizer_max_lines: Optional[int],
    vocab_size: int,
    d_model: int,
    backbone: str,
    seq_len: int,
    batch_size: int,
    steps: int,
    seed: int,
    device: str,
    lr: float,
    weight_decay: float,
    momentum: float,
    ns_steps: int,
    log_every: int,
    save_every: int,
) -> str:
    store = TextModelStore(artifacts_root)
    store.ensure()

    model_id = (model_id or "").strip() or store.new_model_id(prefix="lm")
    model_dir = store.paths.model_dir(model_id)
    if os.path.exists(model_dir):
        raise FileExistsError(f"Model directory already exists: {model_dir}")
    os.makedirs(model_dir, exist_ok=False)

    tok_out = store.paths.tokenizer_json(model_id)
    cfg_out = store.paths.config_json(model_id)
    ckpt_out = store.paths.checkpoint_pt(model_id)
    log_path = store.paths.train_log_jsonl(model_id)

    job_t0 = time.time()

    def _log(event: str, **fields: object) -> None:
        rec = {"event": str(event), "seconds": float(time.time() - job_t0), **fields}
        _append_jsonl(log_path, rec)

    record: dict = {
        "model_id": model_id,
        "created_at_unix": int(time.time()),
        "status": "initializing",
        "tokenizer_path": os.path.relpath(tok_out, start=store.paths.artifacts_root),
        "config_path": os.path.relpath(cfg_out, start=store.paths.artifacts_root),
        "checkpoint_path": os.path.relpath(ckpt_out, start=store.paths.artifacts_root),
        "train_log_path": os.path.relpath(log_path, start=store.paths.artifacts_root),
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "backbone": str(backbone),
        "seq_len": int(seq_len),
        "steps": int(steps),
        "device": str(device),
        "optimizer": {
            "name": "muon",
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "momentum": float(momentum),
            "ns_steps": int(ns_steps),
        },
    }
    store.register_model(model_id, record)

    try:
        _log("initializing", detail="starting training job")
        dev = _device_from_arg(device)
        rng = random.Random(int(seed))
        torch.manual_seed(int(seed))

        _log("collecting_corpus", detail="scanning corpus paths")
        corpus_files = _collect_corpus_files(corpus_paths=corpus_paths, corpus_dirs=corpus_dirs)
        _log("corpus_ready", file_count=len(corpus_files))

        if tokenizer_path:
            record["status"] = "loading_tokenizer"
            store.register_model(model_id, record)
            _log("tokenizer_loading", tokenizer_path=os.path.relpath(tokenizer_path, start=os.getcwd()))
            tok = BpeTokenizer.load(tokenizer_path)
        else:
            record["status"] = "tokenizing"
            store.register_model(model_id, record)
            _log(
                "tokenizer_training",
                vocab_size=int(vocab_size),
                tokenizer_max_lines=(None if tokenizer_max_lines is None else int(tokenizer_max_lines)),
            )

            def _tok_hook(msg: dict) -> None:
                clean = dict(msg)
                clean.pop("seconds", None)  # keep a single clock in train_log.jsonl
                _log("tokenizer_progress", **clean)

            tok = train_bpe_from_files(
                corpus_files,
                vocab_size=vocab_size,
                max_lines=tokenizer_max_lines,
                progress_hook=_tok_hook,
            )

        tok.save(tok_out)

        record["status"] = "encoding_corpus"
        store.register_model(model_id, record)
        _log("encoding_start")
        ids = encode_corpus_files(tok, corpus_files)
        _log("encoding_done", token_count=int(len(ids)))

        cfg = TinyLmConfig(vocab_size=tok.vocab_size, d_model=int(d_model), backbone=backbone)  # type: ignore[arg-type]
        model = TinyLm(cfg).to(dev)
        opt = make_muon_optimizer(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            nesterov=True,
            ns_steps=int(ns_steps),
            adjust_lr_fn=None,
        )

        _atomic_write_json(cfg_out, model.config_dict())

        ckpt0 = {
            "model_id": model_id,
            "step": 0,
            "config": model.config_dict(),
            "model_state": model.state_dict(),
        }
        torch.save(ckpt0, ckpt_out)

        record["status"] = "running"
        record["device"] = str(dev)
        record["vocab_size"] = int(tok.vocab_size)
        store.register_model(model_id, record)
        _log("train_start", device=str(dev), vocab_size=int(tok.vocab_size))

        train_t0 = time.time()
        for step in range(1, int(steps) + 1):
            batch = sample_next_token_batch(ids, batch_size=batch_size, seq_len=seq_len, device=dev, rng=rng)
            logits = model(batch.x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.y.reshape(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_sq = 0.0
            for p in model.parameters():
                if p.grad is None:
                    continue
                grad_sq += float(p.grad.detach().float().pow(2).sum().item())
            grad_norm = float(grad_sq ** 0.5)
            opt.step()

            if step % max(1, int(log_every)) == 0 or step == 1:
                dt = time.time() - train_t0
                tokens_total = int(step) * int(batch_size) * int(seq_len)
                rec = {
                    "step": step,
                    "loss": float(loss.item()),
                    "grad_norm": grad_norm,
                    "seconds": dt,
                    "tokens": tokens_total,
                }
                _append_jsonl(log_path, rec)
                print(json.dumps(rec))

            if step % max(1, int(save_every)) == 0 or step == int(steps):
                ckpt = {
                    "model_id": model_id,
                    "step": step,
                    "config": model.config_dict(),
                    "model_state": model.state_dict(),
                }
                torch.save(ckpt, ckpt_out)
    except Exception as e:
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}"
        record["completed_at_unix"] = int(time.time())
        store.register_model(model_id, record)
        raise
    else:
        record["status"] = "completed"
        record["completed_at_unix"] = int(time.time())
        store.register_model(model_id, record)

    return model_id


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Train a tiny BPE LM (Muon optimizer)")
    p.add_argument("--artifacts_root", type=str, default="artifacts")
    p.add_argument("--model_id", type=str, default="", help="Optional explicit model id")
    p.add_argument("--corpus", nargs="+", default=[], help="One or more UTF-8 text files")
    p.add_argument("--corpus_dir", nargs="+", default=[], help="One or more directories of UTF-8 text files")
    p.add_argument("--tokenizer", type=str, default="", help="Existing tokenizer.json (optional)")
    p.add_argument("--vocab_size", type=int, default=4096, help="Used only if training tokenizer")
    p.add_argument(
        "--tokenizer_max_lines",
        type=int,
        default=2000,
        help="If training a tokenizer, limit BPE training to the first N lines across corpus files (keeps init fast)",
    )
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--backbone", type=str, default="ssm", choices=["ssm", "gru"])
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|mps")

    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--ns_steps", type=int, default=5)

    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=200)
    args = p.parse_args(list(argv) if argv is not None else None)

    tokenizer_path = args.tokenizer.strip() or None
    model_id = train(
        artifacts_root=args.artifacts_root,
        model_id=args.model_id.strip() or None,
        corpus_paths=args.corpus,
        corpus_dirs=args.corpus_dir,
        tokenizer_path=tokenizer_path,
        tokenizer_max_lines=(None if tokenizer_path else int(args.tokenizer_max_lines)),
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        backbone=args.backbone,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        ns_steps=args.ns_steps,
        log_every=args.log_every,
        save_every=args.save_every,
    )
    print(f"Trained model_id={model_id}")


if __name__ == "__main__":
    main()
