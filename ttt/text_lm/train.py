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
from ttt.text_lm.corpus import encode_corpus, load_text, sample_next_token_batch
from ttt.text_lm.model import TinyLm, TinyLmConfig
from ttt.text_lm.store import TextModelStore, _atomic_write_json
from ttt.tokenization.bpe import BpeTokenizer, train_bpe_from_files


def _device_from_arg(device: str) -> torch.device:
    d = device.strip().lower()
    if d == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def _append_jsonl(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def train(
    *,
    artifacts_root: str,
    corpus_paths: Sequence[str],
    tokenizer_path: Optional[str],
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
    model_id = store.new_model_id(prefix="lm")
    model_dir = store.paths.model_dir(model_id)
    os.makedirs(model_dir, exist_ok=False)

    dev = _device_from_arg(device)
    rng = random.Random(int(seed))
    torch.manual_seed(int(seed))

    if tokenizer_path:
        tok = BpeTokenizer.load(tokenizer_path)
    else:
        tok = train_bpe_from_files(corpus_paths, vocab_size=vocab_size)

    tok_out = store.paths.tokenizer_json(model_id)
    tok.save(tok_out)

    text = load_text(corpus_paths)
    ids = encode_corpus(tok, text)

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

    _atomic_write_json(store.paths.config_json(model_id), model.config_dict())

    log_path = store.paths.train_log_jsonl(model_id)
    t0 = time.time()

    for step in range(1, int(steps) + 1):
        batch = sample_next_token_batch(ids, batch_size=batch_size, seq_len=seq_len, device=dev, rng=rng)
        logits = model(batch.x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % max(1, int(log_every)) == 0 or step == 1:
            dt = time.time() - t0
            rec = {"step": step, "loss": float(loss.item()), "seconds": dt}
            _append_jsonl(log_path, rec)
            print(json.dumps(rec))

        if step % max(1, int(save_every)) == 0 or step == int(steps):
            ckpt = {
                "model_id": model_id,
                "step": step,
                "config": model.config_dict(),
                "model_state": model.state_dict(),
            }
            torch.save(ckpt, store.paths.checkpoint_pt(model_id))

    record = {
        "model_id": model_id,
        "created_at_unix": int(time.time()),
        "tokenizer_path": os.path.relpath(tok_out, start=store.paths.artifacts_root),
        "config_path": os.path.relpath(store.paths.config_json(model_id), start=store.paths.artifacts_root),
        "checkpoint_path": os.path.relpath(store.paths.checkpoint_pt(model_id), start=store.paths.artifacts_root),
        "train_log_path": os.path.relpath(log_path, start=store.paths.artifacts_root),
        "vocab_size": int(tok.vocab_size),
        "d_model": int(d_model),
        "backbone": str(backbone),
        "seq_len": int(seq_len),
        "steps": int(steps),
        "device": str(dev),
        "optimizer": {"name": "muon", "lr": float(lr), "weight_decay": float(weight_decay), "momentum": float(momentum), "ns_steps": int(ns_steps)},
    }
    store.register_model(model_id, record)

    return model_id


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Train a tiny BPE LM (Muon optimizer)")
    p.add_argument("--artifacts_root", type=str, default="artifacts")
    p.add_argument("--corpus", nargs="+", required=True, help="One or more UTF-8 text files")
    p.add_argument("--tokenizer", type=str, default="", help="Existing tokenizer.json (optional)")
    p.add_argument("--vocab_size", type=int, default=4096, help="Used only if training tokenizer")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--backbone", type=str, default="gru", choices=["gru", "ssm"])
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
        corpus_paths=args.corpus,
        tokenizer_path=tokenizer_path,
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

