from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

import torch

from ttt.text_lm.model import TinyLm, TinyLmConfig
from ttt.text_lm.store import TextModelStore, _read_json
from ttt.tokenization.bpe import BpeTokenizer


def _device_from_arg(device: str) -> torch.device:
    d = device.strip().lower()
    if d == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return logits
    vals, _ = torch.topk(logits, k)
    cutoff = vals[-1]
    out = logits.clone()
    out[out < cutoff] = float("-inf")
    return out


@torch.no_grad()
def generate(
    *,
    artifacts_root: str,
    model_id: Optional[str],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> str:
    store = TextModelStore(artifacts_root)
    if not model_id:
        model_id = store.latest_model_id()
    if not model_id:
        raise FileNotFoundError("No trained text model found in artifacts/text_models")

    model_dir = store.paths.model_dir(model_id)
    tok = BpeTokenizer.load(store.paths.tokenizer_json(model_id))
    cfg_raw = _read_json(store.paths.config_json(model_id))
    if not isinstance(cfg_raw, dict):
        raise ValueError("Invalid config.json for model")

    cfg = TinyLmConfig(
        vocab_size=int(cfg_raw.get("vocab_size", tok.vocab_size)),
        d_model=int(cfg_raw.get("d_model", 256)),
        backbone=str(cfg_raw.get("backbone", "ssm")),  # type: ignore[arg-type]
    )

    ckpt = torch.load(store.paths.checkpoint_pt(model_id), map_location="cpu")
    model = TinyLm(cfg)
    model.load_state_dict(ckpt["model_state"])
    dev = _device_from_arg(device)
    model = model.to(dev).eval()

    ids = tok.encode(prompt, add_bos=True, add_eos=False)

    for _ in range(int(max_new_tokens)):
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        logits = model(x)[0, -1]
        logits = logits / max(1e-6, float(temperature))
        logits = _top_k_filter(logits, int(top_k))
        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())
        ids.append(next_id)
        if next_id == tok.eos_id:
            break

    return tok.decode(ids, skip_special=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Generate text from a trained tiny BPE LM")
    p.add_argument("--artifacts_root", type=str, default="artifacts")
    p.add_argument("--model_id", type=str, default="", help="Defaults to latest")
    p.add_argument("--prompt", type=str, default="Hello,")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|mps")
    args = p.parse_args(list(argv) if argv is not None else None)

    out = generate(
        artifacts_root=args.artifacts_root,
        model_id=args.model_id.strip() or None,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    print(out)


if __name__ == "__main__":
    main()
