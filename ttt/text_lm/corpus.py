from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from ttt.tokenization.bpe import BpeTokenizer


def load_text(paths: Sequence[str]) -> str:
    chunks: List[str] = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            chunks.append(f.read())
    return "\n".join(chunks)


@dataclass(frozen=True)
class Batch:
    x: torch.Tensor  # (B,T)
    y: torch.Tensor  # (B,T)


def sample_next_token_batch(
    token_ids: List[int],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    rng: random.Random,
) -> Batch:
    if seq_len < 4:
        raise ValueError("seq_len must be >= 4")
    if len(token_ids) < seq_len + 2:
        raise ValueError("Corpus too small for requested seq_len")

    xs: List[List[int]] = []
    ys: List[List[int]] = []

    max_start = len(token_ids) - (seq_len + 1)
    for _ in range(batch_size):
        start = rng.randint(0, max_start)
        window = token_ids[start : start + seq_len + 1]
        xs.append(window[:-1])
        ys.append(window[1:])

    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return Batch(x=x, y=y)


def encode_corpus(tokenizer: BpeTokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_bos=True, add_eos=True)

