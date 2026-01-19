"""
Post-update rollback mechanism for TTT safety.

The rollback mechanism monitors a "canary" text to detect if an update
corrupted the model. If the canary loss spikes after an update, the
weights are reverted to their pre-update state.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .model import ToyTTTModel


def robust_zscore(value: float, history: List[float]) -> Optional[float]:
    """
    Robust z-score using median and MAD.
    Returns None if history is too short (< 8 samples).
    """
    if len(history) < 8:
        return None
    med = statistics.median(history)
    abs_devs = [abs(x - med) for x in history]
    mad = statistics.median(abs_devs)
    if mad == 0:
        mad = 1e-8
    return (value - med) / (1.4826 * mad)


def compute_canary_loss(
    model: "ToyTTTModel", input_ids: torch.Tensor, vocab_size: int
) -> float:
    """
    Compute next-token cross-entropy loss for canary input without backprop.
    Used as a "drift probe" to detect if an update corrupted the model.
    """
    with torch.no_grad():
        logits, _ = model(input_ids, return_emb=False)
        loss = compute_next_token_loss_from_logits(logits, input_ids)
        return float(loss.item())


def compute_next_token_loss_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Next-token cross-entropy given model logits and the corresponding input ids.

    Args:
        logits: (B, T, V)
        input_ids: (B, T) token ids aligned with logits

    Returns:
        Scalar CE loss tensor (on the same device as logits).
    """
    vocab_size = int(logits.size(-1))
    logits2 = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        logits2.view(-1, vocab_size),
        labels.view(-1),
    )
