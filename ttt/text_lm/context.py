from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal

import torch
import torch.nn as nn


ContextKind = Literal["linear"]


@dataclass(frozen=True)
class ContextConfig:
    kind: ContextKind = "linear"

    # TTT update hyperparams (fast net only)
    lr: float = 0.02
    weight_decay: float = 0.0
    momentum: float = 0.95
    ns_steps: int = 5

    # How hard we write each chat turn
    steps_per_message: int = 1
    chunk_tokens: int = 128

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LinearContextNet(nn.Module):
    """
    Fast plastic net (the weight-based "context window").

    This is intentionally small/stable: a linear residual adapter in hidden space:
        h' = h + A(h)
    """

    def __init__(self, d_model: int, *, zero_init: bool = True) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.adapter = nn.Linear(self.d_model, self.d_model, bias=False)
        if zero_init:
            nn.init.zeros_(self.adapter.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.adapter(h)


def create_context_net(*, d_model: int, cfg: ContextConfig) -> nn.Module:
    kind = str(cfg.kind).strip().lower()
    if kind == "linear":
        return LinearContextNet(d_model, zero_init=True)
    raise ValueError(f"Unknown ContextKind: {cfg.kind}")

