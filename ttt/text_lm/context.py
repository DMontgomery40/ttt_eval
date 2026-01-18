from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal

import torch
import torch.nn as nn


ContextKind = Literal["linear", "fast_lowrank_mem"]


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

    # Fast memory geometry (only used for kind=fast_lowrank_mem)
    d_mem: int = 64
    mem_rank: int = 8

    # SPFW: Safety-Projected Fast Weights (project grads before optimizer step)
    spfw_enabled: bool = False
    spfw_eps_dot: float = 0.0
    spfw_eps_cos: float = 0.0
    spfw_passes: int = 1
    spfw_stall_ratio: float = 0.99
    canary_grad_every: int = 1
    canary_texts: List[str] = field(default_factory=list)

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
    if kind == "fast_lowrank_mem":
        from .fast_memory import LowRankFastMemoryConfig, LowRankFastMemoryContext

        mem_cfg = LowRankFastMemoryConfig(
            d_model=int(d_model),
            d_mem=int(cfg.d_mem),
            rank=int(cfg.mem_rank),
        )
        return LowRankFastMemoryContext(mem_cfg)
    raise ValueError(f"Unknown ContextKind: {cfg.kind}")
