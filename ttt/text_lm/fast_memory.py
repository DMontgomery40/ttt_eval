from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LowRankFastMemoryConfig:
    d_model: int
    d_mem: int = 64
    rank: int = 8


class LowRankFastMemoryContext(nn.Module):
    """
    Low-rank fast-weight memory module for TinyLm.

    Slow projections (frozen during chat updates):
        k = W_k h
        q = W_q h
        v = W_v h
        out = W_o m

    Fast state (plastic during chat updates):
        A ∈ R[d_mem, r], B ∈ R[d_mem, r]
        memory map used in row-vector form:
            m = q @ (B @ A^T)
            v_hat = k @ (B @ A^T)

    The module returns a residual in hidden space: `W_o(m)`, so callers can do `h + context(h)`.
    """

    def __init__(self, cfg: LowRankFastMemoryConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.W_k = nn.Linear(cfg.d_model, cfg.d_mem, bias=False)
        self.W_q = nn.Linear(cfg.d_model, cfg.d_mem, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_mem, bias=False)
        self.W_o = nn.Linear(cfg.d_mem, cfg.d_model, bias=False)

        # Fast low-rank state. Initialize so the effective map is ~0 initially.
        self.A = nn.Parameter(torch.randn(cfg.d_mem, cfg.rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(cfg.d_mem, cfg.rank))

        # Freeze slow projections by default; only A/B are "fast".
        for p in self.W_k.parameters():
            p.requires_grad = False
        for p in self.W_q.parameters():
            p.requires_grad = False
        for p in self.W_v.parameters():
            p.requires_grad = False
        for p in self.W_o.parameters():
            p.requires_grad = False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.W_q(h)  # (B,T,d_mem)
        m = (q @ self.B) @ self.A.T  # (B,T,d_mem)
        return self.W_o(m)  # (B,T,d_model)

    def memory_loss(self, h: torch.Tensor) -> torch.Tensor:
        """
        Self-supervised associative loss encouraging the fast memory to store key/value associations:
            L_mem = mean || M(k) - v ||^2
        """
        k = self.W_k(h)
        v = self.W_v(h)
        v_hat = (k @ self.B) @ self.A.T
        return F.mse_loss(v_hat, v)

