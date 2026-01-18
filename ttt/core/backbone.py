"""
Backbone architectures for TTT models.

Provides pluggable recurrent backbones that can be swapped to study
how different architectures affect gradient dynamics and safety signals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn


BackboneType = Literal["gru", "ssm"]


class BaseBackbone(nn.Module, ABC):
    """Abstract base class for TTT backbones.

    All backbones must:
    - Accept (B, T, D) embeddings
    - Return (B, T, D) hidden states
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process embeddings through recurrent layer.

        Args:
            x: Input embeddings of shape (B, T, D)
        Returns:
            Hidden states of shape (B, T, D)
        """
        pass


class GRUBackbone(BaseBackbone):
    """GRU backbone (fallback option).

    Standard gated recurrent unit. The gating mechanism can
    hide or smooth write pressure because the recurrence itself
    is nonlinear and gated.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return h


class DiagonalSelectiveSSM(BaseBackbone):
    """
    Minimal SSM-like backbone with diagonal state update and input-dependent gate.

    This is deliberately NOT Mamba/S4/S5. It's a simple test bed that:
    - Uses linear recurrence (state space flavor)
    - Uses learned input gate (selective flavor)
    - Produces (B,T,D) hidden states like a recurrent backbone
    - Vectorized via cumprod+cumsum for GPU efficiency

    State update per token t:
        s_t = a * s_{t-1} + b * (gate(x_t) * x_t)
        h_t = c * s_t + d * x_t

    Where:
        a = tanh(a_logit) -> bounded in (-1, 1) for stability
        gate = sigmoid(linear(x_t)) -> input-dependent selectivity
        All parameters are per-dimension (diagonal)

    Why this matters for TTT:
        SSM-style linear recurrence ties write pressure more directly
        to eigenvalues/state dynamics than GRU gating. Different gradient
        distributions mean thresholds tuned to GRU may not generalize.

    Performance note:
        The forward pass is vectorized using cumprod+cumsum to avoid
        Python loop overhead. This enables efficient execution on both
        CPU and Apple MPS (GPU). Use --device mps for Apple Silicon.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

        # Diagonal SSM parameters
        # a_logit is transformed via tanh to keep |a| <= 1 for stability
        self.a_logit = nn.Parameter(torch.zeros(d_model))
        self.b = nn.Parameter(torch.ones(d_model))
        self.c = nn.Parameter(torch.ones(d_model))
        self.d = nn.Parameter(torch.zeros(d_model))

        # Input-dependent gate (selectivity)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized SSM forward pass using cumprod+cumsum.

        Closed-form recurrence: s_t = sum_{k<=t} a^{t-k} u_k
        """
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}")

        a = torch.tanh(self.a_logit)  # (D,)

        # Avoid exact zeros (division/powers). Keep sign.
        eps = 1e-4
        eps_vec = torch.full_like(a, eps)
        a_safe = torch.where(a.abs() < eps, torch.where(a < 0, -eps_vec, eps_vec), a)

        g = torch.sigmoid(self.gate(x))  # (B,T,D)
        u = self.b * (g * x)  # (B,T,D)

        # pows[t] = a_safe^t, with pows[0] = 1
        a_rep = a_safe.unsqueeze(0).expand(T, D)  # (T,D)
        if T > 1:
            pows_tail = torch.cumprod(a_rep[1:], dim=0)  # (T-1,D) => [a, a^2, ...]
            pows = torch.cat(
                [torch.ones(1, D, device=x.device, dtype=x.dtype), pows_tail],
                dim=0,
            )  # (T,D)
        else:
            pows = torch.ones(1, D, device=x.device, dtype=x.dtype)

        pows_bt = pows.unsqueeze(0)  # (1,T,D)

        # Closed form recurrence:
        # s_t = sum_{k<=t} a^{t-k} u_k
        v = u / (pows_bt + 1e-12)
        prefix = torch.cumsum(v, dim=1)
        s = prefix * pows_bt

        h = (self.c * s) + (self.d * x)
        return h


def create_backbone(backbone_type: str, d_model: int) -> BaseBackbone:
    """Factory function for backbone creation.

    Args:
        backbone_type: "gru" or "ssm"
        d_model: Hidden dimension

    Returns:
        Backbone instance
    """
    backbone_type = backbone_type.lower().strip()

    if backbone_type == "gru":
        return GRUBackbone(d_model)
    elif backbone_type in ("ssm", "diag_ssm", "diag-ssm", "selective_ssm", "selective-ssm"):
        return DiagonalSelectiveSSM(d_model)
    else:
        raise ValueError(
            f"Unknown backbone type: {backbone_type}. Use 'gru' or 'ssm'."
        )
