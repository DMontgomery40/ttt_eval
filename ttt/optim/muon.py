from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch


def _as_2d_shape(p: torch.Tensor) -> Tuple[int, int]:
    if p.ndim == 0:
        return (1, 1)
    if p.ndim == 1:
        return (int(p.numel()), 1)
    m = int(p.shape[0])
    n = int(p.numel() // max(1, m))
    return (max(1, m), max(1, n))


def newton_schulz_orthogonalize(
    G: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    ns_coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> torch.Tensor:
    """
    Approximate orthogonalization for a 2D matrix using Newton–Schulz iteration.

    Muon uses `O = U V^T` where `B = U S V^T` is an SVD of a (momentum) gradient.
    Newton–Schulz provides an SVD-free approximation to this direction.
    """
    if G.ndim != 2:
        raise ValueError(f"Muon expects a 2D matrix for orthogonalization, got ndim={G.ndim}")
    a, b, c = ns_coefficients
    X = G.to(dtype=torch.float32)
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(int(steps)):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class MuonFallback(torch.optim.Optimizer):
    """
    Minimal Muon optimizer fallback.

    - Applies SGD-momentum (+ optional Nesterov) to form an update accumulator.
    - Orthogonalizes the accumulator direction with Newton–Schulz.
    - Uses decoupled weight decay (AdamW-style).

    This implementation supports arbitrary parameter shapes by flattening each
    parameter into a 2D matrix with shape `(p.shape[0], prod(p.shape[1:]))`.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        eps: float = 1e-7,
        ns_steps: int = 5,
        ns_coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        adjust_lr_fn: Optional[str] = None,  # "original" | "match_rms_adamw" | None
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if momentum < 0:
            raise ValueError("momentum must be >= 0")
        if adjust_lr_fn not in (None, "original", "match_rms_adamw"):
            raise ValueError("adjust_lr_fn must be None, 'original', or 'match_rms_adamw'")

        defaults = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            eps=float(eps),
            ns_steps=int(ns_steps),
            ns_coefficients=tuple(ns_coefficients),
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            mu = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            eps = float(group["eps"])
            ns_steps = int(group["ns_steps"])
            ns_coefficients = tuple(group["ns_coefficients"])
            adjust_lr_fn = group.get("adjust_lr_fn", None)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()

                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p)
                buf = st["momentum_buffer"]
                buf.mul_(mu).add_(grad)

                if nesterov:
                    upd = grad.add(buf, alpha=mu)
                else:
                    upd = buf

                m, n = _as_2d_shape(p)
                O2d = newton_schulz_orthogonalize(
                    upd.reshape(m, n),
                    steps=ns_steps,
                    eps=eps,
                    ns_coefficients=ns_coefficients,
                )
                O = O2d.reshape_as(p)

                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

                lr_eff = lr
                if adjust_lr_fn == "original":
                    lr_eff = lr * max(1.0, float(max(m, n)) / float(min(m, n)))
                elif adjust_lr_fn == "match_rms_adamw":
                    lr_eff = lr * 0.2 * math.sqrt(float(max(m, n)))

                p.add_(O, alpha=-lr_eff)

        return loss


def make_muon_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    adjust_lr_fn: Optional[str] = None,
) -> torch.optim.Optimizer:
    """
    Prefer `torch.optim.Muon` if available, otherwise use the fallback.
    """
    if hasattr(torch.optim, "Muon"):
        return torch.optim.Muon(
            params,
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            ns_steps=int(ns_steps),
            adjust_lr_fn=adjust_lr_fn,
        )
    return MuonFallback(
        params,
        lr=float(lr),
        weight_decay=float(weight_decay),
        momentum=float(momentum),
        nesterov=bool(nesterov),
        ns_steps=int(ns_steps),
        adjust_lr_fn=adjust_lr_fn,
    )

