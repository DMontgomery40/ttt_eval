from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


GradList = List[torch.Tensor]


@dataclass(frozen=True)
class SpfwProjectionStats:
    violations_before: int
    violations_after: int
    min_dot_before: Optional[float]
    min_dot_after: Optional[float]
    proj_removed_ratio: float


def _dot(a: GradList, b: GradList) -> float:
    total = 0.0
    for ga, gb in zip(a, b):
        total += float(torch.sum(ga.float() * gb.float()).item())
    return float(total)


def _norm(a: GradList, *, eps: float = 1e-12) -> float:
    sq = 0.0
    for g in a:
        sq += float(torch.sum(g.float().pow(2)).item())
    return float((sq + eps) ** 0.5)


def _axpy_(dst: GradList, src: GradList, alpha: float) -> None:
    # dst += alpha * src
    with torch.no_grad():
        for d, s in zip(dst, src):
            d.add_(s, alpha=float(alpha))


def _sub(dst: GradList, a: GradList, b: GradList) -> None:
    # dst = a - b
    with torch.no_grad():
        for out, ga, gb in zip(dst, a, b):
            out.copy_(ga).add_(gb, alpha=-1.0)


def _allowed_slack(
    *,
    c_norm: float,
    g_norm: float,
    eps_dot: float,
    eps_cos: float,
) -> float:
    """
    Scale-aware slack for the half-space constraint.

    Constraint form:
        dot(c, g) >= -allowed
    where allowed = max(eps_dot, eps_cos * ||c|| * ||g||).
    """
    return float(max(float(eps_dot), float(eps_cos) * float(c_norm) * float(g_norm)))


def project_into_safe_subspace(
    g_task: GradList,
    canary_grads: Sequence[GradList],
    *,
    eps_dot: float = 0.0,
    eps_cos: float = 0.0,
    passes: int = 1,
    tiny: float = 1e-12,
    numeric_eps_cos: float = 1e-6,
) -> Tuple[GradList, SpfwProjectionStats]:
    """
    Project `g_task` into the intersection of half-spaces defined by canary gradients.

    For each canary gradient c, enforce:
        dot(c, g_safe) >= -max(eps_dot, eps_cos * ||c|| * ||g_safe||)

    We apply sequential half-space projections (one pass by default). With multiple
    canaries, you can increase `passes` (e.g. 2) to reduce order sensitivity.
    """
    # Work on a detached copy to avoid mutating caller state.
    g = [x.detach().clone() for x in g_task]

    def _count_violations(gg: GradList) -> Tuple[int, Optional[float]]:
        if not canary_grads:
            return (0, None)
        gnorm = _norm(gg, eps=tiny)
        v = 0
        min_dot: Optional[float] = None
        for c in canary_grads:
            d = _dot(c, gg)
            cnorm = _norm(c, eps=tiny)
            allowed = _allowed_slack(c_norm=cnorm, g_norm=gnorm, eps_dot=eps_dot, eps_cos=eps_cos)
            tol = float(numeric_eps_cos) * float(cnorm) * float(gnorm) + float(tiny)
            if d < (-allowed - tol):
                v += 1
            if min_dot is None or d < min_dot:
                min_dot = d
        return (v, min_dot)

    violations_before, min_dot_before = _count_violations(g)

    for _ in range(max(1, int(passes))):
        for c in canary_grads:
            cnorm = _norm(c, eps=tiny)
            if cnorm < tiny:
                continue
            gnorm = _norm(g, eps=tiny)
            d = _dot(c, g)
            allowed = _allowed_slack(c_norm=cnorm, g_norm=gnorm, eps_dot=eps_dot, eps_cos=eps_cos)
            tol = float(numeric_eps_cos) * float(cnorm) * float(gnorm) + float(tiny)
            if d >= (-allowed - tol):
                continue
            denom = (cnorm * cnorm) + tiny
            alpha = (d + allowed) / denom
            # g <- g - alpha * c
            _axpy_(g, c, alpha=-alpha)

    violations_after, min_dot_after = _count_violations(g)

    # Projection removal ratio: ||g_task - g_safe|| / ||g_task||
    diff = [x.detach().clone() for x in g_task]
    _sub(diff, g_task, g)
    removed = _norm(diff, eps=tiny)
    base = _norm(g_task, eps=tiny)
    proj_removed_ratio = float(removed / max(base, tiny))

    return (
        g,
        SpfwProjectionStats(
            violations_before=violations_before,
            violations_after=violations_after,
            min_dot_before=min_dot_before,
            min_dot_after=min_dot_after,
            proj_removed_ratio=proj_removed_ratio,
        ),
    )


def write_grads_inplace(
    params: Sequence[torch.nn.Parameter],
    grads: Sequence[torch.Tensor],
) -> None:
    """
    Overwrite `.grad` for each param (creating it if missing).
    """
    if len(params) != len(grads):
        raise ValueError("params and grads must have same length")
    with torch.no_grad():
        for p, g in zip(params, grads):
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad.copy_(g)


def zero_grads_inplace(params: Sequence[torch.nn.Parameter]) -> None:
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            p.grad.zero_()
