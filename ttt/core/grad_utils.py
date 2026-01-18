from __future__ import annotations

from typing import Iterable, List, Sequence

import torch


def grads_for_loss(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    *,
    allow_unused: bool = True,
    retain_graph: bool = False,
) -> List[torch.Tensor]:
    """
    Compute gradients of `loss` w.r.t. `params`.

    This is a best-effort helper for contexts where some parameters may not
    participate in the computation graph (e.g. depending on context kind).
    When `allow_unused=True`, unused params return `None` from autograd; we
    convert those to explicit zero tensors so downstream dot products and
    projections are well-defined.
    """
    grads = torch.autograd.grad(
        loss,
        list(params),
        allow_unused=bool(allow_unused),
        retain_graph=bool(retain_graph),
    )
    out: List[torch.Tensor] = []
    for p, g in zip(params, grads):
        if g is None:
            out.append(torch.zeros_like(p))
        else:
            out.append(g.detach())
    return out


def iter_trainable_params(module: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    for p in module.parameters():
        if getattr(p, "requires_grad", False):
            yield p

