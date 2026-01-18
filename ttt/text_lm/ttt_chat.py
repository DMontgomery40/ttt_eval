from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttt.optim.muon import make_muon_optimizer

from .context import ContextConfig
from .model import TinyLm


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return logits
    vals, _ = torch.topk(logits, k)
    cutoff = vals[-1]
    out = logits.clone()
    out[out < cutoff] = float("-inf")
    return out


def _iter_params(module: nn.Module) -> Iterable[torch.nn.Parameter]:
    for p in module.parameters():
        if p.requires_grad:
            yield p


def _grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        sq += float(p.grad.detach().float().pow(2).sum().item())
    return float(sq ** 0.5)


def _snapshot_params(params: Iterable[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _update_norm(
    params: Sequence[torch.nn.Parameter], before: Sequence[torch.Tensor]
) -> float:
    sq = 0.0
    for p, b in zip(params, before):
        sq += float((p.detach() - b).float().pow(2).sum().item())
    return float(sq ** 0.5)


@dataclass(frozen=True)
class ChatUpdateEvent:
    chunk_index: int
    token_start: int
    token_end: int
    step_in_chunk: int
    loss: float
    grad_norm: float
    update_norm: float


def _optimizer_state_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for st in opt.state.values():
        if not isinstance(st, dict):
            continue
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device)


def adapt_context_on_tokens(
    *,
    model: TinyLm,
    context: nn.Module,
    token_ids: Sequence[int],
    cfg: ContextConfig,
    device: torch.device,
    optimizer_state: Optional[Dict] = None,
) -> Tuple[List[ChatUpdateEvent], Dict]:
    """
    Apply TTT updates to the fast context net on a token sequence.

    - Core model weights stay frozen.
    - Context net weights are updated with Muon (default).
    """
    params = [p for p in _iter_params(context)]
    if not params:
        raise ValueError("Context net has no trainable parameters")

    opt = make_muon_optimizer(
        params,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        momentum=float(cfg.momentum),
        nesterov=True,
        ns_steps=int(cfg.ns_steps),
        adjust_lr_fn=None,
    )
    if optimizer_state:
        try:
            opt.load_state_dict(optimizer_state)
            _optimizer_state_to_device(opt, device)
        except Exception:
            # Optimizer state is best-effort; fall back to fresh state.
            pass

    ids = [int(x) for x in token_ids]
    if len(ids) < 8:
        return ([], opt.state_dict())

    chunk_tokens = max(8, int(cfg.chunk_tokens))
    steps_per_message = max(1, int(cfg.steps_per_message))

    # Freeze core model explicitly (safety)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    context.train()
    for p in context.parameters():
        p.requires_grad = True

    events: List[ChatUpdateEvent] = []
    chunk_index = 0
    for start in range(0, len(ids) - 2, chunk_tokens):
        chunk = ids[start : start + chunk_tokens]
        if len(chunk) < 8:
            continue

        x_ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y_ids = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

        for j in range(steps_per_message):
            opt.zero_grad(set_to_none=True)
            logits = model.forward_with_context(x_ids, context)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_ids.reshape(-1))
            loss.backward()

            gnorm = _grad_norm(params)
            before = _snapshot_params(params)
            opt.step()
            unorm = _update_norm(params, before)

            events.append(
                ChatUpdateEvent(
                    chunk_index=int(chunk_index),
                    token_start=int(start),
                    token_end=int(start + len(chunk)),
                    step_in_chunk=int(j),
                    loss=float(loss.item()),
                    grad_norm=float(gnorm),
                    update_norm=float(unorm),
                )
            )

        chunk_index += 1

    return (events, opt.state_dict())


@torch.no_grad()
def generate_with_context(
    *,
    model: TinyLm,
    context: nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    eos_id: Optional[int] = None,
) -> List[int]:
    ids = [int(x) for x in prompt_ids]
    max_new = max(1, int(max_new_tokens))
    temp = max(1e-6, float(temperature))
    k = int(top_k)

    model.eval()
    context.eval()

    for _ in range(max_new):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model.forward_with_context(x, context)[0, -1]
        logits = logits / temp
        logits = _top_k_filter(logits, k)
        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())
        ids.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
    return ids
