from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttt.optim.muon import make_muon_optimizer
from ttt.core.grad_utils import grads_for_loss
from ttt.core.spfw import project_into_safe_subspace, write_grads_inplace, zero_grads_inplace
from ttt.core.gate import check_gate
from ttt.core.model import tokenize as text_tokenize
from ttt.core.rollback import robust_zscore, compute_next_token_loss_from_logits

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

def _restore_params(params: Sequence[torch.nn.Parameter], snap: Sequence[torch.Tensor]) -> None:
    if len(params) != len(snap):
        raise ValueError("Snapshot length mismatch")
    with torch.no_grad():
        for p, s in zip(params, snap):
            p.copy_(s)


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
    attempted_update_norm: float = 0.0
    # Gate fields (mirrors ttt/monitors/gradient.py)
    gate_allowed: Optional[bool] = None
    gate_reasons: List[str] = field(default_factory=list)
    token_entropy: Optional[float] = None
    token_diversity: Optional[float] = None
    update_skipped: bool = False  # True if update was blocked by gate
    # Rollback fields (mirrors ttt/monitors/gradient.py)
    rollback_triggered: bool = False
    rollback_reasons: List[str] = field(default_factory=list)
    canary_loss_before: Optional[float] = None
    canary_loss_after: Optional[float] = None
    canary_delta: Optional[float] = None
    canary_delta_z: Optional[float] = None
    # Optional preview for debugging
    chunk_preview: Optional[str] = None
    # SPFW projection stats (present when cfg.spfw_enabled)
    spfw_proj_removed_ratio: Optional[float] = None
    spfw_min_canary_dot_before: Optional[float] = None
    spfw_min_canary_dot_after: Optional[float] = None
    spfw_violations_before: Optional[int] = None
    spfw_violations_after: Optional[int] = None
    spfw_write_suppressed: bool = False
    spfw_lr_eff: Optional[float] = None


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
    canary_token_ids_list: Optional[Sequence[Sequence[int]]] = None,
    decode_token_ids: Optional[Callable[[Sequence[int]], str]] = None,
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
    canary_grad_every = max(1, int(getattr(cfg, "canary_grad_every", 1)))
    enable_spfw = bool(getattr(cfg, "enable_spfw", False)) or bool(getattr(cfg, "spfw_enabled", False))
    spfw_eps_dot = float(getattr(cfg, "spfw_eps_dot", 0.0))
    spfw_eps_cos = float(getattr(cfg, "spfw_eps_cos", 0.0))
    spfw_passes = int(getattr(cfg, "spfw_passes", 1))
    spfw_stall_ratio = float(getattr(cfg, "spfw_stall_ratio", 0.99))
    enable_gate = bool(getattr(cfg, "enable_gate", False))
    enable_rollback = bool(getattr(cfg, "enable_rollback", False))

    min_entropy_threshold = float(getattr(cfg, "min_entropy_threshold", 1.0))
    min_diversity_threshold = float(getattr(cfg, "min_diversity_threshold", 0.1))
    ood_loss_threshold = float(getattr(cfg, "ood_loss_threshold", 8.0))
    ood_grad_threshold = float(getattr(cfg, "ood_grad_threshold", 2.0))

    rollback_z_threshold = float(getattr(cfg, "rollback_z_threshold", 6.0))
    rollback_abs_canary_delta = float(getattr(cfg, "rollback_abs_canary_delta", 1.0))
    history_window = max(8, int(getattr(cfg, "history_window", 64)))

    # Freeze core model explicitly (safety)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    context.train()
    # Respect context module's requires_grad flags:
    # - "fast" state should be requires_grad=True
    # - "slow" projections should remain frozen during chat

    events: List[ChatUpdateEvent] = []
    chunk_index = 0

    # Cached canary grads (one GradList per canary)
    cached_canary_grads: List[List[torch.Tensor]] = []
    canary_delta_history: List[float] = []

    def _normalize_canary_ids(ids_in: Sequence[int]) -> List[int]:
        ids2 = [int(x) for x in ids_in]
        if len(ids2) < 8:
            ids2 = (ids2 * 8)[:8] if ids2 else [0] * 8
        return ids2

    canary_ids_list: List[List[int]] = []
    if canary_token_ids_list:
        for seq in canary_token_ids_list:
            canary_ids_list.append(_normalize_canary_ids(seq))
    if enable_rollback and not canary_ids_list:
        raise ValueError("enable_rollback requires canary_token_ids_list (provide a canary via the chat service)")

    for start in range(0, len(ids) - 2, chunk_tokens):
        chunk = ids[start : start + chunk_tokens]
        if len(chunk) < 8:
            continue

        x_ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y_ids = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

        # Refresh canary grads periodically (w.r.t. current context params)
        if enable_spfw and canary_ids_list and (
            chunk_index % canary_grad_every == 0 or not cached_canary_grads
        ):
            cached_canary_grads = []
            for cids in canary_ids_list:
                cx = torch.tensor([cids[:-1]], dtype=torch.long, device=device)
                cy = torch.tensor([cids[1:]], dtype=torch.long, device=device)
                clogits = model.forward_with_context(cx, context)
                canary_loss = F.cross_entropy(clogits.reshape(-1, clogits.size(-1)), cy.reshape(-1))
                cached_canary_grads.append(grads_for_loss(canary_loss, params, allow_unused=True))

        # Canary probe (rollback): measure once per chunk before any updates
        canary_loss_before: Optional[float] = None
        rollback_canary_ids: Optional[List[int]] = None
        if enable_rollback and canary_ids_list:
            rollback_canary_ids = canary_ids_list[0]
            canary_input_ids = torch.tensor([rollback_canary_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                canary_logits = model.forward_with_context(canary_input_ids, context)
                canary_loss_before = float(
                    compute_next_token_loss_from_logits(canary_logits, canary_input_ids).item()
                )

        chunk_text = decode_token_ids(chunk) if decode_token_ids is not None else ""
        chunk_tokens_text = text_tokenize(chunk_text) if chunk_text else [str(x) for x in chunk]
        preview = chunk_text.strip().replace("\n", "\\n")[:160] if chunk_text else None

        for j in range(steps_per_message):
            opt.zero_grad(set_to_none=True)
            kind = str(getattr(cfg, "kind", "linear")).strip().lower()
            if kind == "fast_lowrank_mem" and hasattr(context, "memory_loss"):
                h = model.hidden(x_ids)
                loss = context.memory_loss(h)  # type: ignore[attr-defined]
            else:
                logits = model.forward_with_context(x_ids, context)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_ids.reshape(-1))
            loss.backward()

            gnorm = _grad_norm(params)
            spfw_stats = None
            spfw_write_suppressed = False
            lr_eff = float(cfg.lr)
            update_skipped = False

            gate_allowed: Optional[bool] = None
            gate_reasons: List[str] = []
            token_entropy: Optional[float] = None
            token_diversity: Optional[float] = None

            if enable_gate:
                gate = check_gate(
                    chunk_tokens_text,
                    chunk_text,
                    float(loss.item()),
                    float(gnorm),
                    min_entropy_threshold=min_entropy_threshold,
                    min_diversity_threshold=min_diversity_threshold,
                    ood_loss_threshold=ood_loss_threshold,
                    ood_grad_threshold=ood_grad_threshold,
                )
                gate_allowed = bool(gate.allow_update)
                gate_reasons = list(gate.reasons)
                token_entropy = float(gate.token_entropy)
                token_diversity = float(gate.token_diversity)
                if not gate.allow_update:
                    update_skipped = True

            if enable_spfw and (not update_skipped) and cached_canary_grads:
                g_task: List[torch.Tensor] = []
                for p in params:
                    if p.grad is None:
                        g_task.append(torch.zeros_like(p))
                    else:
                        g_task.append(p.grad.detach().clone())

                g_safe, spfw_stats = project_into_safe_subspace(
                    g_task,
                    cached_canary_grads,
                    eps_dot=spfw_eps_dot,
                    eps_cos=spfw_eps_cos,
                    passes=spfw_passes,
                )
                write_grads_inplace(params, g_safe)

                if spfw_stats.proj_removed_ratio >= spfw_stall_ratio:
                    spfw_write_suppressed = True
                    lr_eff = 0.0
                    zero_grads_inplace(params)

            # If gate blocks, skip the optimizer step entirely (no write, no optimizer state updates).
            if update_skipped:
                events.append(
                    ChatUpdateEvent(
                        chunk_index=int(chunk_index),
                        token_start=int(start),
                        token_end=int(start + len(chunk)),
                        step_in_chunk=int(j),
                        loss=float(loss.item()),
                        grad_norm=float(gnorm),
                        update_norm=0.0,
                        attempted_update_norm=0.0,
                        gate_allowed=gate_allowed,
                        gate_reasons=gate_reasons,
                        token_entropy=token_entropy,
                        token_diversity=token_diversity,
                        update_skipped=True,
                        rollback_triggered=False,
                        rollback_reasons=[],
                        canary_loss_before=canary_loss_before,
                        canary_loss_after=canary_loss_before,
                        canary_delta=0.0 if canary_loss_before is not None else None,
                        canary_delta_z=None,
                        chunk_preview=preview,
                        spfw_proj_removed_ratio=(spfw_stats.proj_removed_ratio if spfw_stats else None),
                        spfw_min_canary_dot_before=(spfw_stats.min_dot_before if spfw_stats else None),
                        spfw_min_canary_dot_after=(spfw_stats.min_dot_after if spfw_stats else None),
                        spfw_violations_before=(spfw_stats.violations_before if spfw_stats else None),
                        spfw_violations_after=(spfw_stats.violations_after if spfw_stats else None),
                        spfw_write_suppressed=bool(spfw_write_suppressed),
                        spfw_lr_eff=(float(lr_eff) if enable_spfw else None),
                    )
                )
                continue

            # Rollback snapshot (transaction semantics): snapshot params + optimizer state
            snap_params: Optional[List[torch.Tensor]] = None
            snap_opt: Optional[Dict] = None
            if enable_rollback and lr_eff > 0.0:
                snap_params = _snapshot_params(params)
                snap_opt = copy.deepcopy(opt.state_dict())

            before = _snapshot_params(params)

            old_lrs = [float(pg.get("lr", lr_eff)) for pg in opt.param_groups]
            for pg in opt.param_groups:
                pg["lr"] = float(lr_eff)
            opt.step()
            for pg, old_lr in zip(opt.param_groups, old_lrs):
                pg["lr"] = float(old_lr)
            unorm = _update_norm(params, before)

            attempted_update_norm = float(unorm)
            update_norm = float(unorm)

            rollback_triggered = False
            rollback_reasons: List[str] = []
            canary_loss_after: Optional[float] = None
            canary_delta: Optional[float] = None
            canary_delta_z: Optional[float] = None

            if enable_rollback and canary_loss_before is not None and lr_eff > 0.0:
                assert rollback_canary_ids is not None
                canary_input_ids = torch.tensor([rollback_canary_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    canary_logits = model.forward_with_context(canary_input_ids, context)
                    canary_loss_after = float(
                        compute_next_token_loss_from_logits(canary_logits, canary_input_ids).item()
                    )
                canary_delta = float(canary_loss_after - float(canary_loss_before))
                canary_delta_z = robust_zscore(canary_delta, canary_delta_history[-history_window:])

                should_rollback = False
                if canary_delta >= rollback_abs_canary_delta:
                    should_rollback = True
                    rollback_reasons.append(
                        f"abs_canary_delta({canary_delta:.3f}>={rollback_abs_canary_delta})"
                    )
                if canary_delta_z is not None and canary_delta_z >= rollback_z_threshold:
                    should_rollback = True
                    rollback_reasons.append(
                        f"canary_delta_z({canary_delta_z:.2f}>={rollback_z_threshold})"
                    )

                if should_rollback and snap_params is not None and snap_opt is not None:
                    rollback_triggered = True
                    _restore_params(params, snap_params)
                    opt.load_state_dict(snap_opt)
                    _optimizer_state_to_device(opt, device)
                    update_norm = 0.0
                    canary_loss_after = float(canary_loss_before)
                    canary_delta = 0.0
                    canary_delta_z = None
                else:
                    canary_delta_history.append(float(canary_delta))
            else:
                # If rollback disabled, still expose canary_before for parity
                canary_loss_after = canary_loss_before

            events.append(
                ChatUpdateEvent(
                    chunk_index=int(chunk_index),
                    token_start=int(start),
                    token_end=int(start + len(chunk)),
                    step_in_chunk=int(j),
                    loss=float(loss.item()),
                    grad_norm=float(gnorm),
                    update_norm=float(update_norm),
                    attempted_update_norm=float(attempted_update_norm),
                    gate_allowed=gate_allowed,
                    gate_reasons=gate_reasons,
                    token_entropy=token_entropy,
                    token_diversity=token_diversity,
                    update_skipped=False,
                    rollback_triggered=bool(rollback_triggered),
                    rollback_reasons=rollback_reasons,
                    canary_loss_before=canary_loss_before,
                    canary_loss_after=canary_loss_after,
                    canary_delta=canary_delta,
                    canary_delta_z=canary_delta_z,
                    chunk_preview=preview,
                    spfw_proj_removed_ratio=(spfw_stats.proj_removed_ratio if spfw_stats else None),
                    spfw_min_canary_dot_before=(spfw_stats.min_dot_before if spfw_stats else None),
                    spfw_min_canary_dot_after=(spfw_stats.min_dot_after if spfw_stats else None),
                    spfw_violations_before=(spfw_stats.violations_before if spfw_stats else None),
                    spfw_violations_after=(spfw_stats.violations_after if spfw_stats else None),
                    spfw_write_suppressed=bool(spfw_write_suppressed),
                    spfw_lr_eff=(float(lr_eff) if enable_spfw else None),
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
