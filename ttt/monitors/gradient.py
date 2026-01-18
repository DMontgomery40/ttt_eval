"""
Gradient monitoring for TTT safety.

Tracks how "hard" input tries to write into the TTT adapter:
- Adapter gradient norm (write pressure)
- Adapter update norm (actual write magnitude)
- Per-token influence (gradient norm w.r.t. embedding vectors)
- Compression ratio (Kolmogorov complexity proxy)
- Gradient alignment with canary (directional damage signal)

Supports multiple backbone architectures (GRU, SSM) and TTT objectives (AR, MLM).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..core.model import (
    ToyTTTModel,
    tokenize,
    ids_from_tokens,
    DEFAULT_CANARY_TEXT,
)
from ..core.backbone import BackboneType
from ..core.objective import ObjectiveType, compute_objective_loss
from ..core.gate import check_gate
from ..core.rollback import compute_canary_loss, robust_zscore
from ..core.spfw import project_into_safe_subspace, write_grads_inplace, zero_grads_inplace
from .signals import (
    compute_compression_ratio,
    compute_canary_gradient,
    compute_gradient_alignment,
    get_canary_grad_norm,
)


@dataclass
class MonitorEvent:
    """Output from monitoring a single chunk."""

    chunk_index: int
    token_start: int
    token_end: int
    loss: float
    grad_norm: float
    update_norm: float  # Effective update magnitude (after any rollback)
    grad_z: Optional[float]
    update_z: Optional[float]
    flagged: bool
    reasons: List[str]
    top_influence_tokens: List[Tuple[str, float]]
    chunk_preview: str
    # Gate decision fields
    gate_allowed: bool
    gate_reasons: List[str]
    token_entropy: float
    token_diversity: float
    update_skipped: bool  # True if update was blocked by gate
    # Rollback fields
    attempted_update_norm: float  # Update magnitude before rollback check
    rollback_triggered: bool
    rollback_reasons: List[str]
    canary_loss_before: Optional[float]
    canary_loss_after: Optional[float]
    canary_delta: Optional[float]
    canary_delta_z: Optional[float]
    # Backbone/objective metadata
    backbone: str = "ssm"
    objective: str = "ar"
    # Compression-based signal (Kolmogorov proxy)
    compression_ratio: Optional[float] = None
    # Canary gradient alignment signals
    canary_grad_norm: Optional[float] = None
    grad_canary_cos: Optional[float] = None  # Cosine similarity
    grad_canary_dot: Optional[float] = None  # Dot product
    # SPFW (Safety-Projected Fast Weights) metrics
    spfw_enabled: bool = False
    spfw_eps_dot: Optional[float] = None
    spfw_eps_cos: Optional[float] = None
    spfw_passes: Optional[int] = None
    spfw_canaries: Optional[int] = None
    spfw_min_canary_dot_before: Optional[float] = None
    spfw_min_canary_dot_after: Optional[float] = None
    spfw_violations_before: Optional[int] = None
    spfw_violations_after: Optional[int] = None
    spfw_proj_removed_ratio: Optional[float] = None
    spfw_write_suppressed: bool = False
    spfw_lr_eff: Optional[float] = None


def run_monitor(
    text: str,
    *,
    device: str = "cpu",
    seed: int = 0,
    vocab_size: int = 8192,
    d_model: int = 64,
    chunk_tokens: int = 128,
    ttt_steps_per_chunk: int = 1,
    lr: float = 0.05,
    topk: int = 10,
    abs_grad_norm_threshold: float = 2.5,
    abs_update_norm_threshold: float = 0.05,
    robust_z_threshold: float = 6.0,
    history_window: int = 64,
    # Gate parameters
    enable_gate: bool = True,
    min_entropy_threshold: float = 1.0,
    min_diversity_threshold: float = 0.1,
    ood_loss_threshold: float = 8.0,
    ood_grad_threshold: float = 2.0,
    # Rollback parameters (post-update safety net)
    enable_rollback: bool = True,
    rollback_z_threshold: float = 6.0,
    rollback_abs_canary_delta: float = 1.0,
    canary_text: str = DEFAULT_CANARY_TEXT,
    # Backbone and objective selection
    backbone: BackboneType = "ssm",
    objective: ObjectiveType = "ar",
    mlm_prob: float = 0.15,
    # Canary gradient alignment monitoring
    enable_canary_grad: bool = True,
    canary_grad_every: int = 1,
    # SPFW (project updates into safe subspace)
    safety_mode: str = "gate_rollback",  # "gate_rollback" | "spfw"
    spfw_eps_dot: float = 0.0,
    spfw_eps_cos: float = 0.0,
    spfw_passes: int = 1,
    spfw_stall_ratio: float = 0.99,
    spfw_gate_scale: float = 0.1,
    canary_texts: Optional[List[str]] = None,
) -> List[MonitorEvent]:
    """
    Run TTT monitoring on input text.

    Args:
        text: Input text to analyze
        backbone: Architecture type ("gru" or "ssm")
        objective: TTT loss function ("ar" or "mlm")
        mlm_prob: Mask probability for MLM objective
        enable_canary_grad: Compute canary gradient alignment
        canary_grad_every: Recompute canary gradient every N chunks
        safety_mode: "gate_rollback" (default) or "spfw" (project gradients then step)
        spfw_eps_dot: Absolute dot slack for constraints
        spfw_eps_cos: Cosine slack for constraints (scale-aware)
        spfw_passes: Projection passes over canaries
        spfw_stall_ratio: If projection removes > this fraction, suppress write
        spfw_gate_scale: If gate triggers, scale LR by this factor (SPFW mode)
        canary_texts: Optional list of canary texts (defaults to [canary_text])

    Returns a list of MonitorEvent objects, one per chunk.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    model = ToyTTTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        backbone=backbone,
    ).to(device)

    # Freeze everything except the adapter
    for p in model.parameters():
        p.requires_grad = False
    model.adapter.weight.requires_grad = True

    opt = torch.optim.SGD([model.adapter.weight], lr=lr)

    tokens = tokenize(text)
    ids = ids_from_tokens(tokens, vocab_size)

    safety_mode = (safety_mode or "").strip().lower()
    spfw_enabled = safety_mode == "spfw"

    # Canary setup for rollback drift detection (single canary)
    rollback_canary_input_ids: Optional[torch.Tensor] = None
    if enable_rollback:
        canary_tokens = tokenize(canary_text)
        canary_ids = ids_from_tokens(canary_tokens, vocab_size)
        if len(canary_ids) < 8:
            canary_ids = (canary_ids * 8)[:8] if canary_ids else [0] * 8
        rollback_canary_input_ids = torch.tensor([canary_ids], dtype=torch.long, device=device)

    # Canary set for gradient alignment + SPFW projection (possibly multiple)
    if canary_texts is None or len(canary_texts) == 0:
        canary_texts = [canary_text]
    canary_grad_input_ids: List[torch.Tensor] = []
    if enable_canary_grad or spfw_enabled:
        for ct in canary_texts:
            toks = tokenize(ct)
            ids2 = ids_from_tokens(toks, vocab_size)
            if len(ids2) < 8:
                ids2 = (ids2 * 8)[:8] if ids2 else [0] * 8
            canary_grad_input_ids.append(torch.tensor([ids2], dtype=torch.long, device=device))

    # Canary gradient for directional alignment (computed periodically)
    cached_canary_grads: List[torch.Tensor] = []
    cached_canary_grad_norm: float = 0.0  # norm of first canary (back-compat)

    grad_history: List[float] = []
    update_history: List[float] = []
    canary_delta_history: List[float] = []

    events: List[MonitorEvent] = []
    chunk_count = 0

    for start in range(0, len(ids), chunk_tokens):
        chunk_ids = ids[start : start + chunk_tokens]
        chunk_toks = tokens[start : start + chunk_tokens]
        if len(chunk_ids) < 4:
            continue

        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=device)
        chunk_text = " ".join(chunk_toks)

        # Compute compression ratio (Kolmogorov proxy)
        compression_ratio = compute_compression_ratio(chunk_text)

        # TTT update loop (usually 1 step per chunk in this toy)
        update_skipped = False
        rollback_triggered = False
        rollback_reasons: List[str] = []
        attempted_update_norm = 0.0
        update_norm = 0.0

        # Canary gradient alignment signals
        grad_canary_cos: Optional[float] = None
        grad_canary_dot: Optional[float] = None

        # Canary measurements
        canary_loss_before: Optional[float] = None
        canary_loss_after: Optional[float] = None
        canary_delta: Optional[float] = None
        canary_delta_z: Optional[float] = None

        # Update cached canary gradients periodically
        if (enable_canary_grad or spfw_enabled) and canary_grad_input_ids and (
            chunk_count % max(1, canary_grad_every) == 0
        ):
            cached_canary_grads = [
                compute_canary_gradient(model, cids, vocab_size) for cids in canary_grad_input_ids
            ]
            cached_canary_grad_norm = (
                get_canary_grad_norm(cached_canary_grads[0]) if cached_canary_grads else 0.0
            )

        # Measure canary before any updates in this chunk
        if enable_rollback and rollback_canary_input_ids is not None:
            canary_loss_before = compute_canary_loss(model, rollback_canary_input_ids, vocab_size)

        for _ in range(ttt_steps_per_chunk):
            opt.zero_grad(set_to_none=True)

            # Compute loss using objective-aware function
            loss, logits, emb = compute_objective_loss(
                model,
                input_ids,
                objective=objective,
                vocab_size=vocab_size,
                mlm_prob=mlm_prob,
                mask_token_id=model.mask_token_id,
                return_emb=True,
            )
            assert emb is not None

            loss.backward()

            grad_norm = float(model.adapter.weight.grad.detach().norm().item())

            # Compute gradient alignment with canary
            if enable_canary_grad and cached_canary_grads:
                chunk_grad = model.adapter.weight.grad.detach().clone()
                grad_canary_cos, grad_canary_dot = compute_gradient_alignment(
                    chunk_grad, cached_canary_grads[0]
                )

            # Per-token influence proxy
            tok_infl = emb.grad.detach().norm(dim=-1).squeeze(0)  # (T,)

            # --- Pre-update gate check ---
            gate_decision = check_gate(
                chunk_toks,
                chunk_text,
                float(loss.item()),
                grad_norm,
                min_entropy_threshold=min_entropy_threshold,
                min_diversity_threshold=min_diversity_threshold,
                ood_loss_threshold=ood_loss_threshold,
                ood_grad_threshold=ood_grad_threshold,
            )

            # Snapshot weights before update
            old = model.adapter.weight.detach().clone()

            # --- Apply update (gate/rollback policy vs SPFW) ---
            lr_eff = float(lr)
            spfw_write_suppressed = False
            spfw_stats = None

            if spfw_enabled:
                # Gate becomes an LR throttle (soft prior)
                if enable_gate and not gate_decision.allow_update:
                    lr_eff *= float(spfw_gate_scale)

                # Project task grad into safe subspace defined by canary grads
                if cached_canary_grads:
                    g_task = [model.adapter.weight.grad.detach().clone()]
                    canary_lists = [[g.detach().clone()] for g in cached_canary_grads]
                    g_safe, spfw_stats = project_into_safe_subspace(
                        g_task,
                        canary_lists,
                        eps_dot=float(spfw_eps_dot),
                        eps_cos=float(spfw_eps_cos),
                        passes=int(spfw_passes),
                    )
                    write_grads_inplace([model.adapter.weight], g_safe)

                    if spfw_stats.proj_removed_ratio >= float(spfw_stall_ratio):
                        spfw_write_suppressed = True
                        lr_eff = 0.0
                        zero_grads_inplace([model.adapter.weight])

                if lr_eff <= 0.0:
                    update_skipped = True
                    # Ensure no optimizer state updates from adversarial grads.
                    zero_grads_inplace([model.adapter.weight])

                # Temporarily override optimizer lr for this step
                old_lrs = [float(pg.get("lr", lr_eff)) for pg in opt.param_groups]
                for pg in opt.param_groups:
                    pg["lr"] = float(lr_eff)

                opt.step()

                # Restore original optimizer lrs
                for pg, old_lr in zip(opt.param_groups, old_lrs):
                    pg["lr"] = float(old_lr)
            else:
                # Original behavior: gate hard-blocks updates
                if not enable_gate or gate_decision.allow_update:
                    opt.step()
                else:
                    update_skipped = True

            if not update_skipped:
                step_update_norm = float(
                    (model.adapter.weight.detach() - old).norm().item()
                )
                attempted_update_norm += step_update_norm

                # --- Post-update rollback check ---
                if (
                    enable_rollback
                    and rollback_canary_input_ids is not None
                    and canary_loss_before is not None
                ):
                    canary_after_step = compute_canary_loss(
                        model, rollback_canary_input_ids, vocab_size
                    )
                    step_delta = canary_after_step - canary_loss_before
                    step_delta_z = robust_zscore(
                        step_delta, canary_delta_history[-history_window:]
                    )

                    should_rollback = False
                    if step_delta >= rollback_abs_canary_delta:
                        should_rollback = True
                        rollback_reasons.append(
                            f"abs_canary_delta({step_delta:.3f}>={rollback_abs_canary_delta})"
                        )
                    if step_delta_z is not None and step_delta_z >= rollback_z_threshold:
                        should_rollback = True
                        rollback_reasons.append(
                            f"canary_delta_z({step_delta_z:.2f}>={rollback_z_threshold})"
                        )

                    if should_rollback:
                        # Revert to pre-step weights
                        rollback_triggered = True
                        with torch.no_grad():
                            model.adapter.weight.copy_(old)
                        canary_loss_after = canary_loss_before
                        canary_delta = 0.0
                        update_norm = 0.0
                        break
                    else:
                        # Update succeeded, record canary delta
                        canary_loss_after = canary_after_step
                        canary_delta = step_delta
                        canary_delta_z = step_delta_z
                        update_norm += step_update_norm
                        canary_delta_history.append(step_delta)
                else:
                    # Rollback disabled, just count the update
                    update_norm += step_update_norm

        # Robust scores relative to recent history
        grad_z = robust_zscore(grad_norm, grad_history[-history_window:])
        update_z = robust_zscore(attempted_update_norm, update_history[-history_window:])

        flagged = False
        reasons: List[str] = []

        if grad_norm >= abs_grad_norm_threshold:
            flagged = True
            reasons.append("abs_grad_norm")

        if attempted_update_norm >= abs_update_norm_threshold:
            flagged = True
            reasons.append("abs_update_norm")

        if grad_z is not None and grad_z >= robust_z_threshold:
            flagged = True
            reasons.append("grad_robust_z")

        if update_z is not None and update_z >= robust_z_threshold:
            flagged = True
            reasons.append("update_robust_z")

        # Top tokens by influence
        k = min(topk, int(tok_infl.numel()))
        top_vals, top_idx = torch.topk(tok_infl, k=k)
        top_items: List[Tuple[str, float]] = []
        for j in range(k):
            idx = int(top_idx[j].item())
            top_items.append((chunk_toks[idx], float(top_vals[j].item())))

        preview = " ".join(chunk_toks[:32])

        events.append(
            MonitorEvent(
                chunk_index=start // chunk_tokens,
                token_start=start,
                token_end=start + len(chunk_ids),
                loss=float(loss.item()),
                grad_norm=grad_norm,
                update_norm=update_norm,
                grad_z=grad_z,
                update_z=update_z,
                flagged=flagged,
                reasons=reasons,
                top_influence_tokens=top_items,
                chunk_preview=preview,
                gate_allowed=gate_decision.allow_update,
                gate_reasons=gate_decision.reasons,
                token_entropy=gate_decision.token_entropy,
                token_diversity=gate_decision.token_diversity,
                update_skipped=update_skipped,
                attempted_update_norm=attempted_update_norm,
                rollback_triggered=rollback_triggered,
                rollback_reasons=rollback_reasons,
                canary_loss_before=canary_loss_before,
                canary_loss_after=canary_loss_after,
                canary_delta=canary_delta,
                canary_delta_z=canary_delta_z,
                # New fields
                backbone=backbone,
                objective=objective,
                compression_ratio=compression_ratio,
                canary_grad_norm=cached_canary_grad_norm if enable_canary_grad else None,
                grad_canary_cos=grad_canary_cos,
                grad_canary_dot=grad_canary_dot,
                spfw_enabled=spfw_enabled,
                spfw_eps_dot=float(spfw_eps_dot) if spfw_enabled else None,
                spfw_eps_cos=float(spfw_eps_cos) if spfw_enabled else None,
                spfw_passes=int(spfw_passes) if spfw_enabled else None,
                spfw_canaries=int(len(cached_canary_grads)) if (spfw_enabled and cached_canary_grads) else None,
                spfw_min_canary_dot_before=(spfw_stats.min_dot_before if spfw_stats else None),
                spfw_min_canary_dot_after=(spfw_stats.min_dot_after if spfw_stats else None),
                spfw_violations_before=(spfw_stats.violations_before if spfw_stats else None),
                spfw_violations_after=(spfw_stats.violations_after if spfw_stats else None),
                spfw_proj_removed_ratio=(spfw_stats.proj_removed_ratio if spfw_stats else None),
                spfw_write_suppressed=bool(spfw_write_suppressed),
                spfw_lr_eff=float(lr_eff) if spfw_enabled else None,
            )
        )

        grad_history.append(grad_norm)
        update_history.append(attempted_update_norm)
        chunk_count += 1

    return events
