from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from ttt.monitors.gradient import run_monitor


class RunTextMonitorRequest(BaseModel):
    text: str = Field(..., min_length=1)
    device: str = "cpu"
    seed: int = 0

    backbone: Literal["gru", "ssm"] = "ssm"
    objective: Literal["ar", "mlm"] = "ar"
    mlm_prob: float = Field(default=0.15, ge=0.0, le=1.0)

    chunk_tokens: int = Field(default=128, ge=4, le=4096)
    ttt_steps_per_chunk: int = Field(default=1, ge=1, le=16)
    lr: float = Field(default=0.05, gt=0.0, le=1.0)

    enable_gate: bool = True
    enable_rollback: bool = True
    enable_canary_grad: bool = True
    canary_grad_every: int = Field(default=1, ge=1, le=64)

    safety_mode: Literal["gate_rollback", "spfw"] = "gate_rollback"
    spfw_eps_dot: float = Field(default=0.0, ge=0.0)
    spfw_eps_cos: float = Field(default=0.0, ge=0.0, le=1.0)
    spfw_passes: int = Field(default=1, ge=1, le=8)
    spfw_stall_ratio: float = Field(default=0.99, ge=0.0, le=1.0)
    spfw_gate_scale: float = Field(default=0.1, ge=0.0, le=1.0)
    spfw_canary_texts: List[str] = Field(default_factory=list)

    abs_grad_norm_threshold: float = Field(default=2.5, gt=0.0)
    abs_update_norm_threshold: float = Field(default=0.05, gt=0.0)
    robust_z_threshold: float = Field(default=6.0, gt=0.0)
    history_window: int = Field(default=64, ge=8, le=512)

    min_entropy_threshold: float = Field(default=1.0, ge=0.0)
    min_diversity_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    ood_loss_threshold: float = Field(default=8.0, ge=0.0)
    ood_grad_threshold: float = Field(default=2.0, ge=0.0)

    rollback_z_threshold: float = Field(default=6.0, gt=0.0)
    rollback_abs_canary_delta: float = Field(default=1.0, gt=0.0)


def run_text_monitor(req: RunTextMonitorRequest) -> Dict[str, Any]:
    events = run_monitor(
        req.text,
        device=req.device,
        seed=req.seed,
        chunk_tokens=req.chunk_tokens,
        ttt_steps_per_chunk=req.ttt_steps_per_chunk,
        lr=req.lr,
        abs_grad_norm_threshold=req.abs_grad_norm_threshold,
        abs_update_norm_threshold=req.abs_update_norm_threshold,
        robust_z_threshold=req.robust_z_threshold,
        history_window=req.history_window,
        enable_gate=req.enable_gate,
        min_entropy_threshold=req.min_entropy_threshold,
        min_diversity_threshold=req.min_diversity_threshold,
        ood_loss_threshold=req.ood_loss_threshold,
        ood_grad_threshold=req.ood_grad_threshold,
        enable_rollback=req.enable_rollback,
        rollback_z_threshold=req.rollback_z_threshold,
        rollback_abs_canary_delta=req.rollback_abs_canary_delta,
        backbone=req.backbone,
        objective=req.objective,
        mlm_prob=req.mlm_prob,
        enable_canary_grad=req.enable_canary_grad,
        canary_grad_every=req.canary_grad_every,
        safety_mode=req.safety_mode,
        spfw_eps_dot=req.spfw_eps_dot,
        spfw_eps_cos=req.spfw_eps_cos,
        spfw_passes=req.spfw_passes,
        spfw_stall_ratio=req.spfw_stall_ratio,
        spfw_gate_scale=req.spfw_gate_scale,
        canary_texts=(req.spfw_canary_texts if req.spfw_canary_texts else None),
    )

    events_dicts: List[Dict[str, Any]] = [asdict(e) for e in events]

    flagged = sum(1 for e in events if e.flagged)
    blocked = sum(1 for e in events if e.update_skipped)
    rolled_back = sum(1 for e in events if e.rollback_triggered)

    return {
        "events": events_dicts,
        "summary": {
            "chunks": len(events),
            "flagged": flagged,
            "blocked": blocked,
            "rollbacks": rolled_back,
        },
    }
