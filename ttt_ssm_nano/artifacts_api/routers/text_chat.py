from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..deps import get_text_chat_service
from ..text_chat_service import TextChatService
from ttt.text_lm.context import ContextConfig

router = APIRouter()


class CreateTextSessionRequest(BaseModel):
    model_id: Optional[str] = None
    kind: Literal["linear", "fast_lowrank_mem"] = "linear"

    # Fast context net defaults (Muon)
    lr: float = Field(default=0.02, gt=0.0, le=10.0)
    weight_decay: float = Field(default=0.0, ge=0.0, le=10.0)
    momentum: float = Field(default=0.95, ge=0.0, le=0.9999)
    ns_steps: int = Field(default=5, ge=1, le=50)

    steps_per_message: int = Field(default=1, ge=1, le=128)
    chunk_tokens: int = Field(default=128, ge=8, le=8192)

    # Fast memory geometry (only for kind=fast_lowrank_mem)
    d_mem: int = Field(default=64, ge=4, le=1024)
    mem_rank: int = Field(default=8, ge=1, le=256)

    # Safety toggles (all default OFF)
    enable_gate: bool = Field(default=False)
    enable_rollback: bool = Field(default=False)
    enable_spfw: bool = Field(default=False)

    # Gate parameters (mirrors ttt/core/gate.py)
    min_entropy_threshold: float = Field(default=1.0, ge=0.0, le=16.0)
    min_diversity_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    ood_loss_threshold: float = Field(default=8.0, ge=0.0, le=100.0)
    ood_grad_threshold: float = Field(default=2.0, ge=0.0, le=100.0)

    # Rollback parameters (mirrors monitor defaults)
    rollback_z_threshold: float = Field(default=6.0, ge=0.0, le=100.0)
    rollback_abs_canary_delta: float = Field(default=1.0, ge=0.0, le=100.0)
    history_window: int = Field(default=64, ge=8, le=512)

    # SPFW: Safety-Projected Fast Weights (back-compat: spfw_enabled)
    spfw_enabled: bool = Field(default=False)
    spfw_eps_dot: float = Field(default=0.0, ge=0.0)
    spfw_eps_cos: float = Field(default=0.0, ge=0.0, le=1.0)
    spfw_passes: int = Field(default=1, ge=1, le=8)
    spfw_stall_ratio: float = Field(default=0.99, ge=0.0, le=1.0)
    canary_grad_every: int = Field(default=1, ge=1, le=64)
    canary_texts: List[str] = Field(default_factory=list)


@router.get("/api/text/sessions")
def list_sessions(
    limit: int = 100,
    svc: TextChatService = Depends(get_text_chat_service),
) -> list:
    try:
        return svc.list_sessions(limit=int(limit))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/text/sessions")
def create_session(
    req: CreateTextSessionRequest,
    svc: TextChatService = Depends(get_text_chat_service),
) -> dict:
    try:
        cfg = ContextConfig(
            kind=req.kind,
            lr=float(req.lr),
            weight_decay=float(req.weight_decay),
            momentum=float(req.momentum),
            ns_steps=int(req.ns_steps),
            steps_per_message=int(req.steps_per_message),
            chunk_tokens=int(req.chunk_tokens),
            d_mem=int(req.d_mem),
            mem_rank=int(req.mem_rank),
            enable_gate=bool(req.enable_gate),
            enable_rollback=bool(req.enable_rollback),
            enable_spfw=bool(req.enable_spfw) or bool(req.spfw_enabled),
            min_entropy_threshold=float(req.min_entropy_threshold),
            min_diversity_threshold=float(req.min_diversity_threshold),
            ood_loss_threshold=float(req.ood_loss_threshold),
            ood_grad_threshold=float(req.ood_grad_threshold),
            rollback_z_threshold=float(req.rollback_z_threshold),
            rollback_abs_canary_delta=float(req.rollback_abs_canary_delta),
            history_window=int(req.history_window),
            spfw_enabled=bool(req.spfw_enabled),
            spfw_eps_dot=float(req.spfw_eps_dot),
            spfw_eps_cos=float(req.spfw_eps_cos),
            spfw_passes=int(req.spfw_passes),
            spfw_stall_ratio=float(req.spfw_stall_ratio),
            canary_grad_every=int(req.canary_grad_every),
            canary_texts=list(req.canary_texts or []),
        )
        return svc.create_session(model_id=req.model_id, context_cfg=cfg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=160, ge=1, le=2048)
    temperature: float = Field(default=0.9, gt=0.0, le=5.0)
    top_k: int = Field(default=50, ge=0, le=5000)


@router.post("/api/text/sessions/{session_id}/chat")
def chat(
    session_id: str,
    req: ChatRequest,
    svc: TextChatService = Depends(get_text_chat_service),
) -> dict:
    try:
        out = svc.chat(
            session_id=session_id,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )
        return out.__dict__
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/text/sessions/{session_id}/reset")
def reset(
    session_id: str,
    svc: TextChatService = Depends(get_text_chat_service),
) -> dict:
    try:
        return svc.reset_session(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
