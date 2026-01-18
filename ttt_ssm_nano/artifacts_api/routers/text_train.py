from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..deps import get_text_train_manager
from ..text_train_manager import TextTrainManager

router = APIRouter()


class StartTextTrainRequest(BaseModel):
    corpus_paths: List[str] = Field(..., min_length=1)
    tokenizer_path: Optional[str] = None

    vocab_size: int = Field(default=4096, ge=512, le=65536)
    d_model: int = Field(default=256, ge=32, le=4096)
    backbone: str = Field(default="ssm")

    seq_len: int = Field(default=128, ge=8, le=4096)
    batch_size: int = Field(default=32, ge=1, le=4096)
    steps: int = Field(default=2000, ge=1, le=10_000_000)
    seed: int = Field(default=0, ge=0, le=1_000_000)
    device: str = Field(default="auto")

    lr: float = Field(default=0.003, gt=0.0, le=10.0)
    weight_decay: float = Field(default=0.0, ge=0.0, le=10.0)
    momentum: float = Field(default=0.95, ge=0.0, le=0.9999)
    ns_steps: int = Field(default=5, ge=1, le=20)

    log_every: int = Field(default=20, ge=1, le=1_000_000)
    save_every: int = Field(default=200, ge=1, le=1_000_000)


@router.get("/api/text/train/jobs")
def list_jobs(mgr: TextTrainManager = Depends(get_text_train_manager)) -> list:
    try:
        return mgr.list_jobs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/text/train")
def start_train(req: StartTextTrainRequest, mgr: TextTrainManager = Depends(get_text_train_manager)) -> dict:
    try:
        return mgr.start_training(
            corpus_paths=req.corpus_paths,
            tokenizer_path=(req.tokenizer_path.strip() if req.tokenizer_path else None),
            vocab_size=req.vocab_size,
            d_model=req.d_model,
            backbone=req.backbone,
            seq_len=req.seq_len,
            batch_size=req.batch_size,
            steps=req.steps,
            seed=req.seed,
            device=req.device,
            lr=req.lr,
            weight_decay=req.weight_decay,
            momentum=req.momentum,
            ns_steps=req.ns_steps,
            log_every=req.log_every,
            save_every=req.save_every,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/text/train/{model_id}/status")
def train_status(model_id: str, mgr: TextTrainManager = Depends(get_text_train_manager)) -> dict:
    try:
        return mgr.get_status(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/text/train/{model_id}/metrics")
def train_metrics(
    model_id: str,
    limit: int = 500,
    mgr: TextTrainManager = Depends(get_text_train_manager),
) -> list:
    try:
        return mgr.get_metrics(model_id, limit=int(limit))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/text/train/{model_id}/cancel")
def cancel_train(model_id: str, mgr: TextTrainManager = Depends(get_text_train_manager)) -> dict:
    try:
        return mgr.cancel(model_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
