from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..deps import get_text_lm_service, get_text_model_store
from ..text_lm_service import TextLmService
from ttt.text_lm.store import TextModelStore

router = APIRouter()


class GenerateTextRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model_id: Optional[str] = None
    max_new_tokens: int = Field(default=120, ge=1, le=2048)
    temperature: float = Field(default=0.9, gt=0.0, le=5.0)
    top_k: int = Field(default=50, ge=0, le=5000)


@router.get("/api/text/models")
def list_models(store: TextModelStore = Depends(get_text_model_store)) -> list:
    try:
        return store.list_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/text/generate")
def generate(
    req: GenerateTextRequest,
    svc: TextLmService = Depends(get_text_lm_service),
) -> dict:
    try:
        return svc.generate(
            prompt=req.prompt,
            model_id=req.model_id,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
