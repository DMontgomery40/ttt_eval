from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_artifact_reader, get_text_run_store
from ..reader import ArtifactReader
from ..text_runs import TextRunStore

router = APIRouter()


@router.get("/api/health")
def health(
    reader: ArtifactReader = Depends(get_artifact_reader),
    text_runs: TextRunStore = Depends(get_text_run_store),
) -> dict:
    return {
        "ok": True,
        "artifacts_root": reader.artifacts_root(),
        "text_runs_root": text_runs.artifacts_root(),
    }

