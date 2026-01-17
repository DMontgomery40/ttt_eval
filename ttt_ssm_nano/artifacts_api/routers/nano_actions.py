from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..actions import ForkSessionRequest, fork_session
from ..deps import get_artifact_reader
from ..reader import ArtifactReader

router = APIRouter()


@router.post("/api/sessions/{parent_session_id}/fork")
def fork(
    parent_session_id: str,
    req: ForkSessionRequest,
    reader: ArtifactReader = Depends(get_artifact_reader),
) -> dict:
    try:
        fork_session(artifacts_root=reader.artifacts_root(), parent_session_id=parent_session_id, req=req)
        return reader.load_session_data(req.child_session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

