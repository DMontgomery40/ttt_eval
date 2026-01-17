from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_artifact_reader
from ..reader import ArtifactReader

router = APIRouter()


@router.get("/api/index")
def index(reader: ArtifactReader = Depends(get_artifact_reader)) -> dict:
    try:
        return reader.load_session_index()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/sessions")
def sessions(reader: ArtifactReader = Depends(get_artifact_reader)) -> list:
    try:
        session_ids = reader.load_session_ids()
        return [reader.load_session_data(sid) for sid in session_ids]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/sessions/{session_id}")
def session(session_id: str, reader: ArtifactReader = Depends(get_artifact_reader)) -> dict:
    try:
        return reader.load_session_data(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

