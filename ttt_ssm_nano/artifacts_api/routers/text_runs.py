from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_text_run_store
from ..text_runs import TextRunStore
from ..ttt_monitor import RunTextMonitorRequest

router = APIRouter()


@router.get("/api/text/runs")
def list_text_runs(
    limit: int = 50,
    store: TextRunStore = Depends(get_text_run_store),
) -> list:
    try:
        return store.list_runs(limit=int(limit))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/text/runs/{run_id}")
def get_text_run(
    run_id: str,
    store: TextRunStore = Depends(get_text_run_store),
) -> dict:
    try:
        return store.get_run(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/text/runs")
def create_text_run(
    req: RunTextMonitorRequest,
    store: TextRunStore = Depends(get_text_run_store),
) -> dict:
    try:
        return store.create_run(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/ttt/monitor")
def legacy_ttt_monitor(
    req: RunTextMonitorRequest,
    store: TextRunStore = Depends(get_text_run_store),
) -> dict:
    """
    Backwards-compatible endpoint.

    Returns the historical shape `{events, summary}`, but now also persists the
    run into `artifacts/text_runs/` and includes `run_id`.
    """
    try:
        run = store.create_run(req)
        return {"run_id": run.get("run_id"), "events": run.get("events", []), "summary": run.get("summary", {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
