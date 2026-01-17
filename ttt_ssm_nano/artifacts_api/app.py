from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .actions import ForkSessionRequest, fork_session
from .reader import ArtifactReader


def create_app(*, artifacts_root: str) -> FastAPI:
    app = FastAPI(title="TTT-SSM Nano Artifacts API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    reader = ArtifactReader(artifacts_root)
    app.state.reader = reader

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "artifacts_root": reader.artifacts_root()}

    @app.get("/api/index")
    def index() -> dict:
        try:
            return reader.load_session_index()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/sessions")
    def sessions() -> list:
        try:
            session_ids = reader.load_session_ids()
            return [reader.load_session_data(sid) for sid in session_ids]
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str) -> dict:
        try:
            return reader.load_session_data(session_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/sessions/{parent_session_id}/fork")
    def fork(parent_session_id: str, req: ForkSessionRequest) -> dict:
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

    return app
