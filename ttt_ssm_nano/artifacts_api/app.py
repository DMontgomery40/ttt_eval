from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .reader import ArtifactReader
from .text_runs import TextRunStore
from .routers import health, nano_actions, nano_artifacts, text_lm, text_runs
from ttt.text_lm.store import TextModelStore
from .text_lm_service import TextLmService


def create_app(*, artifacts_root: str) -> FastAPI:
    app = FastAPI(title="TTT-SSM Unified Artifacts API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.reader = ArtifactReader(artifacts_root)
    app.state.text_run_store = TextRunStore(artifacts_root)
    app.state.text_model_store = TextModelStore(artifacts_root)
    app.state.text_lm_service = TextLmService(store=app.state.text_model_store)

    app.include_router(health.router)
    app.include_router(nano_artifacts.router)
    app.include_router(nano_actions.router)
    app.include_router(text_runs.router)
    app.include_router(text_lm.router)

    return app
