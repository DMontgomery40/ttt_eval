from __future__ import annotations

from fastapi import Request

from .reader import ArtifactReader
from .text_runs import TextRunStore
from ttt.text_lm.store import TextModelStore
from .text_lm_service import TextLmService
from .text_train_manager import TextTrainManager


def get_artifact_reader(request: Request) -> ArtifactReader:
    reader = getattr(request.app.state, "reader", None)
    if not isinstance(reader, ArtifactReader):
        raise RuntimeError("ArtifactReader not configured on app.state.reader")
    return reader


def get_text_run_store(request: Request) -> TextRunStore:
    store = getattr(request.app.state, "text_run_store", None)
    if not isinstance(store, TextRunStore):
        raise RuntimeError("TextRunStore not configured on app.state.text_run_store")
    return store


def get_text_model_store(request: Request) -> TextModelStore:
    store = getattr(request.app.state, "text_model_store", None)
    if not isinstance(store, TextModelStore):
        raise RuntimeError("TextModelStore not configured on app.state.text_model_store")
    return store


def get_text_lm_service(request: Request) -> TextLmService:
    svc = getattr(request.app.state, "text_lm_service", None)
    if not isinstance(svc, TextLmService):
        raise RuntimeError("TextLmService not configured on app.state.text_lm_service")
    return svc


def get_text_train_manager(request: Request) -> TextTrainManager:
    mgr = getattr(request.app.state, "text_train_manager", None)
    if not isinstance(mgr, TextTrainManager):
        raise RuntimeError("TextTrainManager not configured on app.state.text_train_manager")
    return mgr
