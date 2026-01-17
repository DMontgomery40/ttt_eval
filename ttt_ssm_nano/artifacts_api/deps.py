from __future__ import annotations

from fastapi import Request

from .reader import ArtifactReader
from .text_runs import TextRunStore


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

