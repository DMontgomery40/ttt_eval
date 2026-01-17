from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .ttt_monitor import RunTextMonitorRequest, run_text_monitor


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _model_dump(model: BaseModel) -> Dict[str, Any]:
    # Pydantic v2 has model_dump(); v1 has dict().
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[no-any-return]
    return model.dict()  # type: ignore[no-any-return]


def _safe_preview(text: str, *, max_len: int = 160) -> str:
    flat = " ".join(text.strip().split())
    if len(flat) <= max_len:
        return flat
    return flat[: max_len - 1] + "…"


@dataclass(frozen=True)
class TextRunPaths:
    artifacts_root: str

    @property
    def text_runs_dir(self) -> str:
        return os.path.join(self.artifacts_root, "text_runs")

    @property
    def index_json(self) -> str:
        return os.path.join(self.text_runs_dir, "index.json")

    def run_dir(self, run_id: str) -> str:
        return os.path.join(self.text_runs_dir, run_id)

    def run_meta(self, run_id: str) -> str:
        return os.path.join(self.run_dir(run_id), "meta.json")

    def run_request(self, run_id: str) -> str:
        return os.path.join(self.run_dir(run_id), "request.json")

    def run_summary(self, run_id: str) -> str:
        return os.path.join(self.run_dir(run_id), "summary.json")

    def run_events(self, run_id: str) -> str:
        return os.path.join(self.run_dir(run_id), "events.json")

    def run_input(self, run_id: str) -> str:
        return os.path.join(self.run_dir(run_id), "input.txt")


class TextRunSummary(BaseModel):
    run_id: str
    created_at_unix: int
    preview: str = ""
    summary: Dict[str, Any] = Field(default_factory=dict)


class TextRunData(BaseModel):
    run_id: str
    created_at_unix: int
    meta: Dict[str, Any] = Field(default_factory=dict)
    request: Dict[str, Any] = Field(default_factory=dict)
    input_text: str = ""
    summary: Dict[str, Any] = Field(default_factory=dict)
    events: List[Dict[str, Any]] = Field(default_factory=list)


class TextRunStore:
    def __init__(self, artifacts_root: str) -> None:
        self.paths = TextRunPaths(artifacts_root=os.path.abspath(artifacts_root))

    def artifacts_root(self) -> str:
        return self.paths.artifacts_root

    def ensure(self) -> None:
        os.makedirs(self.paths.text_runs_dir, exist_ok=True)
        if not os.path.exists(self.paths.index_json):
            _atomic_write_json(self.paths.index_json, {"schema_version": 1, "runs": {}})

    def _load_index(self) -> Dict[str, Any]:
        if not os.path.exists(self.paths.index_json):
            return {"schema_version": 1, "runs": {}}
        idx = _read_json(self.paths.index_json)
        if not isinstance(idx, dict):
            raise ValueError("Invalid text_runs/index.json (expected object)")
        runs = idx.get("runs", {})
        if not isinstance(runs, dict):
            runs = {}
        return {"schema_version": int(idx.get("schema_version", 1) or 1), "runs": dict(runs)}

    def _write_index(self, idx: Dict[str, Any]) -> None:
        _atomic_write_json(self.paths.index_json, idx)

    def list_runs(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        self.ensure()
        idx = self._load_index()
        runs = list(idx.get("runs", {}).values())
        runs.sort(key=lambda r: int((r or {}).get("created_at_unix", 0)), reverse=True)
        return runs[: max(0, int(limit))]

    def get_run(self, run_id: str) -> Dict[str, Any]:
        self.ensure()
        run_dir = self.paths.run_dir(run_id)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Text run not found: {run_id}")

        meta = _read_json(self.paths.run_meta(run_id))
        request = _read_json(self.paths.run_request(run_id))
        summary = _read_json(self.paths.run_summary(run_id))
        events = _read_json(self.paths.run_events(run_id))

        with open(self.paths.run_input(run_id), "r", encoding="utf-8", errors="replace") as f:
            input_text = f.read()

        if not isinstance(meta, dict):
            raise ValueError("Invalid meta.json (expected object)")
        if not isinstance(request, dict):
            raise ValueError("Invalid request.json (expected object)")
        if not isinstance(summary, dict):
            raise ValueError("Invalid summary.json (expected object)")
        if not isinstance(events, list):
            raise ValueError("Invalid events.json (expected list)")

        created_at = int(meta.get("created_at_unix", 0) or 0)

        payload = TextRunData(
            run_id=run_id,
            created_at_unix=created_at,
            meta=meta,
            request=request,
            input_text=input_text,
            summary=summary,
            events=[e for e in events if isinstance(e, dict)],
        )
        return payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()

    def create_run(self, req: RunTextMonitorRequest) -> Dict[str, Any]:
        self.ensure()
        created_at_unix = int(time.time())
        base = f"text_{created_at_unix}_seed{int(req.seed)}"
        run_id = base
        suffix = 1
        while os.path.exists(self.paths.run_dir(run_id)):
            run_id = f"{base}_{suffix}"
            suffix += 1

        os.makedirs(self.paths.run_dir(run_id), exist_ok=False)

        text_bytes = req.text.encode("utf-8", errors="replace")
        text_sha256 = hashlib.sha256(text_bytes).hexdigest()
        preview = _safe_preview(req.text)

        result = run_text_monitor(req)
        events = result.get("events", [])
        summary = result.get("summary", {})

        if not isinstance(events, list):
            raise ValueError("run_text_monitor returned invalid events (expected list)")
        if not isinstance(summary, dict):
            raise ValueError("run_text_monitor returned invalid summary (expected object)")

        request_payload = _model_dump(req)
        request_payload.pop("text", None)
        request_payload["text_sha256"] = text_sha256
        request_payload["text_bytes"] = int(len(text_bytes))

        meta: Dict[str, Any] = {
            "schema_version": 1,
            "domain": "text",
            "run_id": run_id,
            "created_at_unix": int(created_at_unix),
            "preview": preview,
            "text_sha256": text_sha256,
            "text_bytes": int(len(text_bytes)),
            "backbone": str(req.backbone),
            "objective": str(req.objective),
            "seed": int(req.seed),
        }

        _atomic_write_json(self.paths.run_meta(run_id), meta)
        _atomic_write_json(self.paths.run_request(run_id), request_payload)
        _atomic_write_json(self.paths.run_summary(run_id), summary)
        _atomic_write_json(self.paths.run_events(run_id), [e for e in events if isinstance(e, dict)])

        with open(self.paths.run_input(run_id), "w", encoding="utf-8") as f:
            f.write(req.text)

        idx = self._load_index()
        runs = idx.get("runs", {})
        if not isinstance(runs, dict):
            runs = {}

        entry = TextRunSummary(
            run_id=run_id,
            created_at_unix=int(created_at_unix),
            preview=preview,
            summary=summary,
        )
        runs[run_id] = entry.model_dump() if hasattr(entry, "model_dump") else entry.dict()
        idx["runs"] = runs
        self._write_index(idx)

        payload = TextRunData(
            run_id=run_id,
            created_at_unix=int(created_at_unix),
            meta=meta,
            request=request_payload,
            input_text=req.text,
            summary=summary,
            events=[e for e in events if isinstance(e, dict)],
        )
        return payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
