from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from ttt.text_lm.context import ContextConfig


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _append_jsonl(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _safe_preview(text: str, *, max_len: int = 160) -> str:
    flat = " ".join(text.strip().split())
    if len(flat) <= max_len:
        return flat
    return flat[: max_len - 1] + "…"


@dataclass(frozen=True)
class TextSessionPaths:
    artifacts_root: str

    @property
    def sessions_dir(self) -> str:
        return os.path.join(self.artifacts_root, "text_sessions")

    @property
    def index_json(self) -> str:
        return os.path.join(self.sessions_dir, "index.json")

    def session_dir(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, session_id)

    def meta_json(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "meta.json")

    def context_state_pt(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "context_state.pt")

    def optim_state_pt(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "optim_state.pt")

    def events_jsonl(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "events.jsonl")


class TextSessionStore:
    def __init__(self, artifacts_root: str) -> None:
        self.paths = TextSessionPaths(artifacts_root=os.path.abspath(artifacts_root))

    def ensure(self) -> None:
        os.makedirs(self.paths.sessions_dir, exist_ok=True)
        if not os.path.exists(self.paths.index_json):
            _atomic_write_json(self.paths.index_json, {"schema_version": 1, "sessions": {}})

    def _load_index(self) -> Dict[str, Any]:
        if not os.path.exists(self.paths.index_json):
            return {"schema_version": 1, "sessions": {}}
        idx = _read_json(self.paths.index_json)
        if not isinstance(idx, dict):
            raise ValueError("Invalid text_sessions/index.json (expected object)")
        sessions = idx.get("sessions", {})
        if not isinstance(sessions, dict):
            sessions = {}
        return {"schema_version": int(idx.get("schema_version", 1) or 1), "sessions": dict(sessions)}

    def _write_index(self, idx: Dict[str, Any]) -> None:
        _atomic_write_json(self.paths.index_json, idx)

    def new_session_id(self, *, prefix: str = "chat") -> str:
        self.ensure()
        base = f"{prefix}_{int(time.time())}"
        sid = base
        suffix = 1
        while os.path.exists(self.paths.session_dir(sid)):
            sid = f"{base}_{suffix}"
            suffix += 1
        return sid

    def list_sessions(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        self.ensure()
        idx = self._load_index()
        sessions = list(idx.get("sessions", {}).values())
        sessions.sort(key=lambda r: int((r or {}).get("created_at_unix", 0)), reverse=True)
        return sessions[: max(0, int(limit))]

    def create_session(
        self,
        *,
        session_id: str,
        model_id: str,
        context_cfg: ContextConfig,
        context_state: Dict[str, Any],
        optim_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.ensure()
        sid = str(session_id).strip()
        if not sid:
            raise ValueError("Empty session_id")
        sess_dir = self.paths.session_dir(sid)
        if os.path.exists(sess_dir):
            raise FileExistsError(f"Session already exists: {sid}")
        os.makedirs(sess_dir, exist_ok=False)

        meta: Dict[str, Any] = {
            "schema_version": 1,
            "domain": "text",
            "session_id": sid,
            "created_at_unix": int(time.time()),
            "updated_at_unix": int(time.time()),
            "model_id": str(model_id),
            "context": context_cfg.to_dict(),
        }
        _atomic_write_json(self.paths.meta_json(sid), meta)
        torch.save(context_state, self.paths.context_state_pt(sid))
        torch.save(optim_state, self.paths.optim_state_pt(sid))

        idx = self._load_index()
        sessions = idx.get("sessions", {})
        if not isinstance(sessions, dict):
            sessions = {}
        sessions[sid] = {
            "session_id": sid,
            "created_at_unix": meta["created_at_unix"],
            "updated_at_unix": meta["updated_at_unix"],
            "model_id": str(model_id),
        }
        idx["sessions"] = sessions
        self._write_index(idx)
        return meta

    def load_meta(self, session_id: str) -> Dict[str, Any]:
        self.ensure()
        sid = str(session_id).strip()
        if not sid:
            raise ValueError("Empty session_id")
        path = self.paths.meta_json(sid)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Text session not found: {sid}")
        meta = _read_json(path)
        if not isinstance(meta, dict):
            raise ValueError("Invalid meta.json (expected object)")
        return meta

    def load_state(self, session_id: str) -> Dict[str, Any]:
        self.ensure()
        sid = str(session_id).strip()
        if not sid:
            raise ValueError("Empty session_id")
        if not os.path.isdir(self.paths.session_dir(sid)):
            raise FileNotFoundError(f"Text session not found: {sid}")

        ctx = torch.load(self.paths.context_state_pt(sid), map_location="cpu")
        opt = torch.load(self.paths.optim_state_pt(sid), map_location="cpu")
        if not isinstance(ctx, dict):
            raise ValueError("Invalid context_state.pt (expected dict)")
        if not isinstance(opt, dict):
            opt = {}
        return {"context_state": ctx, "optim_state": opt}

    def save_state(
        self,
        session_id: str,
        *,
        context_state: Dict[str, Any],
        optim_state: Dict[str, Any],
    ) -> None:
        sid = str(session_id).strip()
        if not sid:
            raise ValueError("Empty session_id")
        if not os.path.isdir(self.paths.session_dir(sid)):
            raise FileNotFoundError(f"Text session not found: {sid}")

        torch.save(context_state, self.paths.context_state_pt(sid))
        torch.save(optim_state, self.paths.optim_state_pt(sid))

        meta = self.load_meta(sid)
        meta["updated_at_unix"] = int(time.time())
        _atomic_write_json(self.paths.meta_json(sid), meta)

        idx = self._load_index()
        sessions = idx.get("sessions", {})
        if isinstance(sessions, dict) and sid in sessions:
            sessions[sid]["updated_at_unix"] = meta["updated_at_unix"]
            idx["sessions"] = sessions
            self._write_index(idx)

    def append_chat_event(
        self,
        session_id: str,
        *,
        prompt: str,
        response_preview: str,
        update_events: List[Dict[str, Any]],
    ) -> None:
        sid = str(session_id).strip()
        if not sid:
            raise ValueError("Empty session_id")

        pb = prompt.encode("utf-8", errors="replace")
        prompt_sha256 = hashlib.sha256(pb).hexdigest()
        rec = {
            "t_unix": int(time.time()),
            "prompt_sha256": prompt_sha256,
            "prompt_bytes": int(len(pb)),
            "prompt_preview": _safe_preview(prompt, max_len=120),
            "response_preview": _safe_preview(response_preview, max_len=120),
            "update_events": update_events,
        }
        _append_jsonl(self.paths.events_jsonl(sid), rec)

