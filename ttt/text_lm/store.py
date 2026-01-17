from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


@dataclass(frozen=True)
class TextModelPaths:
    artifacts_root: str

    @property
    def text_models_dir(self) -> str:
        return os.path.join(self.artifacts_root, "text_models")

    @property
    def index_json(self) -> str:
        return os.path.join(self.text_models_dir, "index.json")

    def model_dir(self, model_id: str) -> str:
        return os.path.join(self.text_models_dir, model_id)

    def tokenizer_json(self, model_id: str) -> str:
        return os.path.join(self.model_dir(model_id), "tokenizer.json")

    def config_json(self, model_id: str) -> str:
        return os.path.join(self.model_dir(model_id), "config.json")

    def checkpoint_pt(self, model_id: str) -> str:
        return os.path.join(self.model_dir(model_id), "checkpoint.pt")

    def train_log_jsonl(self, model_id: str) -> str:
        return os.path.join(self.model_dir(model_id), "train_log.jsonl")


class TextModelStore:
    def __init__(self, artifacts_root: str) -> None:
        self.paths = TextModelPaths(artifacts_root=os.path.abspath(artifacts_root))

    def ensure(self) -> None:
        os.makedirs(self.paths.text_models_dir, exist_ok=True)
        if not os.path.exists(self.paths.index_json):
            _atomic_write_json(self.paths.index_json, {"schema_version": 1, "models": {}})

    def _load_index(self) -> Dict[str, Any]:
        if not os.path.exists(self.paths.index_json):
            return {"schema_version": 1, "models": {}}
        idx = _read_json(self.paths.index_json)
        if not isinstance(idx, dict):
            raise ValueError("Invalid text_models/index.json (expected object)")
        models = idx.get("models", {})
        if not isinstance(models, dict):
            models = {}
        return {"schema_version": int(idx.get("schema_version", 1) or 1), "models": dict(models)}

    def _write_index(self, idx: Dict[str, Any]) -> None:
        _atomic_write_json(self.paths.index_json, idx)

    def list_models(self) -> List[Dict[str, Any]]:
        self.ensure()
        idx = self._load_index()
        models = list(idx.get("models", {}).values())
        models.sort(key=lambda m: int((m or {}).get("created_at_unix", 0)), reverse=True)
        return models

    def latest_model_id(self) -> Optional[str]:
        models = self.list_models()
        if not models:
            return None
        return str(models[0].get("model_id") or "")

    def new_model_id(self, *, prefix: str = "lm") -> str:
        self.ensure()
        base = f"{prefix}_{int(time.time())}"
        model_id = base
        suffix = 1
        while os.path.exists(self.paths.model_dir(model_id)):
            model_id = f"{base}_{suffix}"
            suffix += 1
        return model_id

    def register_model(self, model_id: str, record: Dict[str, Any]) -> None:
        self.ensure()
        idx = self._load_index()
        models = idx.get("models", {})
        if not isinstance(models, dict):
            models = {}
        models[str(model_id)] = dict(record)
        idx["models"] = models
        self._write_index(idx)

