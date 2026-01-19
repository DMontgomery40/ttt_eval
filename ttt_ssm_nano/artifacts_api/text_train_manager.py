from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ttt.text_lm.store import TextModelStore


def _now_unix() -> int:
    return int(time.time())


def _resolve_repo_file(repo_root: str, path: str) -> str:
    """
    Resolve a user-provided path to an absolute file path under repo_root.

    This keeps the training UI from becoming an arbitrary file reader.
    """
    repo_root_abs = os.path.realpath(os.path.abspath(repo_root))
    raw = path.strip()
    if not raw:
        raise ValueError("Empty corpus path")

    candidate = raw
    if not os.path.isabs(candidate):
        candidate = os.path.join(repo_root_abs, candidate)
    candidate = os.path.realpath(os.path.abspath(candidate))

    if not candidate.startswith(repo_root_abs + os.sep) and candidate != repo_root_abs:
        raise ValueError(f"Path escapes repo root: {path}")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"Corpus file not found: {path}")
    return candidate


def _resolve_repo_path(repo_root: str, path: str) -> str:
    repo_root_abs = os.path.realpath(os.path.abspath(repo_root))
    raw = path.strip()
    if not raw:
        raise ValueError("Empty corpus path")

    candidate = raw
    if not os.path.isabs(candidate):
        candidate = os.path.join(repo_root_abs, candidate)
    candidate = os.path.realpath(os.path.abspath(candidate))

    if not candidate.startswith(repo_root_abs + os.sep) and candidate != repo_root_abs:
        raise ValueError(f"Path escapes repo root: {path}")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Corpus path not found: {path}")
    return candidate


def _read_jsonl(path: str, *, limit: int = 500) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                import json

                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    if limit > 0 and len(out) > limit:
        return out[-limit:]
    return out


@dataclass
class TrainingJob:
    model_id: str
    pid: int
    started_at_unix: int
    process: subprocess.Popen
    cmd: List[str]

    def poll(self) -> Optional[int]:
        return self.process.poll()


class TextTrainManager:
    def __init__(self, *, store: TextModelStore, repo_root: str) -> None:
        self.store = store
        self.repo_root = os.path.abspath(repo_root)
        self._jobs: Dict[str, TrainingJob] = {}

    def list_jobs(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for model_id, job in list(self._jobs.items()):
            code = job.poll()
            out.append(
                {
                    "model_id": model_id,
                    "pid": job.pid,
                    "started_at_unix": job.started_at_unix,
                    "status": ("running" if code is None else "finished"),
                    "exit_code": code,
                }
            )
        out.sort(key=lambda x: int(x.get("started_at_unix", 0)), reverse=True)
        return out

    def _load_model_record(self, model_id: str) -> Optional[Dict[str, Any]]:
        try:
            import json

            if not os.path.exists(self.store.paths.index_json):
                return None
            with open(self.store.paths.index_json, "r", encoding="utf-8") as f:
                idx = json.load(f)
            if not isinstance(idx, dict):
                return None
            models = idx.get("models", {})
            if not isinstance(models, dict):
                return None
            rec = models.get(model_id)
            return rec if isinstance(rec, dict) else None
        except Exception:
            return None

    def start_training(
        self,
        *,
        corpus_paths: Sequence[str],
        tokenizer_path: Optional[str],
        tokenizer_max_lines: int,
        vocab_size: int,
        d_model: int,
        backbone: str,
        seq_len: int,
        batch_size: int,
        steps: int,
        seed: int,
        device: str,
        lr: float,
        weight_decay: float,
        momentum: float,
        ns_steps: int,
        log_every: int,
        save_every: int,
    ) -> Dict[str, Any]:
        self.store.ensure()
        model_id = self.store.new_model_id(prefix="lm")

        corpus_paths = list(corpus_paths)
        if not corpus_paths:
            corpus_paths = ["training_data"]

        corpus_files: List[str] = []
        corpus_dirs: List[str] = []
        for p in corpus_paths:
            resolved = _resolve_repo_path(self.repo_root, p)
            if os.path.isdir(resolved):
                corpus_dirs.append(resolved)
            elif os.path.isfile(resolved):
                corpus_files.append(resolved)
            else:
                raise FileNotFoundError(f"Corpus path not found: {p}")

        tok_path = None
        if tokenizer_path:
            tok_path = _resolve_repo_file(self.repo_root, tokenizer_path)

        cmd: List[str] = [
            sys.executable,
            "-m",
            "ttt.text_lm.train",
            "--artifacts_root",
            self.store.paths.artifacts_root,
            "--model_id",
            model_id,
            "--vocab_size",
            str(int(vocab_size)),
            "--tokenizer_max_lines",
            str(int(tokenizer_max_lines)),
            "--d_model",
            str(int(d_model)),
            "--backbone",
            str(backbone),
            "--seq_len",
            str(int(seq_len)),
            "--batch_size",
            str(int(batch_size)),
            "--steps",
            str(int(steps)),
            "--seed",
            str(int(seed)),
            "--device",
            str(device),
            "--lr",
            str(float(lr)),
            "--weight_decay",
            str(float(weight_decay)),
            "--momentum",
            str(float(momentum)),
            "--ns_steps",
            str(int(ns_steps)),
            "--log_every",
            str(int(log_every)),
            "--save_every",
            str(int(save_every)),
        ]
        if corpus_files:
            cmd.extend(["--corpus", *corpus_files])
        if corpus_dirs:
            cmd.extend(["--corpus_dir", *corpus_dirs])
        if tok_path:
            cmd.extend(["--tokenizer", tok_path])

        p = subprocess.Popen(
            cmd,
            cwd=self.repo_root,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        job = TrainingJob(
            model_id=model_id,
            pid=int(p.pid),
            started_at_unix=_now_unix(),
            process=p,
            cmd=cmd,
        )
        self._jobs[model_id] = job

        def _wait() -> None:
            p.wait()

        t = threading.Thread(target=_wait, daemon=True)
        t.start()

        return {"model_id": model_id, "pid": int(p.pid)}

    def cancel(self, model_id: str) -> Dict[str, Any]:
        job = self._jobs.get(model_id)
        if job is None:
            raise FileNotFoundError(f"No running job for model_id={model_id}")
        if job.poll() is not None:
            return {"model_id": model_id, "status": "finished", "exit_code": job.poll()}

        job.process.terminate()
        rec = self._load_model_record(model_id)
        if rec is not None:
            rec["status"] = "cancelled"
            rec["completed_at_unix"] = _now_unix()
            self.store.register_model(model_id, rec)
        return {"model_id": model_id, "status": "terminating"}

    def get_status(self, model_id: str) -> Dict[str, Any]:
        job = self._jobs.get(model_id)
        code = job.poll() if job else None
        rec = self._load_model_record(model_id) or {}
        phase = rec.get("status")
        if job and code is None:
            status = "running"
        elif job:
            status = str(rec.get("status", "finished"))
        else:
            status = str(rec.get("status", "unknown"))

        latest = None
        metrics = _read_jsonl(self.store.paths.train_log_jsonl(model_id), limit=1)
        if metrics:
            latest = metrics[-1]

        return {
            "model_id": model_id,
            "status": status,
            "phase": phase,
            "pid": (job.pid if job else None),
            "started_at_unix": (job.started_at_unix if job else None),
            "exit_code": (None if not job else code),
            "latest": latest,
            "error": rec.get("error"),
        }

    def get_metrics(self, model_id: str, *, limit: int = 500) -> List[Dict[str, Any]]:
        return _read_jsonl(self.store.paths.train_log_jsonl(model_id), limit=limit)
