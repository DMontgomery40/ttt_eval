from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


_RUN_ID_RE = re.compile(r"_(?P<ts>\\d{9,})_seed(?P<seed>\\d+)$")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_read_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    return _read_json(path)


def _list_dirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(
        [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    )


def _parse_run_id(run_id: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract (created_at_unix, seed) from run_id if it matches Phase 1 format."""
    m = _RUN_ID_RE.search(run_id)
    if not m:
        return None, None
    return int(m.group("ts")), int(m.group("seed"))


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def _mean_last(xs: List[float], n: int) -> float:
    if not xs:
        return 0.0
    return _mean(xs[-n:])


def _tensor_to_2d_list(t: torch.Tensor) -> List[List[float]]:
    return t.detach().cpu().to(dtype=torch.float32).tolist()  # type: ignore[return-value]


@dataclass(frozen=True)
class ArtifactPaths:
    artifacts_root: str

    @property
    def base_dir(self) -> str:
        return os.path.join(self.artifacts_root, "base")

    @property
    def sessions_dir(self) -> str:
        return os.path.join(self.artifacts_root, "sessions")

    @property
    def base_checkpoint(self) -> str:
        return os.path.join(self.base_dir, "base_checkpoint.pt")

    @property
    def base_meta(self) -> str:
        return os.path.join(self.base_dir, "base_meta.json")

    @property
    def index_json(self) -> str:
        return os.path.join(self.sessions_dir, "index.json")

    def session_dir(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, session_id)

    def session_meta(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "meta.json")

    def session_metrics(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "metrics.json")

    def session_plastic(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "plastic_state.pt")

    def session_runs_dir(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "runs")

    def run_dir(self, session_id: str, run_id: str) -> str:
        return os.path.join(self.session_runs_dir(session_id), run_id)

    def run_per_step_csv(self, session_id: str, run_id: str) -> str:
        return os.path.join(self.run_dir(session_id, run_id), "per_step.csv")

    def run_update_events_json(self, session_id: str, run_id: str) -> str:
        return os.path.join(self.run_dir(session_id, run_id), "update_events.json")


class ArtifactReader:
    def __init__(self, artifacts_root: str) -> None:
        self.paths = ArtifactPaths(artifacts_root=os.path.abspath(artifacts_root))
        self._base_meta_cache: Optional[Dict[str, Any]] = None
        self._base_weights_cache: Optional[Dict[str, List[List[float]]]] = None

    def artifacts_root(self) -> str:
        return self.paths.artifacts_root

    # -------------------------
    # Base
    # -------------------------

    def load_base_meta(self) -> Dict[str, Any]:
        if self._base_meta_cache is None:
            meta = _read_json(self.paths.base_meta)
            if not isinstance(meta, dict):
                raise ValueError("Invalid base_meta.json (expected object)")
            self._base_meta_cache = meta  # type: ignore[assignment]
        return dict(self._base_meta_cache)

    def load_base_weights(self) -> Dict[str, List[List[float]]]:
        if self._base_weights_cache is not None:
            return dict(self._base_weights_cache)

        ckpt = torch.load(self.paths.base_checkpoint, map_location="cpu")
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise ValueError("Invalid base_checkpoint.pt (expected dict with model_state)")
        state = ckpt["model_state"]
        if not isinstance(state, dict):
            raise ValueError("Invalid base_checkpoint.pt (model_state is not a dict)")

        required = ("W_u", "B", "W_o")
        missing = [k for k in required if k not in state]
        if missing:
            raise ValueError(f"Base checkpoint missing keys: {missing}")

        weights = {
            "W_u": _tensor_to_2d_list(state["W_u"]),
            "B": _tensor_to_2d_list(state["B"]),
            "W_o": _tensor_to_2d_list(state["W_o"]),
        }
        self._base_weights_cache = weights
        return dict(weights)

    # -------------------------
    # Runs
    # -------------------------

    def _read_per_step_csv(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []

        out: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue

                # Phase 1 writes: t,base_mse,session_no_update_mse,adaptive_mse,did_update,update_ok
                t = int(float(row.get("t", "0") or "0"))
                base_mse = float(row.get("base_mse", "0") or "0")
                sess_no_up = float(row.get("session_no_update_mse", "0") or "0")
                adaptive_mse = float(row.get("adaptive_mse", "0") or "0")
                did_update = bool(int(float(row.get("did_update", "0") or "0")))
                update_ok = bool(int(float(row.get("update_ok", "0") or "0")))

                out.append(
                    {
                        "t": t,
                        "base_mse": base_mse,
                        "session_start_mse": sess_no_up,  # UI uses "session_start_mse"
                        "adaptive_mse": adaptive_mse,
                        "did_update": did_update,
                        "update_ok": update_ok,
                        # Legacy alias used by some UI components
                        "baseline_mse": base_mse,
                    }
                )
        return out

    def _read_update_events(self, path: str) -> List[Dict[str, Any]]:
        blob = _maybe_read_json(path)
        if blob is None:
            return []
        if not isinstance(blob, list):
            raise ValueError(f"Invalid update_events.json at {path} (expected list)")
        out: List[Dict[str, Any]] = []
        for e in blob:
            if not isinstance(e, dict):
                continue
            out.append(
                {
                    "t": int(e.get("t", 0)),
                    "status": str(e.get("status", "")),
                    "pre_loss": float(e.get("pre_loss", 0.0)),
                    "post_loss": (None if e.get("post_loss", None) is None else float(e["post_loss"])),
                    "grad_norm": float(e.get("grad_norm", 0.0)),
                    "pre_max_h": float(e.get("pre_max_h", 0.0)),
                    "post_max_h": (None if e.get("post_max_h", None) is None else float(e["post_max_h"])),
                }
            )
        return out

    def _compute_run_metrics(
        self,
        *,
        run_id: str,
        seed: int,
        steps: int,
        mu: float,
        env_mode: str,
        per_step: List[Dict[str, Any]],
        update_events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        base = [float(p["base_mse"]) for p in per_step]
        sess = [float(p["session_start_mse"]) for p in per_step]
        adapt = [float(p["adaptive_mse"]) for p in per_step]

        commits = sum(1 for e in update_events if e.get("status") == "commit")
        rollbacks = sum(
            1 for e in update_events if str(e.get("status", "")).startswith("rollback")
        )

        metrics = {
            "run_id": run_id,
            "seed": int(seed),
            "steps": int(steps),
            "mu": float(mu),
            "env_mode": str(env_mode),
            "base_mse_mean": _mean(base),
            "session_no_update_mse_mean": _mean(sess),
            "adaptive_mse_mean": _mean(adapt),
            "base_mse_last100_mean": _mean_last(base, 100),
            "session_no_update_last100_mean": _mean_last(sess, 100),
            "adaptive_last100_mean": _mean_last(adapt, 100),
            "updates_attempted": int(len(update_events)),
            "updates_committed": int(commits),
            "updates_rolled_back": int(rollbacks),
            # Legacy aliases
            "baseline_mse_mean": _mean(base),
            "baseline_mse_last100_mean": _mean_last(base, 100),
            "adaptive_mse_last100_mean": _mean_last(adapt, 100),
        }
        return metrics

    def load_runs(self, session_id: str, *, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        run_ids = _list_dirs(self.paths.session_runs_dir(session_id))

        runs: List[Dict[str, Any]] = []
        for run_id in run_ids:
            ts, seed_from_id = _parse_run_id(run_id)
            created_at_unix = ts if ts is not None else int(
                os.path.getmtime(self.paths.run_dir(session_id, run_id))
            )

            per_step = self._read_per_step_csv(
                self.paths.run_per_step_csv(session_id, run_id)
            )
            update_events = self._read_update_events(
                self.paths.run_update_events_json(session_id, run_id)
            )

            steps = len(per_step)
            seed = seed_from_id if seed_from_id is not None else int(meta.get("seed", 0))
            mu = float(meta.get("mu", 0.0))
            env_mode = str(meta.get("env_mode", "linear"))

            metrics = self._compute_run_metrics(
                run_id=run_id,
                seed=seed,
                steps=steps,
                mu=mu,
                env_mode=env_mode,
                per_step=per_step,
                update_events=update_events,
            )

            runs.append(
                {
                    "run_id": run_id,
                    "created_at_unix": int(created_at_unix),
                    "seed": int(seed),
                    "steps": int(steps),
                    "metrics": metrics,
                    "perStep": per_step,
                    "updateEvents": update_events,
                }
            )

        runs.sort(key=lambda r: int(r.get("created_at_unix", 0)))
        return runs

    # -------------------------
    # Sessions
    # -------------------------

    def load_session_ids(self) -> List[str]:
        idx = _maybe_read_json(self.paths.index_json)
        if isinstance(idx, dict):
            sessions = idx.get("sessions", {})
            if isinstance(sessions, dict):
                return sorted([str(k) for k in sessions.keys()])

        # Fallback to directory scan
        return sorted(
            [
                name
                for name in os.listdir(self.paths.sessions_dir)
                if os.path.isdir(self.paths.session_dir(name))
            ]
        )

    def load_session_index(self) -> Dict[str, Any]:
        idx = _read_json(self.paths.index_json)
        if not isinstance(idx, dict):
            raise ValueError("Invalid sessions/index.json (expected object)")

        schema_version = int(idx.get("schema_version", 1))
        sessions_raw = idx.get("sessions", {})
        if not isinstance(sessions_raw, dict):
            raise ValueError("Invalid sessions/index.json (sessions must be an object)")

        sessions_out: Dict[str, Any] = {}
        for session_id, summary in sessions_raw.items():
            if not isinstance(summary, dict):
                continue

            meta = _maybe_read_json(self.paths.session_meta(str(session_id)))
            meta_dict = meta if isinstance(meta, dict) else {}
            runs = self.load_runs(str(session_id), meta=meta_dict)
            total_runs = len(runs)
            total_commits = sum(int(r["metrics"]["updates_committed"]) for r in runs)
            total_rollbacks = sum(int(r["metrics"]["updates_rolled_back"]) for r in runs)

            sessions_out[str(session_id)] = {
                "session_id": str(summary.get("session_id", session_id)),
                "parent_session_id": summary.get("parent_session_id", None),
                "root_session_id": summary.get("root_session_id", str(session_id)),
                "created_at_unix": int(summary.get("created_at_unix", 0) or 0),
                "last_run_at_unix": summary.get("last_run_at_unix", None),
                "env_mode": str(summary.get("env_mode", "linear")),
                "mu": float(summary.get("mu", 0.0) or 0.0),
                "model_signature": str(summary.get("model_signature", "")),
                "total_runs": int(total_runs),
                "total_updates_committed": int(total_commits),
                "total_updates_rolled_back": int(total_rollbacks),
            }

        return {"schema_version": schema_version, "sessions": sessions_out}

    def _pick_current_run(
        self, *, session_metrics: Optional[Dict[str, Any]], runs: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not runs:
            return None

        if session_metrics:
            run_id = session_metrics.get("run_id")
            if isinstance(run_id, str):
                for r in runs:
                    if r.get("run_id") == run_id:
                        return r

        return max(runs, key=lambda r: int(r.get("created_at_unix", 0)))

    def load_session_data(self, session_id: str) -> Dict[str, Any]:
        meta = _read_json(self.paths.session_meta(session_id))
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid session meta for {session_id} (expected object)")

        session_metrics = _maybe_read_json(self.paths.session_metrics(session_id))
        metrics_dict = session_metrics if isinstance(session_metrics, dict) else None

        runs = self.load_runs(session_id, meta=meta)
        current_run = self._pick_current_run(session_metrics=metrics_dict, runs=runs)

        # Current view data should follow selected/default run.
        if current_run is None:
            per_step: List[Dict[str, Any]] = []
            update_events: List[Dict[str, Any]] = []
            metrics = {
                "run_id": "",
                "seed": 0,
                "steps": 0,
                "mu": float(meta.get("mu", 0.0) or 0.0),
                "env_mode": str(meta.get("env_mode", "linear")),
                "base_mse_mean": 0.0,
                "base_mse_last100_mean": 0.0,
                "session_no_update_mse_mean": 0.0,
                "session_no_update_last100_mean": 0.0,
                "adaptive_mse_mean": 0.0,
                "adaptive_last100_mean": 0.0,
                "updates_attempted": 0,
                "updates_committed": 0,
                "updates_rolled_back": 0,
                "baseline_mse_mean": 0.0,
                "baseline_mse_last100_mean": 0.0,
                "adaptive_mse_last100_mean": 0.0,
            }
        else:
            per_step = list(current_run.get("perStep", []))
            update_events = list(current_run.get("updateEvents", []))
            metrics = dict(current_run.get("metrics", {}))

        # Weights
        plastic = torch.load(self.paths.session_plastic(session_id), map_location="cpu")
        weights = {
            "W_u": _tensor_to_2d_list(plastic["W_u"]),
            "B": _tensor_to_2d_list(plastic["B"]),
            "W_o": _tensor_to_2d_list(plastic["W_o"]),
        }

        parent_weights: Optional[Dict[str, List[List[float]]]] = None
        parent_id = meta.get("parent_session_id")
        if isinstance(parent_id, str) and parent_id:
            parent_plastic = torch.load(
                self.paths.session_plastic(parent_id), map_location="cpu"
            )
            parent_weights = {
                "W_u": _tensor_to_2d_list(parent_plastic["W_u"]),
                "B": _tensor_to_2d_list(parent_plastic["B"]),
                "W_o": _tensor_to_2d_list(parent_plastic["W_o"]),
            }

        base_weights = self.load_base_weights()

        # Global events (add run_id at read time)
        global_events: List[Dict[str, Any]] = []
        for r in runs:
            rid = str(r.get("run_id", ""))
            for e in r.get("updateEvents", []):
                ge = dict(e)
                ge["run_id"] = rid
                global_events.append(ge)

        return {
            "meta": meta,
            "metrics": metrics,
            "perStep": per_step,
            "updateEvents": update_events,
            "globalUpdateEvents": global_events,
            "runs": runs,
            "weights": weights,
            "parentWeights": parent_weights,
            "baseWeights": base_weights,
        }

