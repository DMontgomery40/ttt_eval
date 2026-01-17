#!/usr/bin/env python3
"""
Phase 1: Branching sessions ("git for plastic weights") for TTT-SSM using Muon.

What this adds over Phase 0:
- Persistent artifacts root:
    artifacts/base/base_checkpoint.pt
    artifacts/base/base_meta.json
    artifacts/sessions/index.json
    artifacts/sessions/<session_id>/{meta.json, plastic_state.pt, optim_state.pt, runs/...}

- Branching:
    fork_session clones the parent's *current* plastic weights (+ optionally optimizer momentum)
    and records parent_session_id + root_session_id.

- Session runs:
    run_session produces:
      baseline_base_no_update
      baseline_session_no_update
      adaptive_session_with_updates
    on the SAME trajectory (same actions, same μ, same seed), so differences are meaningful.

Constraints (by design for Phase 1):
- Encoder is identity and frozen.
- Only 2D matrices are plastic: W_u, B, W_o.
- Stability parameter a_raw is frozen: A = -softplus(a_raw).

This script uses torch.optim.Muon if available; otherwise it uses MuonFallback.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

SCHEMA_VERSION = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_load(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(obj: object, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def device_auto() -> torch.device:
    # Tiny toy. CPU avoids MPS edge cases and is deterministic-ish.
    return torch.device("cpu")


def now_unix() -> int:
    return int(time.time())


# -----------------------------
# Muon (fallback)
# -----------------------------

def newton_schulz_orthogonalize(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    ns_coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> torch.Tensor:
    """
    Approximate orthogonalization for a 2D matrix using Newton–Schulz iteration.
    """
    assert G.ndim == 2, "Muon expects 2D matrices"
    a, b, c = ns_coefficients
    X = G.to(dtype=torch.float32)
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class MuonFallback(torch.optim.Optimizer):
    """
    Minimal Muon optimizer fallback (only supports 2D params).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        eps: float = 1e-7,
        ns_steps: int = 5,
        ns_coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        adjust_lr_fn: Optional[str] = None,  # "original" | "match_rms_adamw" | None
    ):
        if adjust_lr_fn not in (None, "original", "match_rms_adamw"):
            raise ValueError("adjust_lr_fn must be None, 'original', or 'match_rms_adamw'")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            eps=eps,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"MuonFallback only supports 2D params, got {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            mu = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            eps = float(group["eps"])
            ns_steps = int(group["ns_steps"])
            ns_coefficients = tuple(group["ns_coefficients"])
            adjust_lr_fn = group.get("adjust_lr_fn", None)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()

                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p)
                buf = st["momentum_buffer"]
                buf.mul_(mu).add_(g)

                if nesterov:
                    upd = g.add(buf, alpha=mu)
                else:
                    upd = buf

                O = newton_schulz_orthogonalize(upd, steps=ns_steps, eps=eps, ns_coefficients=ns_coefficients)

                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

                lr_eff = lr
                if adjust_lr_fn == "original":
                    m, n = p.shape
                    lr_eff = lr * max(1.0, float(max(m, n)) / float(min(m, n)))
                elif adjust_lr_fn == "match_rms_adamw":
                    m, n = p.shape
                    lr_eff = lr * 0.2 * math.sqrt(float(max(m, n)))

                p.add_(O, alpha=-lr_eff)

        return loss


def make_muon_optimizer(
    params,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    adjust_lr_fn: Optional[str],
):
    if hasattr(torch.optim, "Muon"):
        return torch.optim.Muon(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
    return MuonFallback(
        params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        adjust_lr_fn=adjust_lr_fn,
    )


# -----------------------------
# Model + Env
# -----------------------------

@dataclass(frozen=True)
class ModelConfig:
    obs_dim: int = 4
    act_dim: int = 2
    z_dim: int = 4
    u_dim: int = 16
    n_state: int = 32
    dt: float = 1.0


class DiagStableSSM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.z_dim + cfg.act_dim

        self.W_u = nn.Parameter(torch.randn(cfg.u_dim, d_in) * 0.02)
        self.B = nn.Parameter(torch.randn(cfg.n_state, cfg.u_dim) * 0.02)
        self.W_o = nn.Parameter(torch.randn(cfg.z_dim, cfg.n_state) * 0.02)

        self.a_raw = nn.Parameter(torch.randn(cfg.n_state) * 0.02)

    def freeze_stability(self) -> None:
        self.a_raw.requires_grad_(False)

    def forward_step(self, z_t: torch.Tensor, a_t: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A = -F.softplus(self.a_raw)
        decay = torch.exp(A * self.cfg.dt).unsqueeze(0)

        x = torch.cat([z_t, a_t], dim=-1)
        u = x @ self.W_u.T
        h_next = decay * h_t + (u @ self.B.T)
        z_pred_next = h_next @ self.W_o.T
        return z_pred_next, h_next


@dataclass
class EnvConfig:
    dt: float = 1.0
    noise_std: float = 0.0
    mu_min: float = 0.02
    mu_max: float = 0.25

    nonlinear: bool = False
    threshold_scale: float = 2.0
    static_mult: float = 2.0
    dynamic_mult: float = 0.5


class HiddenMuPhysicsEnv:
    def __init__(self, mu: float, cfg: EnvConfig):
        self.mu = float(mu)
        self.cfg = cfg
        self.pos = torch.zeros(2)
        self.vel = torch.zeros(2)

    def reset(self) -> torch.Tensor:
        self.pos.zero_()
        self.vel.zero_()
        return self.observe()

    def observe(self) -> torch.Tensor:
        return torch.cat([self.pos, self.vel], dim=0)

    def step(self, accel: torch.Tensor) -> torch.Tensor:
        accel = accel.to(dtype=torch.float32).view(2)

        if not self.cfg.nonlinear:
            mu_eff = self.mu
        else:
            speed = float(torch.linalg.norm(self.vel).item())
            thresh = self.mu * self.cfg.threshold_scale
            mu_static = min(0.95, self.mu * self.cfg.static_mult)
            mu_dynamic = max(0.0, self.mu * self.cfg.dynamic_mult)
            mu_eff = mu_static if speed < thresh else mu_dynamic

        self.vel = (1.0 - mu_eff) * self.vel + accel
        self.pos = self.pos + self.vel * self.cfg.dt

        if self.cfg.noise_std > 0:
            self.pos = self.pos + torch.randn_like(self.pos) * self.cfg.noise_std
            self.vel = self.vel + torch.randn_like(self.vel) * self.cfg.noise_std

        return self.observe()


@dataclass
class PlasticityConfig:
    lr: float = 5e-3
    weight_decay: float = 0.0
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    adjust_lr_fn: Optional[str] = None

    chunk: int = 32
    buffer_len: int = 32

    rollback_tol: float = 0.20
    grad_norm_max: float = 20.0
    state_norm_max: float = 1e6


class TransitionBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = int(maxlen)
        self.z_t: List[torch.Tensor] = []
        self.a_t: List[torch.Tensor] = []
        self.z_next: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.z_t)

    def append(self, z_t: torch.Tensor, a_t: torch.Tensor, z_next: torch.Tensor) -> None:
        if len(self.z_t) >= self.maxlen:
            self.z_t.pop(0)
            self.a_t.pop(0)
            self.z_next.pop(0)
        self.z_t.append(z_t.detach().cpu())
        self.a_t.append(a_t.detach().cpu())
        self.z_next.append(z_next.detach().cpu())

    def get(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_t = torch.stack(self.z_t, dim=0).to(device)
        a_t = torch.stack(self.a_t, dim=0).to(device)
        z_next = torch.stack(self.z_next, dim=0).to(device)
        return z_t, a_t, z_next


def grad_global_norm(params: List[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(torch.sum(p.grad.detach() ** 2).item())
    return math.sqrt(total)


def snapshot_params(params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def restore_params(params: List[torch.nn.Parameter], snap: List[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, s in zip(params, snap):
            p.copy_(s)


@torch.no_grad()
def rollout_loss_on_buffer(model: DiagStableSSM, z_t: torch.Tensor, a_t: torch.Tensor, z_next: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, float]:
    model.eval()
    T = z_t.shape[0]
    h = torch.zeros(1, model.cfg.n_state, device=device)
    preds: List[torch.Tensor] = []
    max_h = 0.0
    for i in range(T):
        z_i = z_t[i].unsqueeze(0)
        a_i = a_t[i].unsqueeze(0)
        z_pred, h = model.forward_step(z_i, a_i, h)
        preds.append(z_pred.squeeze(0))
        hn = float(torch.linalg.norm(h).item())
        if hn > max_h:
            max_h = hn
    pred = torch.stack(preds, dim=0)
    loss = F.mse_loss(pred, z_next)
    return loss, max_h


@torch.no_grad()
def eval_no_update_mse(model: DiagStableSSM, obs: torch.Tensor, actions: torch.Tensor, device: torch.device) -> List[float]:
    model.eval()
    steps = actions.shape[0]
    h = torch.zeros(1, model.cfg.n_state, device=device)
    errs: List[float] = []
    for t in range(steps):
        z_t = obs[t].to(device).unsqueeze(0)
        a_t = actions[t].to(device).unsqueeze(0)
        z_true_next = obs[t + 1].to(device).unsqueeze(0)
        z_pred_next, h = model.forward_step(z_t, a_t, h)
        errs.append(float(F.mse_loss(z_pred_next, z_true_next).item()))
    return errs


def make_model_signature(cfg: ModelConfig, base_ckpt_hash: str) -> str:
    cfg_json = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    h = hashlib.sha256()
    h.update(cfg_json.encode("utf-8"))
    h.update(base_ckpt_hash.encode("utf-8"))
    return h.hexdigest()


# -----------------------------
# Artifact store (base + sessions)
# -----------------------------

class ArtifactStore:
    def __init__(self, root: str):
        self.root = root
        self.base_dir = os.path.join(root, "base")
        self.sessions_dir = os.path.join(root, "sessions")
        self.base_ckpt = os.path.join(self.base_dir, "base_checkpoint.pt")
        self.base_meta = os.path.join(self.base_dir, "base_meta.json")
        self.index_path = os.path.join(self.sessions_dir, "index.json")

    def ensure(self) -> None:
        ensure_dir(self.base_dir)
        ensure_dir(self.sessions_dir)
        if not os.path.exists(self.index_path):
            json_dump({"schema_version": SCHEMA_VERSION, "sessions": {}}, self.index_path)

    def load_index(self) -> Dict[str, object]:
        self.ensure()
        return json_load(self.index_path)  # type: ignore

    def save_index(self, index: Dict[str, object]) -> None:
        self.ensure()
        json_dump(index, self.index_path)

    def upsert_index_entry(self, session_id: str, entry: Dict[str, object]) -> None:
        index = self.load_index()
        sessions = index.get("sessions", {})
        sessions[session_id] = entry
        index["sessions"] = sessions
        self.save_index(index)

    def session_dir(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, session_id)

    def session_meta_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "meta.json")

    def session_plastic_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "plastic_state.pt")

    def session_optim_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "optim_state.pt")

    def session_runs_dir(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "runs")

    def session_update_log_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "update_events.jsonl")

    def session_metrics_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "metrics.json")

    # ---- base ----

    def base_exists(self) -> bool:
        return os.path.exists(self.base_ckpt) and os.path.exists(self.base_meta)

    def save_base(self, model: DiagStableSSM, cfg: ModelConfig) -> None:
        self.ensure()
        torch.save({"model_state": model.state_dict(), "cfg": dataclasses.asdict(cfg)}, self.base_ckpt)
        base_hash = sha256_file(self.base_ckpt)
        sig = make_model_signature(cfg, base_hash)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "created_at_unix": now_unix(),
            "torch_version": torch.__version__,
            "base_ckpt_hash": base_hash,
            "model_signature": sig,
            "model_cfg": dataclasses.asdict(cfg),
        }
        json_dump(meta, self.base_meta)

    def load_base(self, device: torch.device) -> Tuple[DiagStableSSM, ModelConfig, Dict[str, object]]:
        if not self.base_exists():
            raise FileNotFoundError(f"Base not initialized. Missing: {self.base_ckpt} or {self.base_meta}")
        blob = torch.load(self.base_ckpt, map_location=device)
        cfg = ModelConfig(**blob["cfg"])
        model = DiagStableSSM(cfg).to(device)
        model.load_state_dict(blob["model_state"])
        model.freeze_stability()
        meta = json_load(self.base_meta)  # type: ignore
        return model, cfg, meta  # type: ignore

    # ---- sessions ----

    def session_exists(self, session_id: str) -> bool:
        return os.path.exists(self.session_meta_path(session_id)) and os.path.exists(self.session_plastic_path(session_id)) and os.path.exists(self.session_optim_path(session_id))

    def load_session_meta(self, session_id: str) -> Dict[str, object]:
        return json_load(self.session_meta_path(session_id))  # type: ignore

    def load_session_state(self, session_id: str, device: torch.device) -> Tuple[Dict[str, object], Dict[str, torch.Tensor], Dict[str, object]]:
        meta = self.load_session_meta(session_id)
        plastic = torch.load(self.session_plastic_path(session_id), map_location=device)
        optim_state = torch.load(self.session_optim_path(session_id), map_location="cpu")
        return meta, plastic, optim_state  # type: ignore

    def save_session_state(self, session_id: str, meta: Dict[str, object], plastic: Dict[str, torch.Tensor], optim_state: Dict[str, object], metrics: Dict[str, object]) -> None:
        sdir = self.session_dir(session_id)
        ensure_dir(sdir)
        torch.save(plastic, self.session_plastic_path(session_id))
        torch.save(optim_state, self.session_optim_path(session_id))
        json_dump(meta, self.session_meta_path(session_id))
        json_dump(metrics, self.session_metrics_path(session_id))

        entry = {
            "session_id": session_id,
            "parent_session_id": meta.get("parent_session_id"),
            "root_session_id": meta.get("root_session_id"),
            "created_at_unix": meta.get("created_at_unix"),
            "last_run_at_unix": meta.get("last_run_at_unix"),
            "env_mode": meta.get("env_mode"),
            "mu": meta.get("mu"),
            "model_signature": meta.get("model_signature"),
        }
        self.upsert_index_entry(session_id, entry)

    def append_update_events(self, session_id: str, events: List[Dict[str, object]]) -> None:
        path = self.session_update_log_path(session_id)
        ensure_dir(os.path.dirname(path))
        with open(path, "a", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e, sort_keys=True) + "\n")


# -----------------------------
# Base pretrain (optional)
# -----------------------------

def pretrain_base(
    store: ArtifactStore,
    cfg: ModelConfig,
    env_cfg: EnvConfig,
    p_cfg: PlasticityConfig,
    steps: int,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> None:
    device = device_auto()
    set_seed(seed)

    model = DiagStableSSM(cfg).to(device)
    model.freeze_stability()

    plastic_params = [model.W_u, model.B, model.W_o]
    opt = make_muon_optimizer(
        plastic_params,
        lr=p_cfg.lr,
        weight_decay=p_cfg.weight_decay,
        momentum=p_cfg.momentum,
        nesterov=p_cfg.nesterov,
        ns_steps=p_cfg.ns_steps,
        adjust_lr_fn=p_cfg.adjust_lr_fn,
    )

    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        a_seq = torch.randn(batch_size, seq_len, cfg.act_dim, device=device) * 0.5
        mus = torch.empty(batch_size).uniform_(env_cfg.mu_min, env_cfg.mu_max).tolist()

        pos = torch.zeros(batch_size, 2, device=device)
        vel = torch.zeros(batch_size, 2, device=device)
        h = torch.zeros(batch_size, cfg.n_state, device=device)

        losses: List[torch.Tensor] = []
        for t in range(seq_len):
            obs = torch.cat([pos, vel], dim=-1)
            z_t = obs
            a_t = a_seq[:, t, :]
            z_pred_next, h = model.forward_step(z_t, a_t, h)

            mu_t = torch.tensor(mus, device=device).view(batch_size, 1)
            if not env_cfg.nonlinear:
                mu_eff = mu_t
            else:
                speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
                thresh = mu_t * env_cfg.threshold_scale
                mu_static = torch.clamp(mu_t * env_cfg.static_mult, max=0.95)
                mu_dynamic = torch.clamp(mu_t * env_cfg.dynamic_mult, min=0.0)
                mu_eff = torch.where(speed < thresh, mu_static, mu_dynamic)

            vel = (1.0 - mu_eff) * vel + a_t
            pos = pos + vel * env_cfg.dt
            obs_next = torch.cat([pos, vel], dim=-1)
            z_next = obs_next
            losses.append(F.mse_loss(z_pred_next, z_next))

        loss = torch.stack(losses).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == 1 or step == steps:
            dt = time.time() - t0
            print(f"[pretrain] step={step}/{steps} loss={loss.item():.6f} elapsed_s={dt:.1f}")

    store.save_base(model, cfg)
    meta = json_load(store.base_meta)
    print(f"[base] saved: {store.base_ckpt}")
    print(f"[base] model_signature: {meta['model_signature']}")


# -----------------------------
# Session creation / forking
# -----------------------------

def init_or_verify_base(store: ArtifactStore, device: torch.device) -> Tuple[DiagStableSSM, ModelConfig, Dict[str, object]]:
    store.ensure()
    return store.load_base(device)


def create_new_session(
    store: ArtifactStore,
    session_id: str,
    mu: float,
    env_mode: str,
    p_cfg: PlasticityConfig,
    parent_session_id: Optional[str],
    fork_copy_optimizer: bool,
    fork_reset_optimizer: bool,
) -> None:
    device = device_auto()
    base_model, cfg, base_meta = init_or_verify_base(store, device)
    base_sig = base_meta["model_signature"]
    base_hash = base_meta["base_ckpt_hash"]

    if store.session_exists(session_id):
        raise FileExistsError(f"Session already exists: {session_id}")

    if parent_session_id is None:
        root_session_id = session_id
        parent_plastic = {
            "W_u": base_model.W_u.detach().cpu(),
            "B": base_model.B.detach().cpu(),
            "W_o": base_model.W_o.detach().cpu(),
        }
        # fresh optimizer state (zero momentum)
        tmp_model = DiagStableSSM(cfg)
        tmp_model.freeze_stability()
        tmp_model.W_u.data.copy_(parent_plastic["W_u"])
        tmp_model.B.data.copy_(parent_plastic["B"])
        tmp_model.W_o.data.copy_(parent_plastic["W_o"])
        opt = make_muon_optimizer(
            [tmp_model.W_u, tmp_model.B, tmp_model.W_o],
            lr=p_cfg.lr,
            weight_decay=p_cfg.weight_decay,
            momentum=p_cfg.momentum,
            nesterov=p_cfg.nesterov,
            ns_steps=p_cfg.ns_steps,
            adjust_lr_fn=p_cfg.adjust_lr_fn,
        )
        optim_state = opt.state_dict()
    else:
        if not store.session_exists(parent_session_id):
            raise FileNotFoundError(f"Parent session does not exist: {parent_session_id}")
        parent_meta, parent_plastic, parent_optim = store.load_session_state(parent_session_id, device=device)
        if parent_meta.get("model_signature") != base_sig:
            raise ValueError("Parent session is incompatible with current base (model_signature mismatch).")
        root_session_id = str(parent_meta.get("root_session_id", parent_session_id))

        parent_plastic = {k: v.detach().cpu() for k, v in parent_plastic.items()}

        if fork_reset_optimizer:
            tmp_model = DiagStableSSM(cfg)
            tmp_model.freeze_stability()
            tmp_model.W_u.data.copy_(parent_plastic["W_u"])
            tmp_model.B.data.copy_(parent_plastic["B"])
            tmp_model.W_o.data.copy_(parent_plastic["W_o"])
            opt = make_muon_optimizer(
                [tmp_model.W_u, tmp_model.B, tmp_model.W_o],
                lr=p_cfg.lr,
                weight_decay=p_cfg.weight_decay,
                momentum=p_cfg.momentum,
                nesterov=p_cfg.nesterov,
                ns_steps=p_cfg.ns_steps,
                adjust_lr_fn=p_cfg.adjust_lr_fn,
            )
            optim_state = opt.state_dict()
        else:
            optim_state = parent_optim if fork_copy_optimizer else {}
            if not fork_copy_optimizer:
                tmp_model = DiagStableSSM(cfg)
                tmp_model.freeze_stability()
                tmp_model.W_u.data.copy_(parent_plastic["W_u"])
                tmp_model.B.data.copy_(parent_plastic["B"])
                tmp_model.W_o.data.copy_(parent_plastic["W_o"])
                opt = make_muon_optimizer(
                    [tmp_model.W_u, tmp_model.B, tmp_model.W_o],
                    lr=p_cfg.lr,
                    weight_decay=p_cfg.weight_decay,
                    momentum=p_cfg.momentum,
                    nesterov=p_cfg.nesterov,
                    ns_steps=p_cfg.ns_steps,
                    adjust_lr_fn=p_cfg.adjust_lr_fn,
                )
                optim_state = opt.state_dict()

    meta = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "created_at_unix": now_unix(),
        "last_run_at_unix": None,
        "parent_session_id": parent_session_id,
        "root_session_id": root_session_id,
        "env_mode": env_mode,
        "mu": float(mu),
        "base_ckpt_hash": base_hash,
        "model_signature": base_sig,
        "model_cfg": dataclasses.asdict(cfg),
        "plasticity_cfg": dataclasses.asdict(p_cfg),
    }

    metrics = {
        "notes": "session created",
    }

    store.save_session_state(
        session_id=session_id,
        meta=meta,
        plastic=parent_plastic,
        optim_state=optim_state,
        metrics=metrics,
    )

    print(f"[session] created: {session_id}")
    print(f"[session] parent: {parent_session_id}")
    print(f"[session] root: {root_session_id}")
    print(f"[session] mu: {mu:.6f} env_mode: {env_mode}")


# -----------------------------
# Run a session (baseline vs adaptive)
# -----------------------------

def run_session(
    store: ArtifactStore,
    session_id: str,
    steps: int,
    seed: int,
    p_cfg_override: Optional[PlasticityConfig],
    chunk: Optional[int],
    buffer_len: Optional[int],
    rollback_tol: Optional[float],
    grad_norm_max: Optional[float],
    state_norm_max: Optional[float],
) -> None:
    device = device_auto()
    set_seed(seed)

    base_model, cfg, base_meta = init_or_verify_base(store, device)
    if not store.session_exists(session_id):
        raise FileNotFoundError(f"Session does not exist: {session_id}")

    meta, plastic, optim_state = store.load_session_state(session_id, device=device)
    if meta.get("model_signature") != base_meta.get("model_signature"):
        raise ValueError("Session incompatible with current base (model_signature mismatch).")

    # Plasticity config: session default unless overridden
    p_cfg = PlasticityConfig(**meta["plasticity_cfg"])  # type: ignore
    if p_cfg_override is not None:
        p_cfg = p_cfg_override

    # Inline overrides (handy for experiments)
    if chunk is not None:
        p_cfg.chunk = int(chunk)
    if buffer_len is not None:
        p_cfg.buffer_len = int(buffer_len)
    if rollback_tol is not None:
        p_cfg.rollback_tol = float(rollback_tol)
    if grad_norm_max is not None:
        p_cfg.grad_norm_max = float(grad_norm_max)
    if state_norm_max is not None:
        p_cfg.state_norm_max = float(state_norm_max)

    env_cfg = EnvConfig(nonlinear=(str(meta.get("env_mode")) == "nonlinear"))
    mu = float(meta.get("mu"))

    # Deterministic action sequence for this run
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 12345)
    actions = torch.randn(steps, cfg.act_dim, generator=gen) * 0.5

    env = HiddenMuPhysicsEnv(mu=mu, cfg=env_cfg)
    obs0 = env.reset()
    obs = [obs0.clone()]
    for t in range(steps):
        obs.append(env.step(actions[t]))
    obs = torch.stack(obs, dim=0)

    # Baseline 1: base weights, no updates
    base_eval = DiagStableSSM(cfg).to(device)
    base_eval.load_state_dict(base_model.state_dict())
    base_eval.freeze_stability()
    base_mse = eval_no_update_mse(base_eval, obs, actions, device=device)

    # Baseline 2: session starting weights, no updates
    sess_start = DiagStableSSM(cfg).to(device)
    sess_start.load_state_dict(base_model.state_dict())
    sess_start.freeze_stability()
    with torch.no_grad():
        sess_start.W_u.copy_(plastic["W_u"].to(device))
        sess_start.B.copy_(plastic["B"].to(device))
        sess_start.W_o.copy_(plastic["W_o"].to(device))
    sess_mse = eval_no_update_mse(sess_start, obs, actions, device=device)

    # Adaptive: start from same session weights, update online
    model = DiagStableSSM(cfg).to(device)
    model.load_state_dict(base_model.state_dict())
    model.freeze_stability()
    with torch.no_grad():
        model.W_u.copy_(plastic["W_u"].to(device))
        model.B.copy_(plastic["B"].to(device))
        model.W_o.copy_(plastic["W_o"].to(device))

    plastic_params = [model.W_u, model.B, model.W_o]
    opt = make_muon_optimizer(
        plastic_params,
        lr=p_cfg.lr,
        weight_decay=p_cfg.weight_decay,
        momentum=p_cfg.momentum,
        nesterov=p_cfg.nesterov,
        ns_steps=p_cfg.ns_steps,
        adjust_lr_fn=p_cfg.adjust_lr_fn,
    )
    # Load optimizer state (momentum persistence)
    try:
        opt.load_state_dict(optim_state)
    except Exception:
        # If state is missing or incompatible, start fresh
        pass

    h = torch.zeros(1, cfg.n_state, device=device)
    adaptive_mse: List[float] = []
    update_events: List[Dict[str, object]] = []
    buf = TransitionBuffer(maxlen=p_cfg.buffer_len)

    run_id = f"{session_id}_{now_unix()}_seed{seed}"
    run_dir = os.path.join(store.session_runs_dir(session_id), run_id)
    ensure_dir(run_dir)

    per_step_csv = os.path.join(run_dir, "per_step.csv")
    with open(per_step_csv, "w", encoding="utf-8") as f:
        f.write("t,base_mse,session_no_update_mse,adaptive_mse,did_update,update_ok\n")
        for t in range(steps):
            z_t = obs[t].to(device).unsqueeze(0)
            a_t = actions[t].to(device).unsqueeze(0)
            z_true_next = obs[t + 1].to(device).unsqueeze(0)

            z_pred_next, h = model.forward_step(z_t, a_t, h)
            step_mse = float(F.mse_loss(z_pred_next, z_true_next).item())
            adaptive_mse.append(step_mse)

            buf.append(obs[t], actions[t], obs[t + 1])

            did_update = False
            update_ok = False

            if (t + 1) % p_cfg.chunk == 0 and len(buf) == p_cfg.buffer_len:
                did_update = True

                z_buf, a_buf, z_next_buf = buf.get(device)
                pre_loss, pre_max_h = rollout_loss_on_buffer(model, z_buf, a_buf, z_next_buf, device=device)

                snap_w = snapshot_params(plastic_params)
                snap_opt = copy.deepcopy(opt.state_dict())

                model.train()
                opt.zero_grad(set_to_none=True)

                h_train = torch.zeros(1, cfg.n_state, device=device)
                preds: List[torch.Tensor] = []
                for i in range(z_buf.shape[0]):
                    z_i = z_buf[i].unsqueeze(0)
                    a_i = a_buf[i].unsqueeze(0)
                    z_pred, h_train = model.forward_step(z_i, a_i, h_train)
                    preds.append(z_pred.squeeze(0))
                pred = torch.stack(preds, dim=0)
                loss = F.mse_loss(pred, z_next_buf)
                loss.backward()

                gnorm = grad_global_norm(plastic_params)
                if gnorm > p_cfg.grad_norm_max:
                    restore_params(plastic_params, snap_w)
                    opt.load_state_dict(snap_opt)
                    update_ok = False
                    update_events.append(
                        dict(
                            t=t,
                            status="rollback_grad_norm",
                            pre_loss=float(pre_loss.item()),
                            post_loss=None,
                            grad_norm=gnorm,
                            pre_max_h=pre_max_h,
                        )
                    )
                else:
                    opt.step()
                    post_loss, post_max_h = rollout_loss_on_buffer(model, z_buf, a_buf, z_next_buf, device=device)

                    rollback = False
                    reason = None
                    if float(post_loss.item()) > float(pre_loss.item()) * (1.0 + p_cfg.rollback_tol):
                        rollback = True
                        reason = "rollback_loss_regression"
                    elif post_max_h > p_cfg.state_norm_max:
                        rollback = True
                        reason = "rollback_state_norm"

                    if rollback:
                        restore_params(plastic_params, snap_w)
                        opt.load_state_dict(snap_opt)
                        update_ok = False
                        update_events.append(
                            dict(
                                t=t,
                                status=reason,
                                pre_loss=float(pre_loss.item()),
                                post_loss=float(post_loss.item()),
                                grad_norm=gnorm,
                                pre_max_h=pre_max_h,
                                post_max_h=post_max_h,
                            )
                        )
                    else:
                        update_ok = True
                        update_events.append(
                            dict(
                                t=t,
                                status="commit",
                                pre_loss=float(pre_loss.item()),
                                post_loss=float(post_loss.item()),
                                grad_norm=gnorm,
                                pre_max_h=pre_max_h,
                                post_max_h=post_max_h,
                            )
                        )

                model.eval()

            f.write(
                f"{t},{base_mse[t]},{sess_mse[t]},{adaptive_mse[t]},{int(did_update)},{int(update_ok)}\n"
            )

    # Save update events for this run
    run_events_path = os.path.join(run_dir, "update_events.json")
    json_dump(update_events, run_events_path)

    # Update persistent session log (append-only)
    store.append_update_events(session_id, update_events)

    # Plot
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(base_mse, label="base (no updates)")
        plt.plot(sess_mse, label="session start (no updates)")
        plt.plot(adaptive_mse, label="session start (online updates)")
        plt.xlabel("timestep")
        plt.ylabel("MSE(pred z_next, true z_next)")
        plt.title(f"Phase1 Branching | session={session_id} | mu={mu:.4f} | env={meta.get('env_mode')}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "mse_curve.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    # Persist updated session weights + optimizer state
    new_plastic = {
        "W_u": model.W_u.detach().cpu(),
        "B": model.B.detach().cpu(),
        "W_o": model.W_o.detach().cpu(),
    }
    new_optim_state = opt.state_dict()

    # Metrics summary
    def mean(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    metrics = {
        "run_id": run_id,
        "seed": seed,
        "steps": steps,
        "mu": mu,
        "env_mode": meta.get("env_mode"),
        "base_mse_mean": mean(base_mse),
        "session_no_update_mse_mean": mean(sess_mse),
        "adaptive_mse_mean": mean(adaptive_mse),
        "base_mse_last100_mean": mean(base_mse[-100:]),
        "session_no_update_last100_mean": mean(sess_mse[-100:]),
        "adaptive_last100_mean": mean(adaptive_mse[-100:]),
        "updates_attempted": int(len(update_events)),
        "updates_committed": int(sum(1 for e in update_events if e.get("status") == "commit")),
        "updates_rolled_back": int(sum(1 for e in update_events if str(e.get("status", "")).startswith("rollback"))),
        "run_dir": run_dir,
    }

    meta["last_run_at_unix"] = now_unix()
    store.save_session_state(
        session_id=session_id,
        meta=meta,
        plastic=new_plastic,
        optim_state=new_optim_state,
        metrics=metrics,
    )

    print(f"[run] session={session_id} run_id={run_id}")
    print(f"[run] run_dir={run_dir}")
    print(f"[metrics] {json.dumps(metrics, indent=2)}")


def list_sessions(store: ArtifactStore) -> None:
    idx = store.load_index()
    sessions = idx.get("sessions", {})
    print(json.dumps(sessions, indent=2, sort_keys=True))


def show_session(store: ArtifactStore, session_id: str) -> None:
    if not store.session_exists(session_id):
        raise FileNotFoundError(f"Session does not exist: {session_id}")
    meta = store.load_session_meta(session_id)
    print(json.dumps(meta, indent=2, sort_keys=True))


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_root", type=str, default="artifacts")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init_base")
    p_init.add_argument("--pretrain_steps", type=int, default=2000)
    p_init.add_argument("--pretrain_batch", type=int, default=32)
    p_init.add_argument("--pretrain_seq", type=int, default=32)
    p_init.add_argument("--seed", type=int, default=1337)
    p_init.add_argument("--env_mode", type=str, choices=["linear", "nonlinear"], default="linear")
    p_init.add_argument("--u_dim", type=int, default=16)
    p_init.add_argument("--n_state", type=int, default=32)
    p_init.add_argument("--lr", type=float, default=5e-3)
    p_init.add_argument("--momentum", type=float, default=0.95)
    p_init.add_argument("--ns_steps", type=int, default=5)
    p_init.add_argument("--adjust_lr_fn", type=str, default="none", choices=["none", "original", "match_rms_adamw"])

    p_new = sub.add_parser("new_session")
    p_new.add_argument("--session_id", type=str, required=True)
    p_new.add_argument("--mu", type=float, default=-1.0)
    p_new.add_argument("--env_mode", type=str, choices=["linear", "nonlinear"], default="linear")
    p_new.add_argument("--seed", type=int, default=1337)
    p_new.add_argument("--lr", type=float, default=5e-3)
    p_new.add_argument("--momentum", type=float, default=0.95)
    p_new.add_argument("--ns_steps", type=int, default=5)
    p_new.add_argument("--chunk", type=int, default=32)
    p_new.add_argument("--buffer_len", type=int, default=32)
    p_new.add_argument("--rollback_tol", type=float, default=0.20)
    p_new.add_argument("--grad_norm_max", type=float, default=20.0)
    p_new.add_argument("--state_norm_max", type=float, default=1e6)
    p_new.add_argument("--adjust_lr_fn", type=str, default="none", choices=["none", "original", "match_rms_adamw"])

    p_fork = sub.add_parser("fork_session")
    p_fork.add_argument("--parent_session_id", type=str, required=True)
    p_fork.add_argument("--child_session_id", type=str, required=True)
    p_fork.add_argument("--copy_optimizer", type=int, default=1)
    p_fork.add_argument("--reset_optimizer", type=int, default=0)

    p_run = sub.add_parser("run_session")
    p_run.add_argument("--session_id", type=str, required=True)
    p_run.add_argument("--steps", type=int, default=600)
    p_run.add_argument("--seed", type=int, default=1337)
    p_run.add_argument("--chunk", type=int, default=None)
    p_run.add_argument("--buffer_len", type=int, default=None)
    p_run.add_argument("--rollback_tol", type=float, default=None)
    p_run.add_argument("--grad_norm_max", type=float, default=None)
    p_run.add_argument("--state_norm_max", type=float, default=None)

    sub.add_parser("list_sessions")
    p_show = sub.add_parser("show_session")
    p_show.add_argument("--session_id", type=str, required=True)

    args = parser.parse_args()
    store = ArtifactStore(args.artifacts_root)
    store.ensure()

    if args.cmd == "init_base":
        set_seed(args.seed)
        env_cfg = EnvConfig(nonlinear=(args.env_mode == "nonlinear"))
        cfg = ModelConfig(u_dim=args.u_dim, n_state=args.n_state)
        p_cfg = PlasticityConfig(
            lr=args.lr,
            momentum=args.momentum,
            ns_steps=args.ns_steps,
            adjust_lr_fn=(None if args.adjust_lr_fn == "none" else args.adjust_lr_fn),
        )
        pretrain_base(
            store=store,
            cfg=cfg,
            env_cfg=env_cfg,
            p_cfg=p_cfg,
            steps=args.pretrain_steps,
            batch_size=args.pretrain_batch,
            seq_len=args.pretrain_seq,
            seed=args.seed,
        )
        return

    if args.cmd == "new_session":
        set_seed(args.seed)
        env_cfg = EnvConfig(nonlinear=(args.env_mode == "nonlinear"))
        mu = float(args.mu)
        if mu < 0:
            mu = random.uniform(env_cfg.mu_min, env_cfg.mu_max)
        p_cfg = PlasticityConfig(
            lr=args.lr,
            momentum=args.momentum,
            ns_steps=args.ns_steps,
            adjust_lr_fn=(None if args.adjust_lr_fn == "none" else args.adjust_lr_fn),
            chunk=args.chunk,
            buffer_len=args.buffer_len,
            rollback_tol=args.rollback_tol,
            grad_norm_max=args.grad_norm_max,
            state_norm_max=args.state_norm_max,
        )
        create_new_session(
            store=store,
            session_id=args.session_id,
            mu=mu,
            env_mode=args.env_mode,
            p_cfg=p_cfg,
            parent_session_id=None,
            fork_copy_optimizer=True,
            fork_reset_optimizer=False,
        )
        return

    if args.cmd == "fork_session":
        parent = args.parent_session_id
        child = args.child_session_id
        # child inherits env_mode + mu + plasticity_cfg from parent meta
        device = device_auto()
        if not store.session_exists(parent):
            raise FileNotFoundError(f"Parent session does not exist: {parent}")
        parent_meta = store.load_session_meta(parent)
        mu = float(parent_meta.get("mu"))
        env_mode = str(parent_meta.get("env_mode"))
        p_cfg = PlasticityConfig(**parent_meta["plasticity_cfg"])  # type: ignore
        create_new_session(
            store=store,
            session_id=child,
            mu=mu,
            env_mode=env_mode,
            p_cfg=p_cfg,
            parent_session_id=parent,
            fork_copy_optimizer=bool(args.copy_optimizer),
            fork_reset_optimizer=bool(args.reset_optimizer),
        )
        return

    if args.cmd == "run_session":
        run_session(
            store=store,
            session_id=args.session_id,
            steps=args.steps,
            seed=args.seed,
            p_cfg_override=None,
            chunk=args.chunk,
            buffer_len=args.buffer_len,
            rollback_tol=args.rollback_tol,
            grad_norm_max=args.grad_norm_max,
            state_norm_max=args.state_norm_max,
        )
        return

    if args.cmd == "list_sessions":
        list_sessions(store)
        return

    if args.cmd == "show_session":
        show_session(store, args.session_id)
        return


if __name__ == "__main__":
    main()
