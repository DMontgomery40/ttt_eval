from __future__ import annotations

from pydantic import BaseModel, Field

from .. import phase1_branching_muon as muon


class ForkSessionRequest(BaseModel):
    child_session_id: str = Field(..., min_length=1)
    copy_optimizer: bool = True
    reset_optimizer: bool = False


def fork_session(
    *,
    artifacts_root: str,
    parent_session_id: str,
    req: ForkSessionRequest,
) -> None:
    """
    Create a real on-disk fork using the Phase 1 Muon implementation.

    This is intentionally thin: the canonical semantics live in
    `ttt_ssm_nano/phase1_branching_muon.py`.
    """
    store = muon.ArtifactStore(artifacts_root)
    store.ensure()

    if not store.session_exists(parent_session_id):
        raise FileNotFoundError(f"Parent session does not exist: {parent_session_id}")

    parent_meta = store.load_session_meta(parent_session_id)
    mu = float(parent_meta.get("mu"))
    env_mode = str(parent_meta.get("env_mode"))
    p_cfg = muon.PlasticityConfig(**parent_meta["plasticity_cfg"])  # type: ignore[arg-type]

    muon.create_new_session(
        store=store,
        session_id=req.child_session_id,
        mu=mu,
        env_mode=env_mode,
        p_cfg=p_cfg,
        parent_session_id=parent_session_id,
        fork_copy_optimizer=bool(req.copy_optimizer),
        fork_reset_optimizer=bool(req.reset_optimizer),
    )
