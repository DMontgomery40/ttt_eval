Phase 1 artifact layout (branch-ready from day 1)

Everything lives under an artifacts_root directory:

artifacts/
  base/
    base_checkpoint.pt
    base_meta.json
  sessions/
    index.json
    <session_id>/
      meta.json
      plastic_state.pt
      optim_state.pt
      metrics.json
      update_events.jsonl
      runs/
        <run_id>/
          per_step.csv
          update_events.json
          mse_curve.png

Why these fields exist (the “don’t brick yourself later” bits)

Every meta.json includes:

schema_version: format version

model_signature: hash(config + base checkpoint hash)
If this doesn’t match, the loader refuses to run. This prevents “session files silently used with a different model.”

base_ckpt_hash: ties sessions to the exact base checkpoint

parent_session_id and root_session_id: branching lineage

Rollback trigger (concrete, exactly what you asked for)

During online updates, we maintain a rolling buffer of the last buffer_len transitions.

At each update step:

Compute pre_loss (MSE of predicted z_next on the buffer)

Snapshot plastic weights + optimizer state

Take one Muon step

Compute post_loss on the same buffer

Rollback if:

post_loss > pre_loss * (1 + rollback_tol)

Default rollback_tol = 0.20 (20% regression).
Also rollback if hidden state norms explode beyond state_norm_max.

That’s your “transaction semantics” in toy form: attempt, validate, commit or undo.

Phase 1 runnable implementation

I wrote a single script that implements all of this:

init_base (optional pretrain)

new_session

fork_session

run_session

list_sessions

show_session

Download it here:
Download phase1_branching_muon.py

One-command demo (init → session → run → fork → run branch)

After you’ve saved phase1_branching_muon.py into your folder:

echo "install deps" && python3 -m pip -q install matplotlib && echo "init base" && python3 phase1_branching_muon.py --artifacts_root artifacts init_base --pretrain_steps 500 --pretrain_batch 32 --pretrain_seq 32 --env_mode linear --u_dim 16 --n_state 32 --lr 0.005 --momentum 0.95 --ns_steps 5 && echo "new session s1" && python3 phase1_branching_muon.py --artifacts_root artifacts new_session --session_id s1 --mu 0.12 --env_mode linear --lr 0.005 --momentum 0.95 --ns_steps 5 --chunk 32 --buffer_len 32 --rollback_tol 0.20 --grad_norm_max 20 --state_norm_max 1000000 && echo "run s1" && python3 phase1_branching_muon.py --artifacts_root artifacts run_session --session_id s1 --steps 600 --seed 1 && echo "fork s1 -> s1b (copy optimizer)" && python3 phase1_branching_muon.py --artifacts_root artifacts fork_session --parent_session_id s1 --child_session_id s1b --copy_optimizer 1 --reset_optimizer 0 && echo "run s1b" && python3 phase1_branching_muon.py --artifacts_root artifacts run_session --session_id s1b --steps 600 --seed 2 && echo "list sessions" && python3 phase1_branching_muon.py --artifacts_root artifacts list_sessions


Where to look afterward:

artifacts/sessions/s1/runs/.../mse_curve.png

artifacts/sessions/s1b/runs/.../mse_curve.png

artifacts/sessions/index.json for lineage and registry

Tiny knob worth knowing (fork momentum)

If you want a fork that starts “same weights but fresh optimizer,” do:

--reset_optimizer 1

That gives you: same brain state, different “learning inertia.”

If you run that demo and paste just the final [metrics] blocks for s1 and s1b, I’ll tell you what to tweak first (lr vs chunk vs rollback_tol) based on the shape of the three curves: base, session-no-update, session-with-updates.
