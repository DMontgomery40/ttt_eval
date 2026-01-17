# TTT-SSM Evaluation - Claude Code Instructions

## Essential Commands

### Python Package Installation
```bash
pip install -e .
```

### Running the Monitor (CLI)
```bash
# Built-in demos
python run_monitor.py --demo                    # Mixed benign/adversarial
python run_monitor.py --demo_high_entropy       # High-entropy triggers

# Custom text analysis
python run_monitor.py --text "Your text here"
python run_monitor.py --file input.txt
cat document.txt | python run_monitor.py --stdin

# Architecture selection
python run_monitor.py --demo --backbone ssm     # SSM instead of GRU
python run_monitor.py --demo --objective mlm    # MLM instead of AR

# Test safety mechanisms
python run_monitor.py --demo_high_entropy --disable_gate --chunk_tokens 32
python run_monitor.py --demo --disable_rollback
```

### Dashboard (Web UI)
```bash
# Start FastAPI dashboard
python -m ttt.ui.dashboard

# Or with auto-reload for development
uvicorn ttt.ui.dashboard:app --reload --port 6677
# Opens at http://127.0.0.1:6677
```

### Dashboard Development (React/TypeScript)
```bash
cd dashboard
npm install           # Install dependencies
npm run dev          # Development server with HMR
npm run build        # Production build
npm run preview      # Preview production build
```

### Red Team Attacks
```bash
# Adversarial optimization
python -m ttt.attacks.red_team                  # Default: GRU + AR
python -m ttt.attacks.red_team --backbone ssm   # Test against SSM
python -m ttt.attacks.red_team --objective mlm  # Test against MLM

# Or click "⚔️ Red Team" button in dashboard UI
```

### Phase 0/1 Session Management (Advanced)
```bash
# Phase 0: Muon optimizer with hidden-μ physics
python ttt_ssm_nano/phase0_muon.py

# Phase 1: Session branching with transaction semantics
python ttt_ssm_nano/phase1_branching_muon.py --artifacts_root artifacts init_base --pretrain_steps 500
python ttt_ssm_nano/phase1_branching_muon.py --artifacts_root artifacts new_session --session_id s1
python ttt_ssm_nano/phase1_branching_muon.py --artifacts_root artifacts run_session --session_id s1 --steps 600
python ttt_ssm_nano/phase1_branching_muon.py --artifacts_root artifacts fork_session --parent_session_id s1 --child_session_id s1b
python ttt_ssm_nano/phase1_branching_muon.py --artifacts_root artifacts list_sessions
```

## Architecture Overview

### Core Concept: Test-Time Training Safety
TTT models update weights during inference (not just training). This creates a fundamentally different safety challenge: **how do you prevent the model from learning things it shouldn't at inference time?**

### Defense-in-Depth Strategy
Two layers of protection:
1. **Pre-update gate** (ttt/core/gate.py): Blocks suspicious inputs BEFORE they write to adapter weights
2. **Post-update rollback** (ttt/core/rollback.py): Reverts updates that corrupt model behavior AFTER they happen

This provides transaction-like semantics: attempt → validate → commit or undo.

### Pluggable Backbone Architecture
The system is **backbone-agnostic** by design. Same safety signals work across:
- **GRU** (default): Gated recurrent unit, production-like baseline
- **SSM**: Diagonal selective state space model with vectorized cumprod/cumsum ops

Key insight: Gradient monitoring works regardless of recurrence mechanism. Add new backbones by implementing the `RecurrentBackbone` interface in `ttt/core/backbone.py`.

### Model Architecture (Frozen vs Plastic)
```
Input tokens
    ↓
[Embedding Layer]  ← Frozen (random init, never updated)
    ↓
[Backbone: GRU/SSM]  ← Frozen (random init, never updated)
    ↓
[TTT Adapter]  ← Plastic (updated at test time via gradient descent)
    ↓
Output logits
```

Only the adapter learns from inputs. Backbone provides recurrent context but stays frozen.

### TTT Objectives (Pluggable)
- **AR (Autoregressive)**: Next-token prediction. High loss on garbage → natural anomaly signal.
- **MLM (Masked Language Model)**: Masked token prediction. Can have *lower* loss on weird text → requires different detection.

Extend by implementing `TTTObjective` interface in `ttt/core/objective.py`.

### Monitoring Signal Pipeline
Each input chunk flows through:
1. **Gradient computation**: How hard does input try to update adapter?
2. **Statistical analysis**: Robust z-scores relative to recent history
3. **Pre-update gate**: Check entropy, diversity, blob patterns, jailbreak attempts, OOD+heavy-write
4. **Update application**: If gate allows, apply gradient descent step
5. **Canary probe**: Measure loss on fixed "canary" text after update
6. **Rollback decision**: Revert if canary loss spikes (corruption detected)
7. **Event logging**: Record all metrics, decisions, top tokens

All signals are backbone-agnostic and TTT-objective-agnostic.

### Directional Monitoring (Beyond Magnitude)
Gradient **magnitude** alone is insufficient. An update can stay under norm thresholds but push the model in a harmful direction.

**Canary Gradient Alignment** (ttt/monitors/signals.py):
- Computes cosine similarity between chunk gradient and canary gradient
- `cos > 0.3`: Chunk aligned with canary harm (suspicious)
- `cos < -0.3`: Chunk opposes canary harm (likely benign)
- `|cos| < 0.3`: Orthogonal, independent directions

**Compression Ratio** (Kolmogorov complexity proxy):
- Uses zlib compression to detect high/low entropy patterns
- Low CR (~0.3-0.5): Random-looking data, base64 blobs
- High CR (~0.7-0.9): Repetitive patterns, repeated tokens

### Phase 0/1: Advanced Session Branching
Extensions in `ttt_ssm_nano/` that add hidden-μ SSM physics and session persistence:

**Phase 0** (phase0_muon.py):
- Muon optimizer with momentum in hidden state space
- SSM with learnable diagonal A, B, C matrices
- Online gradient descent during inference

**Phase 1** (phase1_branching_muon.py):
- Session branching: fork from any session to create divergent learning paths
- Artifact persistence: weights, optimizer state, metrics, update events
- Transaction semantics: snapshot → update → validate → commit or rollback
- Rollback trigger: `post_loss > pre_loss * (1 + rollback_tol)` on rolling buffer

Artifact structure:
```
artifacts/
  base/
    base_checkpoint.pt
    base_meta.json
  sessions/
    index.json              # Registry with lineage graph
    <session_id>/
      meta.json             # Schema version, model signature, parent lineage
      plastic_state.pt      # Adapter weights
      optim_state.pt        # Muon optimizer state
      metrics.json
      update_events.jsonl
      runs/
        <run_id>/
          per_step.csv
          update_events.json
          mse_curve.png
```

Critical fields:
- `model_signature`: Hash of config + base checkpoint. Prevents session reuse with wrong model.
- `parent_session_id` / `root_session_id`: Branching lineage tracking
- `base_ckpt_hash`: Ties sessions to exact base checkpoint

## Key Design Patterns

### Adding New Backbones
1. Implement `RecurrentBackbone` interface in `ttt/core/backbone.py`
2. Add to `BACKBONE_REGISTRY` dictionary
3. Ensure forward pass returns `(output, hidden_state)` tuple
4. Monitoring signals automatically work (gradient norm, update norm, etc.)

Example: SSM backbone uses vectorized operations for GPU efficiency (supports MPS on Apple Silicon).

### Adding New TTT Objectives
1. Implement `TTTObjective` interface in `ttt/core/objective.py`
2. Add to `OBJECTIVE_REGISTRY` dictionary
3. Implement `compute_loss(model, tokens)` method
4. Safety gates and rollback automatically apply

Example: MLM masks random tokens and predicts them. Lower loss on garbage → inverts AR anomaly signal.

### Extending Safety Gates
Gate checks live in `ttt/core/gate.py`. Current checks:
- Low token entropy (repeated tokens)
- Low token diversity (< 10% unique)
- Blob detection (base64, hex, minified code)
- Instruction override (jailbreak patterns)
- OOD + heavy write (high loss AND high gradient norm)

Add new checks by extending `should_gate_update()` function. Return tuple: `(should_block: bool, reasons: list[str])`.

### Adding Monitoring Signals
Signals live in `ttt/monitors/signals.py`. Current signals:
- Gradient norm (magnitude of adapter gradient)
- Update norm (actual weight change magnitude)
- Robust z-scores (MAD-based anomaly detection)
- Compression ratio (zlib-based Kolmogorov proxy)
- Canary gradient alignment (cosine similarity, dot product)

Add new signals by extending `MonitorEvent` dataclass in `ttt/monitors/gradient.py`.

## Critical Files (Non-Obvious Interactions)

### Core Abstractions
- `ttt/core/model.py`: ToyTTTModel orchestrates backbone + adapter. Only adapter is plastic.
- `ttt/core/backbone.py`: RecurrentBackbone interface. GRU vs SSM are drop-in replacements.
- `ttt/core/objective.py`: TTTObjective interface. AR vs MLM change attack surface.
- `ttt/core/gate.py`: Pre-update gate with multiple heuristic checks.
- `ttt/core/rollback.py`: Canary-based post-update validation and reversion.

### Monitoring Pipeline
- `ttt/monitors/gradient.py`: MonitorEvent dataclass and run_monitor() orchestration.
- `ttt/monitors/signals.py`: Directional signals (compression, gradient alignment).

### Red Team
- `ttt/attacks/red_team.py`: Gumbel-Softmax adversarial optimization to find "Silent Killer" payloads.

### Dashboard
- `ttt/ui/dashboard.py`: FastAPI backend serving monitoring API.
- `dashboard/src/`: React/TypeScript frontend with Recharts visualization, Zustand state, Tailwind styling.

### Phase 0/1 Extensions
- `ttt_ssm_nano/phase0_muon.py`: Muon optimizer with hidden-μ SSM.
- `ttt_ssm_nano/phase1_branching_muon.py`: Session branching, artifact persistence, transaction semantics.

## Important Context

### TTT Safety Philosophy
Traditional transformers have frozen weights at inference. TTT models learn from every input during inference. This fundamentally changes the threat model:
- **Input = code**: Every chunk is a gradient descent step that writes into model weights
- **Adversarial inputs**: Not just about triggering bad outputs, but corrupting future behavior
- **Cumulative damage**: Many small writes can add up over time

Defense requires **both** magnitude monitoring (gradient/update norms) **and** directional monitoring (canary gradient alignment).

### Robust Z-Score Anomaly Detection
Uses Median Absolute Deviation (MAD) instead of standard deviation. Why?
- Traditional z-scores use mean/std, which are sensitive to outliers
- MAD uses median, which is robust to outliers
- Formula: `z = (x - median) / (1.4826 * MAD)`
- Threshold typically 3-6 for anomaly flagging

This prevents adversarial inputs from poisoning the baseline distribution.

### Why Pluggable Architectures Matter
Gradient dynamics differ wildly across backbones:
- **GRU gating**: Can hide/smooth write pressure
- **SSM eigenvalues**: Tie write pressure directly to state dynamics
- **Future architectures**: Unknown gradient characteristics

By keeping monitoring backbone-agnostic, we can evaluate architectures that don't exist yet. Thresholds tuned to GRU may not generalize to SSM or Mamba.

### Limitations (Educational Sandbox)
This is a research sandbox, not production-ready:
- Toy scale: 64-dim embeddings, 8K vocab (real models are 1000x larger)
- No pretrained weights: Random initialization only
- Simplified SSM: Diagonal selective SSM, not full Mamba/S4/S5

The goal is interpretable gradient dynamics, not language modeling performance.

## Common Workflows

### Evaluate New Backbone
1. Implement `RecurrentBackbone` in `ttt/core/backbone.py`
2. Add to `BACKBONE_REGISTRY`
3. Test: `python run_monitor.py --demo --backbone your_backbone`
4. Compare gradient norms, update norms, and gate/rollback rates vs GRU baseline

### Tune Safety Thresholds
1. Start dashboard: `python -m ttt.ui.dashboard`
2. Load demo text and observe telemetry
3. Adjust entropy threshold, OOD loss threshold, rollback delta via UI controls
4. Export JSON for offline analysis
5. Update defaults in `run_monitor.py` CLI flags

### Red Team New Defense
1. Add new check to `ttt/core/gate.py` or `ttt/core/rollback.py`
2. Run: `python -m ttt.attacks.red_team` to find bypass attempts
3. If "SILENT KILLER" attacks succeed, strengthen defenses
4. Iterate until optimizer can't find bypasses

### Create Phase 1 Session Branch
1. Initialize base: `python phase1_branching_muon.py --artifacts_root artifacts init_base --pretrain_steps 500`
2. Create session: `python phase1_branching_muon.py --artifacts_root artifacts new_session --session_id s1`
3. Run learning: `python phase1_branching_muon.py --artifacts_root artifacts run_session --session_id s1 --steps 600`
4. Fork branch: `python phase1_branching_muon.py --artifacts_root artifacts fork_session --parent_session_id s1 --child_session_id s1b`
5. Run fork: `python phase1_branching_muon.py --artifacts_root artifacts run_session --session_id s1b --steps 600 --seed 2`
6. View artifacts: `artifacts/sessions/*/runs/*/mse_curve.png`
