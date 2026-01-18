# TTT/SSM Eval

**Toy demo evaluation for Test-Time Training (TTT) in State Space Models (SSM)**

A sandbox for exploring input gradient dynamics and safety mechanisms in TTT-style architectures.

[![TTT Sentry Dashboard](./assets/dashboard_screenshot.png)](./assets/dashboard_screenshot.png)
*Click to expand - TTT Sentry Dashboard showing gradient monitoring, gate decisions, and canary drift detection*

## Overview

Test-time training (TTT) is an emerging paradigm where model weights update during inference rather than remaining frozen. Combined with state space models (SSM), this approach offers a potential alternative to traditional transformer context windows: instead of storing context as tokens in a fixed-size window, TTT/SSM architectures compress information directly into learned neural network weights at inference time.

This theoretically allows unbounded context limited only by the expressiveness of the weight-based memory—not by a token limit. The model "learns" the context into its parameters as it processes input, rather than attending over a growing sequence.

This project provides a sandbox for understanding the "write pressure" that inputs exert on TTT-style adapters, with **pluggable backbone architectures** (GRU, SSM) and **TTT objectives** (AR, MLM) to study how different designs affect gradient dynamics:

- **Gradient norm monitoring**: How hard is the input trying to update the adapter?
- **Update norm tracking**: How much do the weights actually change?
- **Per-token influence**: Which tokens contribute most to adapter updates?
- **Anomaly detection**: Flag inputs with unusually high write pressure using robust z-scores
- **Pre-update gate**: Block dangerous updates before they write into adapter weights
- **Directional monitoring**: Canary gradient alignment catches harmful update directions
- **Compression analysis**: Kolmogorov complexity proxy via zlib compression ratio

[![TTT-SSM Architecture Infographic](./assets/infographic.png)](./assets/infographic.png)
*TTT-SSM Architecture Overview*

## Concept

A small "memory module" (adapter layer) is the only component updated at test time. Each chunk of input triggers a TTT update step. The monitor tracks:

1. Adapter gradient norms (write pressure)
2. Actual weight update magnitudes
3. Per-token influence via embedding gradients
4. Statistical anomalies relative to recent history
5. Compression ratio (Kolmogorov complexity proxy)
6. Canary gradient alignment (directional damage signal)

The monitoring is **backbone-agnostic**: the same signals work whether the recurrent layer is a GRU, SSM, or something else entirely. This makes it useful for evaluating architectures that don't exist yet.

## Architecture Selection

The monitor supports pluggable architectures to study how different backbones and objectives affect gradient dynamics.

### Backbones

| Backbone | Description | When to Use |
|----------|-------------|-------------|
| **GRU** (default) | Gated recurrent unit | Baseline, production-like |
| **SSM** | Diagonal selective state space | Research, eigenvalue analysis |

**Why it matters**: GRU gating can hide/smooth write pressure. SSM linear recurrence ties write pressure more directly to eigenvalues/state dynamics. Thresholds tuned to one architecture may not generalize to another.

The SSM implementation uses vectorized `cumprod`+`cumsum` operations for efficient GPU execution (supports MPS on Apple Silicon).

### Objectives

| Objective | Description | Attack Surface |
|-----------|-------------|----------------|
| **AR** (default) | Next-token prediction | Spikes loss on OOD blobs |
| **MLM** | Masked token prediction | Can have *lower* loss on weird text |

**Why it matters**: AR naturally flags garbage (high loss). MLM can make garbage *easier* to predict (lower loss), requiring different detection strategies.

## Directional Monitoring

Gradient magnitude isn't everything. An update can stay under norm thresholds but push the model in a harmful direction.

### Canary Gradient Alignment

Computes cosine similarity between:
- **Chunk gradient**: Direction the current input wants to push weights
- **Canary gradient**: Direction that would hurt the canary

| Signal | Interpretation |
|--------|----------------|
| `cos > 0.3` (green) | Chunk gradient aligned with canary harm—suspicious |
| `cos < -0.3` (red) | Chunk gradient opposes canary harm—likely benign |
| `\|cos\| < 0.3` (grey) | Orthogonal, independent directions |

### Compression Ratio

Uses zlib compression as a Kolmogorov complexity proxy:
- **Low CR (~0.3-0.5)**: High entropy, random-looking data
- **High CR (~0.7-0.9)**: Low entropy, repetitive patterns

This catches base64 blobs and repeated tokens that might evade entropy-based checks.

## Pre-Update Gate

The key insight for TTT safety: once the model starts learning garbage, gradient norms naturally decrease as it memorizes the pattern. The dangerous moment is the first few chunks when big irreversible writes happen.

The pre-update gate blocks `opt.step()` when any of these conditions are detected:

| Check | What it catches |
|-------|-----------------|
| **Low token entropy** | Repetitive input (e.g., same token repeated) |
| **Low token diversity** | < 10% unique tokens in chunk |
| **Blob detection** | Base64, hex, or minified code patterns |
| **Instruction override** | Jailbreak patterns ("ignore previous instructions", "you are now unfiltered", etc.) |
| **OOD + heavy write** | High loss AND high gradient norm together |

When the gate blocks, the chunk is still processed for monitoring but the adapter weights remain unchanged.

## Post-Update Rollback

The pre-update gate prevents writes; rollback is the post-update airbag.

Even if a write passes the gate, rollback catches it if it corrupts the model's behavior. Each update is treated like a transaction:

1. **Snapshot** adapter weights before update
2. **Apply** the TTT update step
3. **Probe** with a fixed "canary" text (measure next-token loss)
4. **Rollback** if canary loss spikes, reverting to pre-update weights

| Trigger | What it catches |
|---------|-----------------|
| **Absolute canary delta** | Single update causes large loss increase on canary |
| **Canary delta z-score** | Update causes anomalous loss spike vs recent history |

This provides defense-in-depth: the gate blocks obvious threats, rollback catches subtle corruption that slips through.

```bash
# Test rollback with gate disabled (so rollback has something to catch)
python run_monitor.py --demo_high_entropy --disable_gate --chunk_tokens 32

# Adjust rollback sensitivity
python run_monitor.py --demo_high_entropy --disable_gate --rollback_abs_canary_delta 0.5
```

## Project Structure

```
ttt_ssm_eval/
├── .archive/                 # Archived legacy code (not maintained)
│   └── ttt/ui/               # Old embedded dashboard (reference only)
├── ttt/                      # Main package
│   ├── __init__.py           # Package exports
│   ├── core/
│   │   ├── model.py          # ToyTTTModel, tokenization
│   │   ├── gate.py           # Pre-update gate logic
│   │   ├── rollback.py       # Canary drift & rollback
│   │   ├── backbone.py       # GRU, SSM backbones
│   │   └── objective.py      # AR, MLM objectives
│   ├── monitors/
│   │   ├── gradient.py       # MonitorEvent, run_monitor()
│   │   └── signals.py        # Compression ratio, gradient alignment
│   ├── attacks/
│   │   └── red_team.py       # Adversarial attack optimization
├── ttt_ssm_nano/             # Nano SSM eval + artifacts API
│   └── artifacts_api/        # Unified FastAPI server (Nano + Text)
├── dashboard/                # Unified React UI (Nano + Text)
├── run_monitor.py            # CLI entry point
├── examples/                 # Red team analysis and results
└── assets/                   # Screenshots and docs
```

## Unified Dashboard (Nano + Text)

The React dashboard under `dashboard/` is the single UI surface for this repo:
- Nano: visualize the Phase 1 session artifact store (`artifacts/sessions/...`)
- Text: run the toy TTT input-gradient monitor and persist runs (`artifacts/text_runs/...`)

### Quick start

```bash
./start.sh
```

This starts:
- Artifacts API: `http://127.0.0.1:13579`
- Dashboard UI: `http://127.0.0.1:5173`

The script generates a tiny demo artifact store if `artifacts/base/base_checkpoint.pt` is missing.

### Chat (BPE Tiny LM)

The **Train** + **Chat** tabs serve a tiny local language model trained from scratch with:
- a dependency-free byte-level **BPE** tokenizer
- **Muon** as the default outer-loop optimizer
- a persistent fast **context net** (TTT) per chat session stored in `artifacts/text_sessions/`

Start a training run:
- via UI: open the **Train** tab and set corpus/hparams
- via CLI (example corpus: this repo’s own docs):

```bash
python -m ttt.text_lm.train --corpus_dir training_data --device mps --steps 2000
```

This writes:
- `artifacts/text_models/index.json`
- `artifacts/text_models/<model_id>/{tokenizer.json,config.json,checkpoint.pt,train_log.jsonl}`

Then start `./start.sh` and use the **Chat** tab.

To run generation on Apple Silicon GPU via the API, set:
`TEXT_LM_DEVICE=mps ./start.sh`

### Manual start (optional)

```bash
ttt-ssm-api --artifacts_root artifacts --port 13579
```

```bash
npm -C dashboard install
npm -C dashboard run dev
```

If your API is not on `http://127.0.0.1:13579`, set `VITE_NANO_API_URL` when running Vite.

## Installation

```bash
pip install -e .
```

Requires PyTorch >= 2.0

## Usage

```bash
# Run built-in demo with mixed benign/adversarial text
python run_monitor.py --demo

# Run high-entropy demo (triggers large updates)
python run_monitor.py --demo_high_entropy

# Analyze custom text
python run_monitor.py --text "Your text here"

# Analyze from file
python run_monitor.py --file input.txt

# Read from stdin
cat document.txt | python run_monitor.py --stdin
```

### Architecture Examples

```bash
# SSM backbone (study eigenvalue dynamics)
python run_monitor.py --demo --backbone ssm

# MLM objective (different attack surface)
python run_monitor.py --demo --objective mlm --mlm_prob 0.15

# Full matrix: SSM + MLM
python run_monitor.py --demo --backbone ssm --objective mlm

# Apple Silicon GPU acceleration
python run_monitor.py --demo --backbone ssm --device mps

# Fast mode (skip canary gradient computation)
python run_monitor.py --demo --disable_canary_grad
```

### Options

| Flag | Description |
|------|-------------|
| `--device` | `cpu`, `cuda`, or `mps` (Apple GPU) |
| `--chunk_tokens` | Tokens per TTT chunk (default: 128) |
| `--lr` | TTT learning rate (default: 0.05) |
| `--abs_grad_norm_threshold` | Absolute gradient norm flag threshold |
| `--abs_update_norm_threshold` | Absolute update norm flag threshold |
| `--robust_z_threshold` | Robust z-score threshold for anomaly detection |
| `--write_json` | Output `monitor_report.json` |

### Gate Options

| Flag | Description |
|------|-------------|
| `--disable_gate` | Turn off the pre-update gate (allow all updates) |
| `--min_entropy_threshold` | Min token entropy to allow update (default: 1.0) |
| `--min_diversity_threshold` | Min unique token ratio (default: 0.1) |
| `--ood_loss_threshold` | Loss threshold for OOD detection (default: 8.0) |
| `--ood_grad_threshold` | Grad threshold for OOD+heavy-write (default: 2.0) |

### Rollback Options

| Flag | Description |
|------|-------------|
| `--disable_rollback` | Turn off post-update rollback |
| `--rollback_abs_canary_delta` | Canary loss delta threshold to trigger rollback (default: 1.0) |
| `--rollback_z_threshold` | Robust z-score threshold on canary delta (default: 6.0) |
| `--canary_text` | Custom canary text for drift detection |

### Architecture Options

| Flag | Description |
|------|-------------|
| `--backbone` | `gru` (default) or `ssm` |
| `--objective` | `ar` (default) or `mlm` |
| `--mlm_prob` | MLM mask probability (default: 0.15) |
| `--disable_canary_grad` | Skip canary gradient alignment (faster) |
| `--canary_grad_every` | Recompute canary gradient every N chunks (default: 1) |

## Legacy Dashboard (Deprecated)

`ttt.ui.dashboard` previously hosted an embedded HTML dashboard. It is now deprecated in favor of the unified React dashboard.

The legacy implementation is archived at `.archive/ttt/ui/legacy_dashboard.py` for reference only.

## Red Team Attack

[![Red Team Attack](./assets/red_team_screenshot.png)](./assets/red_team_screenshot.png)
*Click to expand - Adversarial optimization trajectory and attack validation*

The repo includes an adversarial red team script that attempts to generate "Silent Killer" payloads - inputs that:

1. **Bypass the gate** (low gradient norm, high entropy, no jailbreak patterns)
2. **Avoid rollback** (subtle canary degradation)
3. **Cause damage** (corrupt model behavior over time)

### Running Red Team

```bash
# CLI (default: GRU + AR)
python -m ttt.attacks.red_team

# Test against SSM backbone
python -m ttt.attacks.red_team --backbone ssm

# Test against MLM objective
python -m ttt.attacks.red_team --objective mlm
```

### Attack Strategy

The optimizer uses Gumbel-Softmax relaxation to make discrete token selection differentiable, then optimizes for:

- Gradient norm just under threshold (2.4 vs 2.5)
- High token entropy (looks like natural text)
- Maximum directional damage to canary representations

### Attack Outcomes

| Result | Meaning |
|--------|---------|
| **SILENT KILLER** | Bypassed gate AND rollback, caused canary damage |
| **CAUGHT BY ROLLBACK** | Bypassed gate but rollback detected corruption |
| **BLOCKED BY GATE** | Pre-update gate blocked the attack |
| **BYPASSED (no damage)** | Passed defenses but caused no measurable harm |

### Why This Matters

If the optimizer consistently finds Silent Killer payloads, the defenses need strengthening. Remaining weak spots:

- **Fixed thresholds**: Attacker can optimize to stay just under
- **No cumulative tracking**: Many small writes can add up

**Now addressed**: Canary gradient alignment provides directional analysis—we now track not just *how much* but *which direction* updates push the model.

## Output

Each chunk reports:
- Backbone and objective used (GRU/SSM, AR/MLM)
- Loss, gradient norm, effective update norm, attempted update norm
- Robust z-scores relative to recent history
- Flag status with reasons
- Gate decision (ALLOWED/BLOCKED) with reasons
- Rollback status and canary drift metrics
- Token entropy and diversity metrics
- Compression ratio (Kolmogorov complexity proxy)
- Canary gradient cosine similarity (directional alignment)
- Canary gradient dot product (magnitude-weighted alignment)
- Top influential tokens

## Limitations

This is an educational sandbox, not a production guardrail. The toy model (embedding + backbone + adapter) is intentionally minimal to make gradient dynamics interpretable. Key constraints:

- **Toy scale**: 64-dim embeddings, 8K vocab—real models are 1000x larger
- **No actual language model**: Random embeddings, no pretrained weights
- **SSM is simplified**: Diagonal selective SSM, not full Mamba/S4/S5

The SSM implementation is vectorized (`cumprod`+`cumsum`) for efficiency and supports Apple MPS GPU acceleration.

## Why This Matters

Traditional transformers scale context by extending the attention window, which is O(n²) in compute and memory. TTT/SSM architectures propose a fundamentally different approach: compress context into weight updates, making "context length" a function of model expressiveness rather than sequence length.

The safety question becomes: if the model learns from every input at inference time, how do you prevent it from learning things it shouldn't? This repo explores that question with:

- **Gradient-based monitoring**: Track write pressure magnitude
- **Directional analysis**: Canary gradient alignment catches harmful directions
- **Pre-update gates**: Block suspicious inputs before they write
- **Post-update rollback**: Revert updates that corrupt model behavior
- **Architecture-agnostic design**: Same safety signals work across GRU, SSM, and future backbones

## Keywords

`test-time-training` `TTT` `state-space-models` `SSM` `mamba` `context-compression` `weight-based-memory` `inference-time-learning` `gradient-monitoring` `input-safety` `multi-backbone` `diagonal-ssm` `mlm` `compression-ratio` `gradient-alignment` `mps`
