# TTT/SSM Eval

**Toy demo evaluation for Test-Time Training (TTT) in State Space Models (SSM)**

A sandbox for exploring input gradient dynamics and safety mechanisms in TTT-style architectures.

[![TTT Sentry Dashboard](./dashboard_screenshot.png)](./dashboard_screenshot.png)
*Click to expand - TTT Sentry Dashboard showing gradient monitoring, gate decisions, and canary drift detection*

## Overview

Test-time training (TTT) is an emerging paradigm where model weights update during inference rather than remaining frozen. Combined with state space models (SSM), this approach offers a potential alternative to traditional transformer context windows: instead of storing context as tokens in a fixed-size window, TTT/SSM architectures compress information directly into learned neural network weights at inference time.

This theoretically allows unbounded context limited only by the expressiveness of the weight-based memory—not by a token limit. The model "learns" the context into its parameters as it processes input, rather than attending over a growing sequence.

This project provides a sandbox for understanding the "write pressure" that inputs exert on TTT-style adapters:

- **Gradient norm monitoring**: How hard is the input trying to update the adapter?
- **Update norm tracking**: How much do the weights actually change?
- **Per-token influence**: Which tokens contribute most to adapter updates?
- **Anomaly detection**: Flag inputs with unusually high write pressure using robust z-scores
- **Pre-update gate**: Block dangerous updates before they write into adapter weights

## Concept

A small "memory module" (adapter layer) is the only component updated at test time. Each chunk of input triggers a TTT update step using next-token prediction loss. The monitor tracks:

1. Adapter gradient norms (write pressure)
2. Actual weight update magnitudes
3. Per-token influence via embedding gradients
4. Statistical anomalies relative to recent history

High-entropy or adversarial inputs often produce larger gradient norms, making this useful for studying input characteristics that drive aggressive weight updates.

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
python ttt_input_gradient_monitor.py --demo_high_entropy --disable_gate --chunk_tokens 32

# Adjust rollback sensitivity
python ttt_input_gradient_monitor.py --demo_high_entropy --disable_gate --rollback_abs_canary_delta 0.5
```

## Installation

```bash
pip install -e .
```

Requires PyTorch >= 2.0

## Usage

```bash
# Run built-in demo with mixed benign/adversarial text
python ttt_input_gradient_monitor.py --demo

# Run high-entropy demo (triggers large updates)
python ttt_input_gradient_monitor.py --demo_high_entropy

# Analyze custom text
python ttt_input_gradient_monitor.py --text "Your text here"

# Analyze from file
python ttt_input_gradient_monitor.py --file input.txt

# Read from stdin
cat document.txt | python ttt_input_gradient_monitor.py --stdin
```

### Options

| Flag | Description |
|------|-------------|
| `--device` | `cpu` or `cuda` |
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

## Dashboard UI

The TTT Sentry Dashboard provides a web-based interface for interactive monitoring and visualization.

```bash
# Start the dashboard
python ttt_dashboard.py

# Or with uvicorn for auto-reload during development
uvicorn ttt_dashboard:app --reload --port 6677
```

Then open http://127.0.0.1:6677 in your browser.

### Features

| Component | Description |
|-----------|-------------|
| **Input Stream** | Paste or type text to analyze, with demo presets |
| **Parameter Controls** | Adjust chunk size, entropy threshold, OOD loss threshold, rollback delta |
| **Telemetry Chart** | Real-time visualization of gradient norm, loss, and canary delta |
| **Event Log** | Per-chunk breakdown with gate decisions, metrics, and top tokens |
| **Session Stats** | Summary of chunks processed, blocked, rolled back, and max gradient |
| **Export JSON** | Download full event log for offline analysis |

### Visual Indicators

- **Green (OK)**: Update accepted, chunk learned into adapter
- **Red (BLOCKED)**: Pre-update gate triggered, weights unchanged
- **Orange (ROLLBACK)**: Post-update canary drift detected, weights reverted

Vertical dashed lines on the chart mark blocked and rollback events for easy correlation.

## Red Team Attack

[![Red Team Attack](./red_team_screenshot.png)](./red_team_screenshot.png)
*Click to expand - Adversarial optimization trajectory and attack validation*

The repo includes an adversarial red team script that attempts to generate "Silent Killer" payloads - inputs that:

1. **Bypass the gate** (low gradient norm, high entropy, no jailbreak patterns)
2. **Avoid rollback** (subtle canary degradation)
3. **Cause damage** (corrupt model behavior over time)

### Running Red Team

```bash
# CLI
python red_team_attack.py

# Or from the dashboard UI
# Click the "⚔️ Red Team" button
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

If the optimizer consistently finds Silent Killer payloads, the defenses need strengthening. Current weak spots:

- **Fixed thresholds**: Attacker can optimize to stay just under
- **No cumulative tracking**: Many small writes can add up
- **No directional analysis**: Gradient magnitude isn't everything

## Output

Each chunk reports:
- Loss, gradient norm, effective update norm, attempted update norm
- Robust z-scores relative to recent history
- Flag status with reasons
- Gate decision (ALLOWED/BLOCKED) with reasons
- Rollback status and canary drift metrics
- Token entropy and diversity metrics
- Top influential tokens

## Limitations

This is an educational sandbox, not a production guardrail. The toy model (embedding + GRU + adapter) is intentionally minimal to make gradient dynamics interpretable.

## Why This Matters

Traditional transformers scale context by extending the attention window, which is O(n²) in compute and memory. TTT/SSM architectures propose a fundamentally different approach: compress context into weight updates, making "context length" a function of model expressiveness rather than sequence length.

The safety question becomes: if the model learns from every input at inference time, how do you prevent it from learning things it shouldn't? This repo explores that question with gradient-based monitoring and pre-update gates.

## Keywords

`test-time-training` `TTT` `state-space-models` `SSM` `mamba` `context-compression` `weight-based-memory` `inference-time-learning` `gradient-monitoring` `input-safety`
