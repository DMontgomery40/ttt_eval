# TTT Input Gradient Monitor

A toy implementation for exploring test-time training (TTT) dynamics and input monitoring.

## Overview

Test-time training is an emerging paradigm where model weights are updated during inference rather than remaining frozen. Instead of storing context in a fixed-size token window, TTT approaches compress information into the model's weights themselves—theoretically allowing unbounded context limited only by the expressiveness of the learned representations.

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

## Output

Each chunk reports:
- Loss, gradient norm, update norm
- Robust z-scores relative to recent history
- Flag status with reasons
- Gate decision (ALLOWED/BLOCKED) with reasons
- Token entropy and diversity metrics
- Top influential tokens

## Limitations

This is an educational sandbox, not a production guardrail. The toy model (embedding + GRU + adapter) is intentionally minimal to make gradient dynamics interpretable.
