# ttt_ssm_eval — TTT Safety Research Infrastructure

Test-Time Training turns every input into a training example. In a transformer, garbage in = garbage out. In TTT, garbage in = garbage *learned*. This repo is infrastructure for studying how to make that safe.

[![Dashboard](./assets/dashboard_screenshot.png)](./assets/dashboard_screenshot.png)
[![TTT-SSM Architecture Infographic](./assets/infographic.png)](./assets/infographic.png)

## Architecture Philosophy

### The Problem

Standard TTT implementations treat weight updates as pure math—token goes in, gradient happens, weights change. The "learning" is implicit inside `forward()`. This is fine for research benchmarks on benign data. It's catastrophic for anything adversarial.

### The Stance

Safety checks are deliberately kept *outside* the computation graph:

- **Inside the graph:** Safety becomes a learned approximation. The model can meta-learn to bypass it. Adversarial gradients can "convince" safety checks to approve poison.
- **Outside the graph:** Safety is explicit Python. Gradients can't flow through regex. The gate can't be gradient-hacked because it's not differentiable.

The tradeoff: the model can't learn to be safe—only to do tasks while safety logic watches. We accept this because "auditable and limited" beats "learned and hackable."

### Defense in Depth

No single layer is sufficient. The architecture stacks defenses with different failure modes:

| Layer | Speed | What it catches | Failure mode |
|-------|-------|-----------------|--------------|
| **Gate** | Fast, static | Known attack patterns, entropy anomalies, instruction overrides | Novel obfuscation bypasses it |
| **Canary** | Fast, dynamic | Catastrophic corruption (loss spikes, weight explosions) | Subtle drift doesn't spike loss |
| **Auditor** | Slow, smart | Patterns across sessions, semantic attacks | Blind spots in the auditor |
| **Human** | Slowest | Everything else | Inattention, scale |

The goal isn't perfect defense. It's making failures observable before they compound.

### Complementary Learning Systems (Wake/Sleep)

Inspired by how biological memory actually works:

- **Hippocampus (fast, plastic):** Learns quickly during the day, temporary storage
- **Neocortex (slow, stiff):** Learns slowly during sleep, permanent storage
- **Sleep:** Replay experiences, selectively consolidate what matters

Mapped to this repo:

| Biological | This Repo | When |
|------------|-----------|------|
| Hippocampus | Fast context weights (per-session) | During chat |
| Neocortex | Core model weights | Never during inference |
| Sleep | `ttt.text_lm.sleep` | Offline consolidation |

During inference, only fast weights update. Core weights are frozen. Users cannot send gradients to the core model.

During sleep (offline, no user interaction):
1. Harvest chat traces from the day
2. Filter through auditor (reject poison)
3. Interleave with "core knowledge" (catastrophic forgetting prevention)
4. Update core model slowly, carefully
5. Reset fast weights for tomorrow

This creates asymmetric timescales: fast learning happens continuously but temporarily. Permanent learning happens rarely, supervised, offline.

---

## What Exists vs What Is Claimed

### Exists (Concrete)

- Artifact store with branching sessions ("git for plastic weights")
- TTT safety monitor with gate + rollback + directional signals
- Tiny trainable LM (BPE + Muon) with per-session fast weights
- Sleep consolidation that replays chat traces into core model
- Dashboard for observing all of the above

### Thesis (Being Tested)

- Fast weights behave like a learned context window
- Externalizing safety from the graph is worth the tradeoff
- Sleep consolidation can be made selective and delta-based
- The whole stack produces observable, correctable failures

---

## Table of Contents

- [Quick Start](#quick-start)
- [Product Surface](#product-surface)
- [Nano Domain](#nano-domain-ssm--branching-sessions)
- [Text Domain](#text-domain)
  - [World A: TTT Sentry](#text-world-a--ttt-sentry)
  - [World B: TinyLM](#text-world-b--tinylm-offline-training)
  - [Chat Sessions](#text-world-b--chat-sessions)
  - [Sleep Consolidation](#sleep-consolidation)
- [Safety Coverage Matrix](#safety-coverage-matrix)
- [Training Data](#training-data--scaling)
- [Troubleshooting](#troubleshooting)
- [Repo Map](#repo-map)

---

## Quick Start

### Install

```bash
python -m pip install -e .
```

### Start Everything

```bash
./start.sh
```

- API: `http://127.0.0.1:13579`
- Dashboard: `http://127.0.0.1:5173`

Environment knobs:
- `ARTIFACTS_ROOT=/path/to/artifacts`
- `TEXT_LM_DEVICE=auto|cpu|mps`

### Manual Start

API only:
```bash
python -m ttt_ssm_nano.artifacts_api --artifacts_root artifacts --host 127.0.0.1 --port 13579
```

Dashboard only:
```bash
npm -C dashboard install
npm -C dashboard run dev
```

---

## Product Surface

Single system: FastAPI server + React dashboard + artifact store.

### Dashboard Tabs

- **Nano:** Branching sessions, update event logs
- **Text:** TTT Sentry safety monitor
- **Train:** Offline training jobs, live loss curves
- **Chat:** Per-session fast weights, TTT during conversation

### Artifact Layout

```
artifacts/
├── base/                    # Nano base checkpoint
├── sessions/                # Nano branching sessions
├── text_runs/               # TTT Sentry monitor runs
├── text_models/             # Offline-trained LMs
└── text_sessions/           # Chat sessions (fast weights + traces)
```

The dashboard reads directly from these files. No hidden database.

---

## Nano Domain (SSM + Branching Sessions)

The "most blended" TTT×SSM implementation: online plastic matrices inside an SSM with persistence and branching.

### Architecture

- Core recurrence: diagonal stable SSM
- Plastic parameters: `W_u`, `B`, `W_o` (2D matrices only)
- Stability parameter: frozen (`A = -softplus(a_raw)`)

### Branching

Sessions are explicitly branchable. `fork_session` clones plastic weights and optionally optimizer momentum. This enables counterfactual comparisons: same starting point, different update histories.

---

## Text Domain

Two parallel implementations sharing utilities but with different safety coverage:

### Text World A — TTT Sentry

The canonical safety-instrumented update loop.

**Model:** `ToyTTTModel`
- Only `adapter.weight` is plastic
- Everything else frozen
- Regex tokenization (not BPE)

**Safety Loop (per chunk):**
1. Compute objective loss
2. Compute write pressure (adapter grad norm)
3. Compute auxiliary signals (compression proxy, canary alignment)
4. Gate decision (allow/block)
5. Apply update or SPFW-projected update
6. Post-update canary rollback check

**Gate (`ttt/core/gate.py`):**
- Blocks on low entropy, low diversity, blobs, instruction overrides, OOD + heavy write

**Rollback (`ttt/core/rollback.py`):**
- Probes canary before/after update
- Rolls back on z-score or absolute delta threshold breach

**Red Team (`ttt/attacks/red_team.py`):**
- Adversarial search for payloads that bypass gate but damage canary

### Text World B — TinyLM Offline Training

Trainable tiny LM for the Train + Chat tabs.

**Model:** `TinyLm`
- Recurrent backbone (SSM default, GRU fallback)
- No attention layers
- BPE tokenization

**Training:**
```bash
python -m ttt.text_lm.train --corpus_dir training_data --steps 2000
```

### Text World B — Chat Sessions

Fast weights as a per-session context window.

**Per message:**
1. Encode prompt
2. Muon steps on context net only (core frozen)
3. Generate with `hidden = core(tokens) + context(hidden)`
4. Persist `context_state.pt`, `optim_state.pt`, `trace.jsonl`

**Current safety coverage:** SPFW projection available, gate/rollback not yet integrated.

### Sleep Consolidation

Offline replay of chat traces into core model.

```bash
python -m ttt.text_lm.sleep --artifacts_root artifacts
```

**What it does:**
1. Load base model
2. Harvest traces from sessions that used that model
3. Optionally mix with core knowledge (`--core_path`, `--core_ratio`)
4. Train backbone + layernorm only (embed + head frozen)
5. Write candidate checkpoint with `status="sleep_candidate"`

**What it doesn't do yet:**
- Auditor filtering (approve/reject memories)
- Importance-weighted delta transfer
- Adapter residual initialization for next day

---

## Safety Coverage Matrix

| Mechanism | TTT Sentry (A) | Chat (B) | Nano |
|-----------|:--------------:|:--------:|:----:|
| Online TTT updates | ✓ | ✓ | ✓ |
| Gate (entropy/blob/override) | ✓ | ✗ | ✗ |
| Rollback (canary probe) | ✓ | ✗ | ✓ |
| Directional monitoring | ✓ | via SPFW | ✓ |
| SPFW projection | ✓ | ✓ | ✗ |
| Sleep consolidation | ✗ | ✓ | ✗ |

---

## Training Data + Scaling

### Corpus Location

Put files under `training_data/`. Trainer recursively loads `*.txt/*.md/*.text/*.tex/*.rst`.

### Recommended Starter

```bash
git lfs install
git clone https://huggingface.co/datasets/roneneldan/TinyStories training_data/TinyStories
cp training_data/TinyStories/TinyStoriesV2-GPT4-valid.txt training_data/
```

### Memory

- Token buffer: ~2 bytes per token ID
- 100M tokens ≈ 200MB buffer
- 1B tokens ≈ 2GB buffer

For large corpora, pretrain tokenizer on a subset and reuse via `--tokenizer`.

---

## Troubleshooting

**404 on Train/Chat:** Stale server on same port, or no trained model exists.

**"No usable text model":** Train one first. Requires `checkpoint.pt` + `tokenizer.json`.

**Git LFS pointers:** Run `git -C training_data/TinyStories lfs pull`.

**Sleep fails "corpus too small":** Need enough traces to sample `--seq_len` sequences. Generate more chat turns or provide `--core_path`.

---

## Repo Map

```
ttt_ssm_eval/
├── start.sh                           # Single entry point
├── dashboard/                         # React UI
├── ttt_ssm_nano/
│   ├── phase0_muon.py                 # Single-run sandbox
│   ├── phase1_branching_muon.py       # Branching sessions
│   └── artifacts_api/                 # FastAPI server
├── ttt/
│   ├── core/                          # ToyTTTModel + gate/rollback/SPFW
│   ├── monitors/                      # TTT Sentry
│   ├── attacks/                       # Red team harness
│   ├── optim/                         # Muon
│   └── text_lm/                       # TinyLM + chat + sleep
├── run_monitor.py                     # CLI for TTT Sentry
├── training_data/                     # Corpora (gitignored)
└── artifacts/                         # Outputs (gitignored)
```

### Key Files

**Safety:**
- `ttt/core/gate.py` — Pre-update gate
- `ttt/core/rollback.py` — Post-update canary rollback
- `ttt/core/spfw.py` — Safety-projected fast weights
- `ttt/monitors/gradient.py` — Instrumented update loop

**Models:**
- `ttt/core/model.py` — ToyTTTModel
- `ttt/text_lm/model.py` — TinyLm
- `ttt/text_lm/context.py` — Fast context nets

**Learning:**
- `ttt/text_lm/train.py` — Offline training
- `ttt/text_lm/ttt_chat.py` — Online TTT in chat
- `ttt/text_lm/sleep.py` — Sleep consolidation

---

## What's Next

- [ ] Gate + rollback integration into chat TTT
- [ ] Auditor filtering in sleep (LLM review of traces)
- [ ] Importance-weighted consolidation (not just replay)
- [ ] Adapter residual from consolidated delta (meta-learning bias)
- [ ] Dashboard "flag for review" → `flagged_prompts.json` → red team loop
- [ ] Anomaly detection on update logs (surface outliers for human review)