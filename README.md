# ttt_ssm_eval — TTT × SSM Eval Lab

This repo is a working prototype + eval harness for a **blended TTT×SSM architecture** that is not a standard, off‑the‑shelf design anywhere else:

- A **slow / core recurrent model** (SSM-first; GRU fallback only).
- A **fast plastic “context net”** whose weights update online (TTT) and function as a **weight-based context window**.

There are no transformers/attention layers in the text stack here. “Context” is implemented via **fast weights**, not attention.

[![Dashboard](./assets/dashboard_screenshot.png)](./assets/dashboard_screenshot.png)
[![TTT-SSM Architecture Infographic](./assets/infographic.png)](./assets/infographic.png)

## What Exists (Concrete) vs What Is Claimed (Thesis)

This repo is explicitly trying to make something “real” while staying honest about what’s implemented:

- **Exists**:
  - A Nano “git-for-plastic-weights” artifact store + UI (branching sessions, update logs).
  - A TTT safety monitor (“TTT Sentry”) with gate + rollback + directional signals.
  - A tiny local LM (BPE) that can be trained offline (Muon), and a per-session fast-weight context net that adapts during chat (Muon).
  - A prototype **sleep consolidation** path that replays chat traces into the slow/core model (Muon) and writes a new candidate checkpoint.
- **Thesis**:
  - “Fast weights behave like a context window” (weight-based memory).
  - Sleep becomes **selective and delta-based** (transfer only the weight changes that matter from fast→slow), rather than the current first-pass trace-replay consolidation.

## Table Of Contents

- Quick Start
- Product Surface (UI + API)
- Service Layer (orchestration)
- Artifact Store (schema + layout)
- Nano Domain (SSM + branching sessions)
- Text Domain
  - World A: TTT Sentry (ToyTTTModel + monitor + red-team)
  - World B: TinyLM offline training (BPE + Muon)
  - World B: Chat sessions (fast weights updated online)
  - Sleep consolidation (fast→slow)
- Safety Coverage Matrix (what applies where)
- Training Data + Scaling (tokens/words/memory)
- Troubleshooting
- Repo Map (key files)
- Legacy / Archive

## Quick Start

### Install

```bash
python -m pip install -e .
```

Runtime dependencies live in `pyproject.toml`.

### Start Everything (API + Dashboard)

```bash
./start.sh
```

Defaults:
- API: `http://127.0.0.1:13579`
- Dashboard: `http://127.0.0.1:5173`

Environment knobs:
- `ARTIFACTS_ROOT=/path/to/artifacts` (relocate all outputs)
- `NANO_API_HOST=127.0.0.1` / `NANO_API_PORT=13580`
- `DASHBOARD_HOST=127.0.0.1` / `DASHBOARD_PORT=5174`
- `TEXT_LM_DEVICE=auto|cpu|mps` (text model device selection)

`./start.sh` also generates a tiny Nano demo artifact store if `artifacts/base/base_checkpoint.pt` is missing.

### Manual start (optional)

API only:

```bash
python -m ttt_ssm_nano.artifacts_api --artifacts_root artifacts --host 127.0.0.1 --port 13579
```

Dashboard only:

```bash
npm -C dashboard install
npm -C dashboard run dev
```

When running the dashboard manually, set `VITE_NANO_API_URL` if the API isn’t at the default URL:

```bash
VITE_NANO_API_URL="http://127.0.0.1:13579" npm -C dashboard run dev
```

### If pyenv is installed

`./start.sh` intentionally does not assume `python3.11` exists as a shim. Use:

```bash
pyenv shell 3.11.7
python -m pip install -e .
```

## Product Surface (One Repo, One UI)

This repo is meant to feel like a single system, not two repos taped together:

- **FastAPI server**: `ttt_ssm_nano/artifacts_api/`
- **React dashboard**: `dashboard/`
- **Single start script**: `start.sh`
- **Single artifact root**: `artifacts/` (gitignored by default)

### Dashboard Tabs (Current)

- **Nano** tabs: inspect branching sessions + update events from `ttt_ssm_nano/phase1_branching_muon.py`.
- **Text** tab: run the TTT safety monitor (“TTT Sentry”) and persist runs.
- **Train** tab: start offline training jobs for a tiny local LM (Muon) and watch live loss + grad-norm.
- **Chat** tab: create per-session fast-weight state and do **TTT updates during chat** (fast weights only).

### API Endpoints (Current)

Health + Nano:
- `GET /api/health`
- `GET /api/index`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `POST /api/sessions/{parent_session_id}/fork`

Text monitor (“TTT Sentry”):
- `GET /api/text/runs`
- `GET /api/text/runs/{run_id}`
- `POST /api/text/runs` (runs monitor and persists)
- `POST /api/ttt/monitor` (legacy alias; still persists)

Text LM (offline training jobs):
- `POST /api/text/train`
- `GET /api/text/train/jobs`
- `GET /api/text/train/{model_id}/status`
- `GET /api/text/train/{model_id}/metrics`
- `POST /api/text/train/{model_id}/cancel`

Text LM (loading + simple generation; no fast weights):
- `GET /api/text/models`
- `POST /api/text/generate`

Text LM (TTT chat sessions; fast weights):
- `GET /api/text/sessions`
- `POST /api/text/sessions`
- `POST /api/text/sessions/{session_id}/chat`
- `POST /api/text/sessions/{session_id}/reset`

## Service Layer (orchestration)

Most “business process” logic (loading models, running updates, persisting state) lives in the FastAPI server code, not inside the raw PyTorch modules.

Text training jobs:
- API router: `ttt_ssm_nano/artifacts_api/routers/text_train.py`
- Manager: `ttt_ssm_nano/artifacts_api/text_train_manager.py`
- Implementation: spawns `python -m ttt.text_lm.train ...` as a subprocess using the same interpreter as the server.
- Progress: Train tab polls `train_log.jsonl` via `GET /api/text/train/{model_id}/metrics`.

Text chat sessions:
- API router: `ttt_ssm_nano/artifacts_api/routers/text_chat.py`
- Service: `ttt_ssm_nano/artifacts_api/text_chat_service.py`
- Storage: `ttt/text_lm/session_store.py`
- Mechanics per call: load session → load model/tokenizer → run TTT updates on fast weights → generate → persist `context_state.pt` + `optim_state.pt` → append `events.jsonl` + `trace.jsonl`.

## Artifact Store (What Gets Written)

Everything important is written to disk under the artifact root (default `./artifacts`):

```
artifacts/
├── base/                               # Nano base checkpoint
│   ├── base_checkpoint.pt
│   └── base_meta.json
├── sessions/                            # Nano sessions (“git for plastic weights”)
│   ├── index.json
│   └── <session_id>/
│       ├── meta.json
│       ├── metrics.json
│       ├── plastic_state.pt
│       ├── optim_state.pt
│       ├── update_events.jsonl
│       └── runs/<run_id>/
│           ├── per_step.csv
│           └── update_events.json
├── text_runs/                           # Text monitor runs (TTT Sentry)
│   ├── index.json
│   └── <run_id>/
│       ├── meta.json
│       ├── request.json
│       ├── summary.json
│       ├── events.json
│       └── input.txt
├── text_models/                         # Offline-trained tiny LMs (BPE)
│   ├── index.json
│   └── <model_id>/
│       ├── tokenizer.json
│       ├── config.json
│       ├── checkpoint.pt
│       └── train_log.jsonl
└── text_sessions/                       # Chat sessions: fast weights + optimizer state
    ├── index.json
    └── <session_id>/
        ├── meta.json
        ├── context_state.pt
        ├── optim_state.pt
        ├── events.jsonl
        └── trace.jsonl
```

The dashboard reads from these files (no hidden DB).

## Nano Domain (SSM + Branching Sessions)

Nano is the “most blended” TTT×SSM track in this repo right now: online plastic matrices inside an SSM, plus persistence + branching semantics.

Primary scripts:
- `ttt_ssm_nano/phase0_muon.py`
  - single-run sandbox (“hidden‑μ physics + diagonal stable SSM + online weight updates”)
  - includes rollback-like semantics in-script
- `ttt_ssm_nano/phase1_branching_muon.py`
  - persistent artifact store + session branching (“git for plastic weights”)

### Phase 1: Plastic vs fixed (by design)

Phase 1 is intentionally constrained for interpretability:

- Core recurrence is a diagonal stable SSM (`DiagStableSSM`).
- Only 2D matrices are plastic:
  - `W_u`, `B`, `W_o`
- Stability parameter is frozen (`A = -softplus(a_raw)`).

### What “branching” means

Phase 1 sessions are explicitly branchable:

- `fork_session` clones a parent session’s plastic weights (and optionally optimizer momentum).
- Runs write comparable baselines (same trajectory; differences are meaningful):
  - base/no-update
  - session/no-update
  - session/with-updates

These artifacts are what the Nano UI visualizes.

## Text Domain

The `ttt/` package contains **two parallel “worlds”**:

- **World A**: a safety-eval harness around a toy model and a fully-instrumented update loop.
- **World B**: a trainable tiny LM (BPE) + chat sessions with per-session fast weights.

They share some utilities (Muon, SPFW), but they are not the same codepath, and safety coverage is currently different.

### Text World A — TTT Sentry (ToyTTTModel + monitor + red-team)

This is the canonical reference safety loop in the repo.

Entry points:
- CLI: `run_monitor.py`
- API: `POST /api/text/runs` (persists under `artifacts/text_runs/`)
- UI: **Text** tab

#### Model: `ToyTTTModel` (adapter-only plasticity)

Defined in `ttt/core/model.py`.

Architecture:

```
token_ids
  -> Embedding
  -> Backbone (SSM default; GRU fallback)
  -> LayerNorm
  -> Residual + Adapter           # only the adapter is plastic
  -> Vocab head
```

Plasticity:
- only `adapter.weight` receives gradients and updates
- everything else is frozen

Tokenization:
- regex tokenization + stable hashing to ids (`blake2b`), not BPE
- see `ttt/core/model.py: tokenize()` and `ttt/core/model.py: token_to_id()`

Backbones:
- `DiagonalSelectiveSSM` (default) in `ttt/core/backbone.py`
- `GRUBackbone` (fallback) in `ttt/core/backbone.py`

Objectives:
- `AR` and `MLM` in `ttt/core/objective.py`

#### The safety loop (what is actually wired)

Implemented in `ttt/monitors/gradient.py`.

Per chunk:
1) compute objective loss
2) compute “write pressure” (adapter grad norm)
3) compute auxiliary signals (compression proxy, canary gradient alignment)
4) gate decision (hard allow/block by default)
5) apply update (SGD) or apply SPFW-projected update
6) post-update canary rollback (“transaction semantics”)

Safety components:

- Pre-update gate: `ttt/core/gate.py`
  - blocks updates on low entropy, low diversity, blobs, instruction overrides, or OOD+heavy-write
- Post-update rollback: `ttt/core/rollback.py`
  - probe canary before/after; robust z-score + absolute delta thresholds
- Directional monitoring + compression proxy: `ttt/monitors/signals.py`
  - canary gradient alignment (cos + dot)
  - zlib compression ratio as a “Kolmogorov-ish” proxy
- SPFW (Safety-Projected Fast Weights): `ttt/core/spfw.py`
  - projects task gradients into an intersection of half-spaces defined by canary gradients

#### Red-team harness (targets World A)

`ttt/attacks/red_team.py` is an adversarial optimization harness targeting **ToyTTTModel + the TTT Sentry loop**, not the chat LM.

It searches for payloads that:
- keep write pressure under thresholds
- keep “normal-ish” statistics
- still cause canary damage

### Text World B — TinyLM offline training (BPE + Muon)

This is the tiny “real-ish” LM used by Train + Chat.

Implementation:
- model: `ttt/text_lm/model.py` (`TinyLm`)
- training: `ttt/text_lm/train.py` (`python -m ttt.text_lm.train`)
- tokenizer: `ttt/tokenization/bpe.py` (dependency-free byte-level BPE)
- optimizer: `ttt/optim/muon.py` (prefers `torch.optim.Muon` when compatible, otherwise uses a Muon-style fallback)

`TinyLm` architecture:

```
token_ids
  -> Embedding
  -> Backbone (SSM default; GRU fallback)
  -> LayerNorm
  -> Head -> logits
```

No attention layers.

Training outputs:
- `artifacts/text_models/index.json`
- `artifacts/text_models/<model_id>/checkpoint.pt`
- `artifacts/text_models/<model_id>/train_log.jsonl` (the Train tab plots this)

CLI example (same thing the Train tab triggers):

```bash
python -m ttt.text_lm.train --corpus_dir training_data --device auto --steps 2000
```

### Text World B — Chat sessions (fast weights updated online)

Chat sessions treat fast weights as a per-session context window.

Core components:

- Context nets:
  - `ttt/text_lm/context.py` (`LinearContextNet`, default)
  - `ttt/text_lm/fast_memory.py` (`LowRankFastMemoryContext`)
- TTT update loop + sampling:
  - `ttt/text_lm/ttt_chat.py`
- Session persistence:
  - `ttt/text_lm/session_store.py`
  - stored under `artifacts/text_sessions/`
  - `events.jsonl` is compact (previews + update stats); `trace.jsonl` stores full prompt/completion for replay during sleep.

Mechanics per message:

1) encode prompt with the session’s core model tokenizer
2) run a few Muon steps that update **only** the context net params
3) generate tokens with `core + context`:
   - hidden `h = core.hidden(tokens)`
   - residual `h' = h + context(h)`
   - logits from `h'`

This is the “two nets” rule in concrete form:

- **slow/core weights**: trained offline (Train tab)
- **fast/context weights**: updated online (Chat tab)

#### Safety in chat (honest state)

In the chat TTT loop (`ttt/text_lm/ttt_chat.py`):
- Muon updates are used for fast weights.
- SPFW projection can be enabled via session config (uses canary gradients).
- The rule-based gate (`ttt/core/gate.py`) and rollback (`ttt/core/rollback.py`) are **not yet integrated** into chat updates.

That means:
- The strongest “full safety stack” is currently in **TTT Sentry** (World A).
- Chat sessions currently prioritize “fast-weight adaptation mechanics + persistence” over gate/rollback policy.

### Sleep consolidation (fast→slow)

Sleep is a prototype mechanism for turning chat history into slow/core updates.

Current implementation is **trace-replay consolidation** (not delta-based transfer):

- Chat turns are appended to `artifacts/text_sessions/<session_id>/trace.jsonl` (prompt + completion).
- Sleep reads those traces for a specific base model id, mixes them with optional “core” text, and runs a small offline training loop to update a subset of the slow model parameters.

CLI entrypoint:

```bash
python -m ttt.text_lm.sleep --artifacts_root artifacts
```

What it does (concrete):

1) Load a base model from `artifacts/text_models/<base_model_id>/`.
2) Harvest up to `--max_memories` chat turns from sessions that used that base model:
   - reads `artifacts/text_sessions/*/trace.jsonl`
   - only includes sessions whose `meta.json` model_id matches the chosen base model id
3) Optionally load additional “core” text via `--core_path`:
   - `.txt` is treated as raw text
   - `.jsonl` expects records with a `"text"` field
4) Mix “core” and “memory” texts according to `--core_ratio` (0..1) and encode with the base model’s BPE tokenizer.
5) Train for `--train_steps` steps on next-token cross entropy, using Muon at a very low LR by default.
6) Write a new model under `artifacts/text_models/<sleep_id>/`:
   - copies `tokenizer.json` + `config.json` from the base model
   - writes `checkpoint.pt` (updated weights)
   - writes `sleep_manifest.json` (pre/post eval loss on core and mix)
   - registers the model in `artifacts/text_models/index.json` with `status="sleep_candidate"`

What is updated during sleep (current design):

- Only `TinyLm.backbone` and `TinyLm.ln` are trained (embed + head remain frozen).
- This is a first-pass “consolidate into dynamics” approach rather than a full fine-tune.

Interaction with Chat defaults:

- `TextChatService` avoids automatically selecting models marked as sleep candidates when it chooses “latest usable”.
- A sleep model can still be selected explicitly by passing `model_id` when creating a chat session.

## Safety Coverage Matrix (Current Wiring)

| Mechanism | TTT Sentry (World A) | Chat TTT (World B) | Nano (Phase 1) |
|---|---:|---:|---:|
| Online updates (TTT) | ✅ adapter updates | ✅ fast weights | ✅ plastic matrices |
| Gate (entropy/blob/override/OOD) | ✅ | ❌ | ❌ |
| Rollback (canary probe) | ✅ | ❌ | ✅ (transaction-like update events) |
| Directional monitoring | ✅ metrics | ✅ only via SPFW canaries | ✅ update logs |
| SPFW projection | ✅ optional | ✅ optional | ❌ |

## Consolidation (“sleep”) Status

| Mechanism | World A (TTT Sentry) | World B (TinyLM) | Nano |
|---|---:|---:|---:|
| Fast weights during interaction | ✅ adapter updates | ✅ context net updates | ✅ plastic matrices |
| Sleep consolidation (fast→slow) | ❌ | ✅ `python -m ttt.text_lm.sleep` | ❌ |

## Training Data + Scaling (tokens/words/memory)

### Where the corpus lives

Put corpora under `training_data/` and point training at it (the UI defaults to this).

- `training_data/` is gitignored; it is intended to be synced locally however desired.
- The trainer recursively loads `*.txt/*.md/*.text/*.tex/*.rst`.
- No JSONL preprocessing is required; the trainer consumes raw UTF‑8 text files.

### TinyStories (recommended starter corpus)

```bash
git lfs install
git clone https://huggingface.co/datasets/roneneldan/TinyStories training_data/TinyStories
cp training_data/TinyStories/TinyStoriesV2-GPT4-valid.txt training_data/
```

If a corpus file begins with `version https://git-lfs.github.com/spec/v1`, it’s an LFS pointer; run `git -C training_data/TinyStories lfs pull` and retry.

### “How many words can the corpus be?”

Two constraints dominate:

1) **Tokenizer training cost (BPE)**: merge counting can get expensive on multi‑GB corpora. For large corpora, pretrain the tokenizer on a subset and reuse it via `--tokenizer`.
2) **Token buffer RAM**: the trainer encodes the corpus into a compact `uint16` token-id array in memory.

Rough memory rule:
- ~2 bytes per token-id (plus model/optimizer/activation memory).
- 100M tokens ≈ ~200MB for the token buffer alone.
- 1B tokens ≈ ~2GB for the token buffer alone.

Word↔token ratios depend on content. With this byte-level BPE, English often lands around ~1–2 tokens/word, but code/math/Unicode-heavy text can be higher.

### What “training in chat” means here

Chat updates the **fast context net** only. The slow/core model is **frozen during chat**.

Slow/core weights change only via offline processes:
- Offline training jobs (`ttt.text_lm.train`, Train tab)
- Sleep consolidation (`ttt.text_lm.sleep`, which writes a new candidate checkpoint)

That is deliberate: fast weights are treated as a mutable context window, not as the long-term model.

## Troubleshooting

### UI shows `404 Not Found` when clicking Train/Chat

Most common causes:

- The API server isn’t the one started by `./start.sh` (stale server running on the same port). `./start.sh` tries to detect this by probing endpoints.
- No trained text model exists yet. Chat and `/api/text/generate` require a checkpoint in `artifacts/text_models/`.

### Chat tab says “No usable text model”

Train at least one model first (Train tab or CLI). Required files per model:
- `artifacts/text_models/<model_id>/checkpoint.pt`
- `artifacts/text_models/<model_id>/tokenizer.json`

### Git LFS pointers instead of real data

If a corpus file begins with:

```
version https://git-lfs.github.com/spec/v1
```

then Git LFS content hasn’t been materialized yet. Run `git -C training_data/TinyStories lfs pull`.

### Training uses “way too much” RAM

For large corpora, the expensive parts are:
- tokenizer training (BPE)
- the in-memory token-id buffer

The trainer stores token ids as compact `uint16` (≈2 bytes/token-id). If RAM spikes into tens of GB on a modest corpus, it likely indicates an outdated checkout or a tokenization/path mistake.

### Sleep consolidation fails with “sleep corpus too small”

`ttt.text_lm.sleep` requires enough tokens to sample sequences of length `--seq_len` (default 128).
If there aren’t enough chat traces yet, either:

- generate more chat turns in sessions for the base model (so `trace.jsonl` exists and grows), and/or
- provide a `--core_path` and set a higher `--core_ratio`, and/or
- reduce `--seq_len`.

## Repo Map (Key Files)

```
ttt_ssm_eval/
├── start.sh
├── dashboard/                              # React UI
├── ttt_ssm_nano/
│   ├── phase0_muon.py                      # single-run Nano sandbox
│   ├── phase1_branching_muon.py            # artifact store + branching sessions
│   └── artifacts_api/                      # FastAPI server for Nano + Text
├── ttt/
│   ├── core/                               # ToyTTTModel + gate/rollback/SPFW
│   ├── monitors/                           # TTT Sentry monitor + signals
│   ├── attacks/                            # red-team harness (ToyTTTModel)
│   ├── optim/                              # Muon (+ fallback)
│   └── text_lm/                            # TinyLM training + chat sessions
├── run_monitor.py                          # CLI for TTT Sentry
├── training_data/                          # local corpora (gitignored)
├── artifacts/                              # outputs (gitignored)
└── .archive/                               # legacy code + old notes
```

### Key modules (one-liners)

Nano:
- `ttt_ssm_nano/phase0_muon.py`: single-run Nano sandbox with rollback-like semantics.
- `ttt_ssm_nano/phase1_branching_muon.py`: persistent artifact store + branching sessions + update logs.
- `ttt_ssm_nano/artifacts_api/app.py`: FastAPI app wiring and shared services.
- `ttt_ssm_nano/artifacts_api/reader.py`: reads Nano artifacts for the dashboard.
- `ttt_ssm_nano/artifacts_api/actions.py`: fork-session action invoked by the API.

TTT Sentry (World A):
- `run_monitor.py`: CLI wrapper around the text monitor.
- `ttt/core/model.py`: `ToyTTTModel` + regex+hash tokenization.
- `ttt/core/backbone.py`: GRU + minimal diagonal selective SSM backbones.
- `ttt/core/objective.py`: AR + MLM objective implementations.
- `ttt/core/gate.py`: rule-based pre-update gate (entropy/diversity/blob/override/OOD+heavy-write).
- `ttt/core/rollback.py`: canary-loss probe + robust z-score rollback logic.
- `ttt/core/spfw.py`: SPFW projection (half-space constraints from canary gradients).
- `ttt/monitors/gradient.py`: canonical “gate/rollback/SPFW” monitoring loop and event schema.
- `ttt/monitors/signals.py`: compression proxy + canary-gradient alignment helpers.
- `ttt/attacks/red_team.py`: adversarial payload search against the World A monitor.

TinyLM (World B):
- `ttt/tokenization/bpe.py`: dependency-free byte-level BPE (train/encode/decode).
- `ttt/text_lm/model.py`: `TinyLm` core model (recurrent/SSM).
- `ttt/text_lm/train.py`: offline training loop that writes `artifacts/text_models/…`.
- `ttt/text_lm/corpus.py`: corpus ingestion + sampling helpers (compact token buffer).
- `ttt/text_lm/context.py`: `LinearContextNet` and session config for fast weights.
- `ttt/text_lm/fast_memory.py`: optional low-rank fast-memory context module.
- `ttt/text_lm/ttt_chat.py`: fast-weight update loop + sampling (core weights frozen).
- `ttt/text_lm/sleep.py`: sleep consolidation (replay chat traces into slow/core; writes a candidate model).
- `ttt/text_lm/store.py`: `artifacts/text_models/` index + paths.
- `ttt/text_lm/session_store.py`: `artifacts/text_sessions/` index + persistence.

Optimizers:
- `ttt/optim/muon.py`: Muon wrapper + fallback (supports non-2D params).

Dashboard:
- `dashboard/src/components/tabs/TextTrainTab.tsx`: training UI + live charts.
- `dashboard/src/components/tabs/ChatTab.tsx`: TTT sessions UI (fast-weight chat).
- `dashboard/src/components/tabs/TextMonitorTab.tsx`: TTT Sentry UI (monitor runs + analytics).

## Legacy / Archive

Anything deprecated lives under `.archive/`. The active UI is `dashboard/` only.
