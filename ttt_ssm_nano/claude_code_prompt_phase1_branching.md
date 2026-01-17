# Follow-Up: Add Phase 1 Branching to TTT-SSM Dashboard

## Context

You already built the Phase 0 UI. Phase 1 adds **session branching** — think "git for neural network weights." Sessions can now fork from other sessions, creating a tree of weight evolution.

---

## What's New in Phase 1

### 1. Session Hierarchy (Git-Like)

Sessions now have lineage:
```json
{
  "session_id": "experiment_v2",
  "parent_session_id": "experiment_v1",   // Who I forked from
  "root_session_id": "experiment_v1",     // The original ancestor
  "created_at_unix": 1704307200,
  "last_run_at_unix": 1704393600
}
```

A session forked from base has `parent_session_id: null` and `root_session_id: self`.

### 2. Three-Way Comparison (Not Two)

Phase 0 compared: baseline vs adaptive.
Phase 1 compares **three** models on the same trajectory:

| Model | Description |
|-------|-------------|
| `base` | Original pretrained weights, no updates |
| `session_start` | This session's weights at run start, no updates |
| `adaptive` | This session's weights with online TTT updates |

This reveals:
- `base → session_start` gap = what previous runs learned (persistent learning)
- `session_start → adaptive` gap = what this run learned (online learning)

### 3. Artifact Store Structure

```
artifacts/
├── base/
│   ├── base_checkpoint.pt
│   └── base_meta.json
└── sessions/
    ├── index.json                    # All sessions summary
    └── <session_id>/
        ├── meta.json                 # Session metadata
        ├── plastic_state.pt          # Current plastic weights
        ├── optim_state.pt            # Optimizer momentum
        ├── metrics.json              # Latest metrics
        ├── update_events.jsonl       # Append-only log of ALL updates
        └── runs/
            └── <run_id>/
                ├── per_step.csv
                ├── update_events.json
                └── mse_curve.png
```

### 4. Session Index (`index.json`)

```json
{
  "schema_version": 1,
  "sessions": {
    "session_001": {
      "session_id": "session_001",
      "parent_session_id": null,
      "root_session_id": "session_001",
      "created_at_unix": 1704307200,
      "last_run_at_unix": 1704393600,
      "env_mode": "linear",
      "mu": 0.1234,
      "model_signature": "abc123..."
    },
    "session_001_fork_a": {
      "session_id": "session_001_fork_a",
      "parent_session_id": "session_001",
      "root_session_id": "session_001",
      ...
    }
  }
}
```

### 5. New Metrics Structure

```json
{
  "run_id": "run_20240103_143052",
  "seed": 1337,
  "steps": 600,
  "mu": 0.1234,
  "env_mode": "linear",
  "base_mse_mean": 0.00456,
  "session_no_update_mse_mean": 0.00234,
  "adaptive_mse_mean": 0.00089,
  "base_mse_last100_mean": 0.00567,
  "session_no_update_last100_mean": 0.00189,
  "adaptive_last100_mean": 0.00045,
  "updates_attempted": 18,
  "updates_committed": 15,
  "updates_rolled_back": 3
}
```

### 6. Append-Only Update Log (`update_events.jsonl`)

All update events across all runs, one JSON object per line:
```jsonl
{"t":32,"status":"commit","pre_loss":0.00789,"post_loss":0.00456,"grad_norm":2.34,"run_id":"run_001"}
{"t":64,"status":"rollback_loss_regression","pre_loss":0.00567,"post_loss":0.00890,"grad_norm":15.67,"run_id":"run_001"}
{"t":32,"status":"commit","pre_loss":0.00345,"post_loss":0.00234,"grad_norm":1.89,"run_id":"run_002"}
```

---

## UI Changes Required

### New Tab: Session Tree (Add as Tab 7 or replace Session Management)

**The Hero Visualization**: A tree/DAG showing session lineage.

```
[base_checkpoint]
       │
       ▼
  [session_001] ──────────────────┐
   μ=0.12, linear               │
   15 commits, 3 rollbacks        │
       │                          │
       ├──► [session_001_fork_a]  │
       │     μ=0.12, linear       │
       │     8 commits            │
       │                          │
       └──► [session_001_fork_b] ◄┘
             μ=0.12, nonlinear
             12 commits
```

**Requirements**:
- Nodes show: session_id, μ, env_mode, update stats
- Edges show parent→child relationship
- Click node to select session (loads in other tabs)
- Color code by: recency, performance improvement, or lineage depth
- Support collapsing subtrees for large session counts
- "Fork from here" button on each node

Consider using:
- D3 force-directed graph
- Tree layout (d3-hierarchy)
- Or a vertical timeline with branching

### Updated: Overview Dashboard

**Three-line MSE chart** (not two):
- Gray dashed: `base` (frozen pretrained weights)
- Blue dashed: `session_start` (session weights, no updates this run)
- Green solid: `adaptive` (session weights + online updates)

**New insight cards**:
- "Persistent Learning": % improvement from base → session_start
- "Online Learning": % improvement from session_start → adaptive
- "Total Learning": % improvement from base → adaptive

**Session lineage breadcrumb**:
```
base → session_001 → session_001_fork_a (current)
```

### Updated: Session Management Tab

**Session list table** now includes:
- parent_session_id column
- Lineage depth (how many forks from base)
- Total runs count
- Cumulative updates committed

**New actions**:
- "Fork Session" button → opens modal to name child session
- "Compare with Parent" → side-by-side view
- "View Full Lineage" → expands tree to this session

### New: Run History Panel

Sessions can have multiple runs. Add a sub-panel or accordion:

```
Session: session_001
├── run_20240103_143052 (latest)
│   └── 600 steps, 15 commits, 3 rollbacks
├── run_20240102_091534
│   └── 600 steps, 12 commits, 5 rollbacks
└── run_20240101_220815
    └── 300 steps, 8 commits, 1 rollback
```

Click run to load its data in the visualizations.

### Updated: Weight Evolution Tab

Add **"Compare to Parent"** toggle:
- Shows weight delta heatmap: `W_current - W_parent`
- Highlights which weights diverged most since fork

Add **"Compare to Base"** toggle:
- Shows cumulative drift from original pretrained weights

### New: Global Update Events Timeline

The `update_events.jsonl` file accumulates across runs. Visualize the full history:

- Horizontal timeline spanning all runs
- Each run is a segment
- Commits/rollbacks shown as events
- Useful for seeing long-term learning patterns

---

## Data Loading Changes

### Load Session Index
```typescript
interface SessionIndex {
  schema_version: number;
  sessions: Record<string, SessionSummary>;
}

interface SessionSummary {
  session_id: string;
  parent_session_id: string | null;
  root_session_id: string;
  created_at_unix: number;
  last_run_at_unix: number | null;
  env_mode: "linear" | "nonlinear";
  mu: number;
  model_signature: string;
}
```

### Build Session Tree
```typescript
function buildSessionTree(index: SessionIndex): TreeNode {
  // Root is the base checkpoint (virtual node)
  // Children are sessions where parent_session_id === null
  // Recursively attach children based on parent_session_id
}
```

### Load Run List for Session
```typescript
// List directories in artifacts/sessions/<session_id>/runs/
// Each run has: per_step.csv, update_events.json, mse_curve.png
```

---

## Visual Design Notes

### Session Tree Styling
- Base checkpoint node: special styling (gold/bronze, larger)
- Root sessions (forked from base): one color family
- Child sessions: inherit parent's hue, vary lightness
- Hover shows full metadata tooltip
- Selected node has glow/ring

### Three-Line MSE Chart
- Base line: gray, dashed, thin (it's the reference)
- Session-start line: blue, dashed, medium weight
- Adaptive line: green, solid, thick (it's the star)
- Fill area between session-start and adaptive (the "online learning" region)
- Different fill between base and session-start (the "persistent learning" region)

### Lineage Breadcrumb
- Clickable segments
- Shows → arrows between segments
- Current session is bold/highlighted
- Truncate middle if too long: `base → ... → parent → current`

---

## Summary Checklist

- [ ] Session tree visualization (DAG/tree layout)
- [ ] Three-way MSE comparison chart
- [ ] "Persistent Learning" vs "Online Learning" metrics
- [ ] Session lineage breadcrumb
- [ ] Fork session action
- [ ] Run history panel (multiple runs per session)
- [ ] Weight diff vs parent visualization
- [ ] Global update events timeline (across runs)
- [ ] Updated session list with hierarchy info
- [ ] Load/parse `index.json` and `update_events.jsonl`

Keep the Phase 0 visualizations intact — this is additive. The session tree becomes the new "home" for navigation, with the MSE chart and other views responding to selection.
