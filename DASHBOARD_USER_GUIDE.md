# TTT-SSM Dashboard User Guide

## Quick Start

### Starting the Dashboard

```bash
# From the project root
./start.sh

# Dashboard will be available at:
# - React UI: http://localhost:5173
# - API: http://localhost:13579
```

### Running Text Monitoring

```bash
# CLI demos
python run_monitor.py --demo                    # Mixed benign/adversarial
python run_monitor.py --demo_high_entropy       # High-entropy blob attacks

# Custom text
python run_monitor.py --text "Your text here"
python run_monitor.py --file input.txt

# Architecture variants
python run_monitor.py --demo --backbone ssm     # SSM instead of GRU
python run_monitor.py --demo --objective mlm    # MLM instead of AR
```

---

## Dashboard Navigation

### Main Tabs

1. **Session Tree** - Git-like visualization of session branching (Phase 1 artifacts)
2. **Overview** - Session metrics, MSE comparison, learning breakdown
3. **Text Monitor** - ⭐ **NEW VISUALIZATIONS HERE** ⭐
4. **Chat** - TTT context net conversations
5. **Train** - Text LM training with Muon optimizer
6. **Weights** - Heatmap visualizations of plastic weights
7. **Transactions** - Update event timeline
8. **Sessions** - Session catalog and metadata

---

## Text Monitor Tab - New Features

### Input Panel (Left Side)

**Controls**:
- **Text Input**: Paste or type text to analyze
- **Load Demo**: Pre-loaded mixed benign/adversarial text
- **High Entropy**: Base64 blob pattern (tests blob detection)

**Configuration**:
- **Backbone**: GRU (default) or SSM
- **Objective**: AR (autoregressive) or MLM (masked language model)
- **Chunk Tokens**: Number of tokens per chunk (default: 128)
- **Enable gate**: Pre-update blocking (default: ON)
- **Enable rollback**: Post-update reversion (default: ON)
- **Canary gradient alignment**: Directional monitoring (default: ON)

**Action**:
- Click **"Run Monitor"** to analyze the text

---

### Results Panel (Right Side)

**Summary Badges**:
- **Flagged**: Chunks with anomalous signals
- **Blocked**: Chunks rejected by pre-update gate
- **Rollbacks**: Chunks that caused model corruption (reverted)

**Per-Chunk Event Cards**:
Each chunk displays:
- **Status**: ok / flagged / blocked / rollback
- **Metrics**: loss, grad norm, update norm, canary delta
- **Compression ratio**: Color-coded (red if < 0.5)
- **Alignment**: Canary gradient cosine similarity (red if > 0.3)
- **Chunk preview**: First 200 characters
- **Gate reasons** (if blocked): Interactive tags
  - `low_entropy`: Repeated tokens
  - `low_diversity`: < 10% unique tokens
  - `blob_detected`: Base64/hex/minified code
  - `instruction_override`: Jailbreak patterns
  - `ood_heavy_write`: High loss + high gradient
- **Rollback reasons** (if rolled back)

---

### Signal Summary Panel (NEW ✅)

**Overview Metrics**:
- **Total Chunks**: Number of chunks analyzed
- **Gate Blocked**: How many chunks were blocked pre-update (with %)
- **Rollbacks**: How many updates were reverted (with %)
- **Low Compression**: Chunks with compression ratio < 0.5 (blob attacks)
- **Harmful Alignment**: Chunks with canary cos > 0.3 (directional attacks)

---

### Compression Ratio Timeline (NEW ✅)

**What it shows**:
- zlib compression ratio per chunk (Kolmogorov complexity proxy)
- **Low ratio (< 0.5)**: Random/blob data, suspicious
- **High ratio (0.7-0.9)**: Natural text with patterns

**Visual Cues**:
- **Green dots**: Normal text (0.7-0.9)
- **Yellow dots**: Medium compression (0.5-0.7)
- **Red dots**: Suspicious (< 0.5)
- **Orange outline**: Gate blocked this chunk

**Reference line**: Horizontal line at 0.5 (suspicious threshold)

---

### Directional Monitoring Card (NEW ✅)

**Purpose**: Detects harmful update directions beyond magnitude thresholds.

**Summary Stats**:
- **Harmful Aligned**: Chunks with cos > 0.3 (aligned with canary harm)
- **Benign Aligned**: Chunks with cos < -0.3 (opposes canary harm)
- **Neutral**: Chunks with |cos| < 0.3 (orthogonal)
- **Max Harmful Alignment**: Worst offender with chunk preview

**Scatter Plot**: Gradient Magnitude vs Canary Alignment
- **X-axis**: Gradient norm (magnitude of adapter gradient)
- **Y-axis**: Canary alignment (cosine similarity, -1 to 1)
- **Point colors**:
  - 🔴 Red: Harmful (cos > 0.3) - aligned with canary corruption
  - 🟡 Yellow: Neutral (|cos| < 0.3) - orthogonal
  - 🟢 Green: Benign (cos < -0.3) - opposes canary corruption
- **Point size**: Scaled by canary gradient norm
- **Tooltip**: Hover for chunk preview + metrics

**Timeline**: Canary Alignment Over Chunks
- Blue line: Cosine similarity trajectory
- Red horizontal line: Harmful threshold (+0.3)
- Green horizontal line: Benign threshold (-0.3)
- Red dots: Rollback events

**Alert Box** (if applicable):
- Shows chunk with highest harmful alignment
- Preview of problematic text

---

### Canary Loss Card (NEW ✅)

**Purpose**: Measures model corruption on fixed reference text after updates.

**Summary Stats**:
- **Total Rollbacks**: How many updates were reverted
- **Avg Canary Δ**: Mean corruption per chunk
- **Max Canary Δ**: Worst corruption event with chunk preview
- **Corruption Rate**: % of chunks with delta > threshold

**Dual-Line Chart**: Canary Loss Before/After Updates
- **Blue line**: Loss on canary text BEFORE update
- **Green line**: Loss on canary text AFTER update (if committed)
- **Red dots**: Rollback events (update was reverted)

**Delta Distribution Histogram**:
- 20 bins showing distribution of canary deltas
- **Green bars**: Safe updates (delta < 1.0)
- **Orange bars**: Corrupt updates (delta ≥ 1.0)
- **Vertical line**: Rollback threshold (default: 1.0)

**Alert Box** (if applicable):
- Shows chunk with maximum corruption
- Indicates whether it was rolled back

---

### Gate Analytics Card (NEW ✅)

**Purpose**: Analyzes pre-update gate blocking decisions.

**Summary Stats**:
- **Total Blocked**: How many chunks the gate rejected
- **Low Entropy**: Chunks with entropy < 1.0
- **Low Diversity**: Chunks with diversity < 10%
- **Unique Reasons**: Number of distinct rejection types

**Pie Chart**: Gate Rejection Reasons
- Breakdown by reason: low_entropy, low_diversity, blob_detected, instruction_override, ood_heavy_write
- Color-coded slices
- Legend with counts

**Scatter Plot**: Token Entropy vs Diversity
- **X-axis**: Token entropy (Shannon entropy)
- **Y-axis**: Token diversity (fraction of unique tokens)
- **Point colors**:
  - 🟢 Green: Allowed through gate
  - 🟠 Orange: Blocked by gate
- **Reference lines**: Threshold boundaries

**Dual-Axis Timeline**: Entropy & Diversity Over Chunks
- **Left axis (blue)**: Token entropy
- **Right axis (purple)**: Token diversity
- **Orange horizontal line**: Entropy threshold (1.0)
- **Yellow horizontal line**: Diversity threshold (0.1)
- **Orange dots**: Blocked chunks

---

## Interpreting Results

### Healthy Input Pattern
```
✅ Compression ratio: 0.7-0.9 (natural text)
✅ Canary alignment: -0.3 to 0.3 (neutral/benign)
✅ Token entropy: > 1.0 (varied vocabulary)
✅ Token diversity: > 10% (not repetitive)
✅ Gate: Allowed
✅ Rollback: None
```

### Blob/Encoding Attack
```
🚨 Compression ratio: < 0.5 (random-looking data)
🚨 Token diversity: < 10% (repetitive patterns)
🚨 Token entropy: < 1.0 (limited vocabulary)
⚠️ Gate: BLOCKED (blob_detected, low_diversity)
```

### Instruction Override Attack
```
🚨 Gate: BLOCKED (instruction_override)
🚨 Canary alignment: Potentially > 0.3 (harmful)
🚨 Compression ratio: 0.5-0.7 (suspicious patterns)
⚠️ Matched patterns: "IGNORE ALL PREVIOUS INSTRUCTIONS", etc.
```

### Stealthy Directional Attack
```
⚠️ Gradient norm: < 2.5 (below magnitude threshold)
🚨 Canary alignment: > 0.3 (harmful direction!)
🚨 Canary delta: > 1.0 (corruption detected)
✅ Rollback: TRIGGERED (model reverted)
```

---

## Advanced Usage

### Tuning Gate Thresholds

If you're seeing too many false positives:
```bash
python run_monitor.py --demo \
  --min_entropy_threshold 0.5 \      # Reduce entropy threshold
  --min_diversity_threshold 0.05 \   # Reduce diversity threshold
  --ood_loss_threshold 10.0          # Increase OOD threshold
```

If you're seeing attacks slip through:
```bash
python run_monitor.py --demo \
  --min_entropy_threshold 2.0 \      # Increase entropy threshold
  --min_diversity_threshold 0.15 \   # Increase diversity threshold
  --ood_loss_threshold 6.0           # Reduce OOD threshold
```

### Disabling Safety Mechanisms (for testing)

**Disable gate** (allow all updates):
```bash
python run_monitor.py --demo --disable_gate
```

**Disable rollback** (no post-update reversion):
```bash
python run_monitor.py --demo --disable_rollback
```

**Disable canary gradient** (no directional monitoring):
```bash
python run_monitor.py --demo --disable_canary_grad
```

### Testing Different Backbones

**SSM (diagonal selective state space)**:
```bash
python run_monitor.py --demo --backbone ssm
```

**GRU (gated recurrent unit)**:
```bash
python run_monitor.py --demo --backbone gru
```

### Testing Different Objectives

**AR (autoregressive, next-token prediction)**:
```bash
python run_monitor.py --demo --objective ar
```

**MLM (masked language model)**:
```bash
python run_monitor.py --demo --objective mlm --mlm_prob 0.15
```

---

## Troubleshooting

### Dashboard won't start

**Problem**: `Address already in use`
**Solution**: Kill existing processes
```bash
lsof -ti :13579 | xargs kill -9
lsof -ti :5173 | xargs kill -9
./start.sh
```

### Empty visualizations

**Problem**: Cards show "No data available"
**Solution**: Ensure you're running with canary gradient enabled (default)
```bash
python run_monitor.py --demo  # NOT --disable_canary_grad
```

### High compression ratio on obvious blob

**Problem**: Blob attack shows ratio > 0.5
**Solution**: The blob might be too short. Try longer repeated patterns:
```bash
python run_monitor.py --demo_high_entropy
```

### No rollbacks triggered

**Problem**: Canary delta high but no rollback
**Solution**: Rollback might be disabled or threshold too high
```bash
python run_monitor.py --demo \
  --rollback_abs_canary_delta 0.5  # Lower threshold
```

---

## Performance Tips

### For Large Texts (> 10KB)

Increase chunk size to reduce total chunks:
```bash
python run_monitor.py --file large.txt --chunk_tokens 256
```

### For Real-Time Analysis

Dashboard supports saved runs. To analyze programmatically:
```bash
# API endpoint
curl http://localhost:13579/api/text/runs | jq
```

### For Batch Testing

Save results to JSON:
```bash
python run_monitor.py --demo --write_json
# Creates artifacts/text_runs/<run_id>/report.json
```

---

## Keyboard Shortcuts (Dashboard)

- **Ctrl/Cmd + K**: Focus search (session selector)
- **Tab**: Navigate between tabs
- **Scroll**: Pan through charts
- **Hover**: Show tooltips
- **Click**: Select run from dropdown

---

## Saving and Loading Runs

All text monitoring runs are automatically saved to:
```
artifacts/text_runs/<run_id>/
├── report.json          # MonitorEvent[] + summary
├── meta.json            # Run configuration
└── input.txt            # Original input text
```

To reload a previous run in the dashboard:
1. Go to Text Monitor tab
2. Click the "Saved runs" dropdown
3. Select run by ID
4. Results panel populates with saved events

---

## Next Features (Coming in Phase 2 & 3)

- [ ] Multi-run comparison (side-by-side metrics)
- [ ] Real-time polling (live updates during run)
- [ ] Export to CSV/JSON (download reports)
- [ ] Dark mode toggle
- [ ] Session branching performance heatmap

---

## Getting Help

**Documentation**: See `CLAUDE.md` for command reference
**API Docs**: See `ttt_ssm_nano/artifacts_api/routers/text_runs.py`
**Issues**: File bug reports in the project repo

**Common Questions**:
- "What's a good canary alignment value?" → Between -0.3 and 0.3 is neutral/safe
- "When should I use SSM vs GRU?" → SSM for research, GRU for production-like baseline
- "How do I tune for my domain?" → Start with defaults, adjust based on false positive rate
- "Can I add custom gate checks?" → Yes, edit `ttt/core/gate.py`
