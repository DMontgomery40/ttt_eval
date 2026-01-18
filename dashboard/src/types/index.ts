// TTT-SSM Phase 1 Dashboard Types

export interface ModelConfig {
  obs_dim: number;
  act_dim: number;
  z_dim: number;
  u_dim: number;
  n_state: number;
  dt: number;
}

export interface PlasticityConfig {
  lr: number;
  weight_decay: number;
  momentum: number;
  nesterov: boolean;
  ns_steps: number;
  adjust_lr_fn: string | null;
  chunk: number;
  buffer_len: number;
  rollback_tol: number;
  grad_norm_max: number;
  state_norm_max: number;
}

// Phase 1: Session hierarchy with git-like lineage
export interface SessionMeta {
  schema_version: number;
  session_id: string;
  parent_session_id: string | null;  // Who I forked from
  root_session_id: string;           // The original ancestor
  created_at_unix: number;
  last_run_at_unix: number | null;   // When last run occurred
  torch_version?: string;
  env_mode: 'linear' | 'nonlinear';
  mu: number;
  base_ckpt_hash: string;
  model_signature: string;
  model_cfg: ModelConfig;
  plasticity_cfg: PlasticityConfig;
}

// Phase 1: Session summary for index
export interface SessionSummary {
  session_id: string;
  parent_session_id: string | null;
  root_session_id: string;
  created_at_unix: number;
  last_run_at_unix: number | null;
  env_mode: 'linear' | 'nonlinear';
  mu: number;
  model_signature: string;
  // Aggregated stats
  total_runs: number;
  total_updates_committed: number;
  total_updates_rolled_back: number;
}

// Phase 1: Session index for navigation
export interface SessionIndex {
  schema_version: number;
  sessions: Record<string, SessionSummary>;
}

// Phase 1: Three-way metrics (base, session_start, adaptive)
export interface SessionMetrics {
  run_id: string;
  seed: number;
  steps: number;
  mu: number;
  env_mode: 'linear' | 'nonlinear';
  // Base model (frozen pretrained weights)
  base_mse_mean: number;
  base_mse_last100_mean: number;
  // Session start (session weights, no updates this run)
  session_no_update_mse_mean: number;
  session_no_update_last100_mean: number;
  // Adaptive (session weights + online updates)
  adaptive_mse_mean: number;
  adaptive_last100_mean: number;
  // Update stats
  updates_attempted: number;
  updates_committed: number;
  updates_rolled_back: number;
  // Legacy aliases for backward compatibility
  baseline_mse_mean: number;
  baseline_mse_last100_mean: number;
  adaptive_mse_last100_mean: number;
}

// Phase 1: Run metadata
export interface RunMeta {
  run_id: string;
  session_id: string;
  created_at_unix: number;
  seed: number;
  steps: number;
  updates_committed: number;
  updates_rolled_back: number;
}

export type UpdateStatus =
  | 'commit'
  | 'rollback_loss_regression'
  | 'rollback_grad_norm'
  | 'rollback_state_norm';

export interface UpdateEvent {
  t: number;
  status: UpdateStatus;
  pre_loss: number;
  post_loss: number | null;
  grad_norm: number;
  pre_max_h: number;
  post_max_h?: number;
}

// Phase 1: Three-way per-step metrics
export interface PerStepMetric {
  t: number;
  base_mse: number;           // Base model (frozen pretrained)
  session_start_mse: number;  // Session weights, no updates this run
  adaptive_mse: number;       // Session weights + online updates
  did_update: boolean;
  update_ok: boolean;
  // Legacy alias for backward compatibility
  baseline_mse: number;
}

// Phase 1: Update event with run_id for cross-run tracking
export interface GlobalUpdateEvent extends UpdateEvent {
  run_id: string;
}

export interface PlasticWeights {
  W_u: number[][];  // [u_dim, z_dim + act_dim]
  B: number[][];    // [n_state, u_dim]
  W_o: number[][];  // [z_dim, n_state]
}

export interface TrajectoryPoint {
  t: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  ax: number;
  ay: number;
}

// Phase 1: Run data (per-run metrics and events)
export interface RunData {
  run_id: string;
  created_at_unix: number;
  seed: number;
  steps: number;
  metrics: SessionMetrics;
  perStep: PerStepMetric[];
  updateEvents: UpdateEvent[];
  trajectory?: TrajectoryPoint[];
}

export interface SessionData {
  meta: SessionMeta;
  metrics: SessionMetrics;           // Current/latest run metrics
  perStep: PerStepMetric[];          // Current/latest run data
  updateEvents: UpdateEvent[];       // Current/latest run events
  globalUpdateEvents: GlobalUpdateEvent[];  // All events across all runs
  runs: RunData[];                   // All runs for this session
  weights?: PlasticWeights;
  parentWeights?: PlasticWeights;    // Parent session weights for diff
  baseWeights?: PlasticWeights;      // Base checkpoint weights for diff
  trajectory?: TrajectoryPoint[];
}

// Phase 1: Session tree node for visualization
export interface SessionTreeNode {
  session: SessionSummary;
  children: SessionTreeNode[];
  depth: number;
}

export type TabId =
  | 'session-tree'  // Phase 1: New home tab for navigation
  | 'overview'
  | 'text-monitor'
  | 'chat'
  | 'train'
  | 'weights'
  | 'transactions'
  | 'environment'
  | 'architecture'
  | 'sessions';

export interface Tab {
  id: TabId;
  label: string;
  icon: string;
}

// Text chat update event (from ttt_ssm_nano/artifacts_api/text_chat_service.py)
export interface ChatUpdateEvent {
  chunk_index: number;
  token_start: number;
  token_end: number;
  step_in_chunk: number;
  loss: number;
  grad_norm: number;
  update_norm: number;
}

// Text monitoring types (from ttt/monitors/gradient.py)
export interface MonitorEvent {
  chunk_index: number;
  token_start: number;
  token_end: number;
  chunk_preview: string;
  loss: number;
  grad_norm: number;
  update_norm: number;
  grad_z: number | null;
  update_z: number | null;
  flagged: boolean;
  reasons: string[];
  top_influence_tokens: Array<[string, number]>;
  // Gate decision fields
  gate_allowed: boolean;
  gate_reasons: string[];
  token_entropy: number;
  token_diversity: number;
  update_skipped: boolean;
  // Rollback fields
  attempted_update_norm: number;
  rollback_triggered: boolean;
  rollback_reasons: string[];
  canary_loss_before: number | null;
  canary_loss_after: number | null;
  canary_delta: number | null;
  canary_delta_z: number | null;
  // Backbone/objective metadata
  backbone: string;
  objective: string;
  // Compression-based signal (Kolmogorov proxy)
  compression_ratio: number | null;
  // Canary gradient alignment signals
  canary_grad_norm: number | null;
  grad_canary_cos: number | null;  // Cosine similarity [-1, 1]
  grad_canary_dot: number | null;  // Dot product
}
