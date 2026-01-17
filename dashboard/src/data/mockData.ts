import type {
  SessionData,
  PerStepMetric,
  UpdateEvent,
  SessionMeta,
  SessionMetrics,
  PlasticWeights,
  TrajectoryPoint,
  UpdateStatus,
  RunData,
  GlobalUpdateEvent,
  SessionIndex,
  SessionSummary,
} from '../types';
import { buildSessionTree } from '../utils/sessionTree';

// Generate realistic mock data for development

function generatePerStepMetrics(steps: number, chunk: number, sessionImprovement: number = 0): PerStepMetric[] {
  const metrics: PerStepMetric[] = [];
  let baseBase = 0.08;
  let sessionStartBase = 0.08 * (1 - sessionImprovement);  // Session start is better than base
  let adaptiveBase = sessionStartBase;

  for (let t = 0; t < steps; t++) {
    // Base stays constant with small noise (pretrained frozen weights)
    const baseNoise = (Math.random() - 0.5) * 0.02;
    const base_mse = Math.max(0.001, baseBase + baseNoise + Math.sin(t * 0.02) * 0.01);

    // Session start (no updates this run) stays relatively constant
    const sessionNoise = (Math.random() - 0.5) * 0.015;
    const session_start_mse = Math.max(0.0008, sessionStartBase + sessionNoise + Math.sin(t * 0.02) * 0.008);

    // Adaptive improves over time as it learns (online updates)
    const learningFactor = Math.exp(-t / 200);
    const adaptiveNoise = (Math.random() - 0.5) * 0.01;
    const adaptive_mse = Math.max(0.0005, adaptiveBase * learningFactor * 0.3 + adaptiveNoise + 0.002);

    const did_update = (t + 1) % chunk === 0 && t >= chunk - 1;
    const update_ok = did_update && Math.random() > 0.15; // 85% success rate

    metrics.push({
      t,
      base_mse,
      session_start_mse,
      adaptive_mse,
      baseline_mse: base_mse,  // Legacy alias
      did_update,
      update_ok
    });
  }

  return metrics;
}

function generateUpdateEvents(perStep: PerStepMetric[], _chunk: number): UpdateEvent[] {
  const events: UpdateEvent[] = [];

  for (const step of perStep) {
    if (step.did_update) {
      const status: UpdateStatus = step.update_ok
        ? 'commit'
        : (Math.random() > 0.5 ? 'rollback_loss_regression' : 'rollback_grad_norm');

      const pre_loss = step.base_mse * (0.8 + Math.random() * 0.4);
      const post_loss = step.update_ok
        ? pre_loss * (0.6 + Math.random() * 0.3)
        : pre_loss * (1.3 + Math.random() * 0.3);

      events.push({
        t: step.t,
        status,
        pre_loss,
        post_loss: status === 'rollback_grad_norm' ? null : post_loss,
        grad_norm: status === 'rollback_grad_norm' ? 25 + Math.random() * 10 : 1 + Math.random() * 15,
        pre_max_h: 0.5 + Math.random() * 2,
        post_max_h: status !== 'rollback_grad_norm' ? 0.6 + Math.random() * 2.5 : undefined
      });
    }
  }

  return events;
}

function generateGlobalUpdateEvents(runs: RunData[]): GlobalUpdateEvent[] {
  const events: GlobalUpdateEvent[] = [];
  for (const run of runs) {
    for (const event of run.updateEvents) {
      events.push({
        ...event,
        run_id: run.run_id
      });
    }
  }
  return events;
}

function generateWeights(): PlasticWeights {
  const generateMatrix = (rows: number, cols: number): number[][] => {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() - 0.5) * 0.1)
    );
  };

  return {
    W_u: generateMatrix(16, 6),
    B: generateMatrix(32, 16),
    W_o: generateMatrix(4, 32)
  };
}

function generateTrajectory(steps: number, mu: number): TrajectoryPoint[] {
  const trajectory: TrajectoryPoint[] = [];
  let x = 0, y = 0, vx = 0, vy = 0;

  for (let t = 0; t < steps; t++) {
    // Random accelerations
    const ax = (Math.random() - 0.5) * 1.0;
    const ay = (Math.random() - 0.5) * 1.0;

    trajectory.push({ t, x, y, vx, vy, ax, ay });

    // Physics update with friction
    vx = (1 - mu) * vx + ax;
    vy = (1 - mu) * vy + ay;
    x += vx;
    y += vy;
  }

  return trajectory;
}

function generateRun(options: {
  sessionId: string;
  mu: number;
  envMode: 'linear' | 'nonlinear';
  steps?: number;
  seed?: number;
  daysAgo?: number;
  sessionImprovement?: number;
}): RunData {
  const steps = options.steps ?? 600;
  const seed = options.seed ?? Math.floor(Math.random() * 10000);
  const chunk = 32;
  const daysAgo = options.daysAgo ?? 0;
  const sessionImprovement = options.sessionImprovement ?? 0;

  const perStep = generatePerStepMetrics(steps, chunk, sessionImprovement);
  const updateEvents = generateUpdateEvents(perStep, chunk);
  const trajectory = generateTrajectory(steps, options.mu);

  const baseLast100 = perStep.slice(-100);
  const run_id = `run_${Date.now() - daysAgo * 86400000}`;

  const metrics: SessionMetrics = {
    run_id,
    seed,
    steps,
    mu: options.mu,
    env_mode: options.envMode,
    // Three-way metrics
    base_mse_mean: perStep.reduce((sum, s) => sum + s.base_mse, 0) / perStep.length,
    base_mse_last100_mean: baseLast100.reduce((sum, s) => sum + s.base_mse, 0) / 100,
    session_no_update_mse_mean: perStep.reduce((sum, s) => sum + s.session_start_mse, 0) / perStep.length,
    session_no_update_last100_mean: baseLast100.reduce((sum, s) => sum + s.session_start_mse, 0) / 100,
    adaptive_mse_mean: perStep.reduce((sum, s) => sum + s.adaptive_mse, 0) / perStep.length,
    adaptive_last100_mean: baseLast100.reduce((sum, s) => sum + s.adaptive_mse, 0) / 100,
    updates_attempted: updateEvents.length,
    updates_committed: updateEvents.filter(e => e.status === 'commit').length,
    updates_rolled_back: updateEvents.filter(e => e.status !== 'commit').length,
    // Legacy aliases
    baseline_mse_mean: perStep.reduce((sum, s) => sum + s.base_mse, 0) / perStep.length,
    baseline_mse_last100_mean: baseLast100.reduce((sum, s) => sum + s.base_mse, 0) / 100,
    adaptive_mse_last100_mean: baseLast100.reduce((sum, s) => sum + s.adaptive_mse, 0) / 100
  };

  return {
    run_id,
    created_at_unix: Math.floor(Date.now() / 1000) - daysAgo * 86400,
    seed,
    steps,
    metrics,
    perStep,
    updateEvents,
    trajectory
  };
}

export function generateMockSession(options?: {
  steps?: number;
  mu?: number;
  envMode?: 'linear' | 'nonlinear';
  sessionId?: string;
  parentSessionId?: string | null;
  rootSessionId?: string;
  numRuns?: number;
  sessionImprovement?: number;  // How much better than base (0-1)
}): SessionData {
  const steps = options?.steps ?? 600;
  const mu = options?.mu ?? 0.12 + Math.random() * 0.1;
  const envMode = options?.envMode ?? 'linear';
  const sessionId = options?.sessionId ?? 'session_001';
  const parentSessionId = options?.parentSessionId ?? null;
  const rootSessionId = options?.rootSessionId ?? sessionId;
  const numRuns = options?.numRuns ?? 3;
  const sessionImprovement = options?.sessionImprovement ?? 0.3;
  const chunk = 32;

  // Generate multiple runs for the session
  const runs: RunData[] = [];
  for (let i = 0; i < numRuns; i++) {
    runs.push(generateRun({
      sessionId,
      mu,
      envMode,
      steps,
      seed: 1337 + i,
      daysAgo: numRuns - 1 - i,  // Older runs first
      sessionImprovement: sessionImprovement * (i / (numRuns - 1 || 1))  // Progressive improvement
    }));
  }

  // Current run is the latest
  const currentRun = runs[runs.length - 1];
  const globalUpdateEvents = generateGlobalUpdateEvents(runs);
  const weights = generateWeights();
  const parentWeights = parentSessionId ? generateWeights() : undefined;
  const baseWeights = generateWeights();

  const meta: SessionMeta = {
    schema_version: 1,
    session_id: sessionId,
    parent_session_id: parentSessionId,
    root_session_id: rootSessionId,
    created_at_unix: Math.floor(Date.now() / 1000) - 86400 * numRuns,
    last_run_at_unix: currentRun.created_at_unix,
    torch_version: '2.1.0',
    env_mode: envMode,
    mu,
    base_ckpt_hash: 'a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef12345678',
    model_signature: 'f9e8d7c6b5a43210fedcba9876543210fedcba9876543210fedcba9876543210',
    model_cfg: {
      obs_dim: 4,
      act_dim: 2,
      z_dim: 4,
      u_dim: 16,
      n_state: 32,
      dt: 1.0
    },
    plasticity_cfg: {
      lr: 0.005,
      weight_decay: 0.0,
      momentum: 0.95,
      nesterov: true,
      ns_steps: 5,
      adjust_lr_fn: null,
      chunk,
      buffer_len: 32,
      rollback_tol: 0.20,
      grad_norm_max: 20.0,
      state_norm_max: 1e6
    }
  };

  return {
    meta,
    metrics: currentRun.metrics,
    perStep: currentRun.perStep,
    updateEvents: currentRun.updateEvents,
    globalUpdateEvents,
    runs,
    weights,
    parentWeights,
    baseWeights,
    trajectory: currentRun.trajectory
  };
}

// Phase 1: Generate mock session hierarchy
export function generateMockSessionIndex(): SessionIndex {
  const sessions: Record<string, SessionSummary> = {};

  // Root session
  sessions['session_001'] = {
    session_id: 'session_001',
    parent_session_id: null,
    root_session_id: 'session_001',
    created_at_unix: Math.floor(Date.now() / 1000) - 86400 * 7,
    last_run_at_unix: Math.floor(Date.now() / 1000) - 86400 * 2,
    env_mode: 'linear',
    mu: 0.12,
    model_signature: 'f9e8d7c6b5a43210',
    total_runs: 5,
    total_updates_committed: 75,
    total_updates_rolled_back: 12
  };

  // Fork A from session_001
  sessions['session_001_fork_a'] = {
    session_id: 'session_001_fork_a',
    parent_session_id: 'session_001',
    root_session_id: 'session_001',
    created_at_unix: Math.floor(Date.now() / 1000) - 86400 * 5,
    last_run_at_unix: Math.floor(Date.now() / 1000) - 86400 * 1,
    env_mode: 'linear',
    mu: 0.12,
    model_signature: 'f9e8d7c6b5a43210',
    total_runs: 3,
    total_updates_committed: 42,
    total_updates_rolled_back: 8
  };

  // Fork B from session_001 (different env mode)
  sessions['session_001_fork_b'] = {
    session_id: 'session_001_fork_b',
    parent_session_id: 'session_001',
    root_session_id: 'session_001',
    created_at_unix: Math.floor(Date.now() / 1000) - 86400 * 4,
    last_run_at_unix: Math.floor(Date.now() / 1000) - 3600,
    env_mode: 'nonlinear',
    mu: 0.12,
    model_signature: 'f9e8d7c6b5a43210',
    total_runs: 4,
    total_updates_committed: 58,
    total_updates_rolled_back: 10
  };

  // Sub-fork from fork_a
  sessions['session_001_fork_a_v2'] = {
    session_id: 'session_001_fork_a_v2',
    parent_session_id: 'session_001_fork_a',
    root_session_id: 'session_001',
    created_at_unix: Math.floor(Date.now() / 1000) - 86400 * 2,
    last_run_at_unix: Math.floor(Date.now() / 1000) - 1800,
    env_mode: 'linear',
    mu: 0.12,
    model_signature: 'f9e8d7c6b5a43210',
    total_runs: 2,
    total_updates_committed: 28,
    total_updates_rolled_back: 5
  };

  // Independent session with different mu
  sessions['session_002'] = {
    session_id: 'session_002',
    parent_session_id: null,
    root_session_id: 'session_002',
    created_at_unix: Math.floor(Date.now() / 1000) - 86400 * 3,
    last_run_at_unix: Math.floor(Date.now() / 1000) - 86400,
    env_mode: 'linear',
    mu: 0.18,
    model_signature: 'f9e8d7c6b5a43210',
    total_runs: 2,
    total_updates_committed: 32,
    total_updates_rolled_back: 6
  };

  // Fork from session_002
  sessions['session_002_high_mu'] = {
    session_id: 'session_002_high_mu',
    parent_session_id: 'session_002',
    root_session_id: 'session_002',
    created_at_unix: Math.floor(Date.now() / 1000) - 86400 * 1,
    last_run_at_unix: Math.floor(Date.now() / 1000) - 7200,
    env_mode: 'nonlinear',
    mu: 0.22,
    model_signature: 'f9e8d7c6b5a43210',
    total_runs: 1,
    total_updates_committed: 15,
    total_updates_rolled_back: 3
  };

  return {
    schema_version: 1,
    sessions
  };
}

// Pre-generated mock data for immediate use
export const mockSessionIndex = generateMockSessionIndex();
export const mockSessionTree = buildSessionTree(mockSessionIndex);

export const mockSession = generateMockSession({
  sessionId: 'session_001_fork_a_v2',
  parentSessionId: 'session_001_fork_a',
  rootSessionId: 'session_001',
  mu: 0.12,
  envMode: 'linear',
  numRuns: 3,
  sessionImprovement: 0.4
});

// Multiple sessions for comparison
export const mockSessions: SessionData[] = [
  generateMockSession({
    sessionId: 'session_001',
    parentSessionId: null,
    rootSessionId: 'session_001',
    mu: 0.12,
    envMode: 'linear',
    numRuns: 5,
    sessionImprovement: 0.2
  }),
  generateMockSession({
    sessionId: 'session_001_fork_a',
    parentSessionId: 'session_001',
    rootSessionId: 'session_001',
    mu: 0.12,
    envMode: 'linear',
    numRuns: 3,
    sessionImprovement: 0.35
  }),
  generateMockSession({
    sessionId: 'session_001_fork_b',
    parentSessionId: 'session_001',
    rootSessionId: 'session_001',
    mu: 0.12,
    envMode: 'nonlinear',
    numRuns: 4,
    sessionImprovement: 0.3
  }),
  generateMockSession({
    sessionId: 'session_001_fork_a_v2',
    parentSessionId: 'session_001_fork_a',
    rootSessionId: 'session_001',
    mu: 0.12,
    envMode: 'linear',
    numRuns: 2,
    sessionImprovement: 0.45
  }),
  generateMockSession({
    sessionId: 'session_002',
    parentSessionId: null,
    rootSessionId: 'session_002',
    mu: 0.18,
    envMode: 'linear',
    numRuns: 2,
    sessionImprovement: 0.25
  }),
  generateMockSession({
    sessionId: 'session_002_high_mu',
    parentSessionId: 'session_002',
    rootSessionId: 'session_002',
    mu: 0.22,
    envMode: 'nonlinear',
    numRuns: 1,
    sessionImprovement: 0.15
  }),
];
