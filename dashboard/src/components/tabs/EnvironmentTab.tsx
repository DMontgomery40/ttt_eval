import { useDashboardStore } from '../../store';
import { TrajectoryPlot } from '../charts/TrajectoryPlot';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

export function EnvironmentTab() {
  const { currentSession } = useDashboardStore();
  const { meta, trajectory } = currentSession;

  if (!trajectory || trajectory.length === 0) {
    return (
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-6">
        <h3 className="text-sm font-semibold text-text-primary mb-2">System ID (μ)</h3>
        <p className="text-sm text-text-muted">
          No trajectory recorded for this session/run.
        </p>
        <p className="text-xs text-text-muted mt-3">
          Older artifacts only stored loss curves. Newer runs write{' '}
          <span className="font-mono">trajectory.json</span> per run (position/velocity/actions). Re-run the Nano
          session to generate trajectory logging.
        </p>
      </div>
    );
  }

  // State time series data
  const stateData = trajectory.map((p) => ({
    t: p.t,
    x: p.x,
    y: p.y,
    vx: p.vx,
    vy: p.vy,
  }));

  // μ visualization
  const muMin = 0.02;
  const muMax = 0.25;
  const muNormalized = ((meta.mu - muMin) / (muMax - muMin)) * 100;

  // Nonlinear friction thresholds (if applicable)
  const thresholdScale = 2.0;
  const staticMult = 2.0;
  const dynamicMult = 0.5;

  return (
    <div className="space-y-6">
      {/* Physics Parameter Display */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold text-text-primary mb-1">Hidden Environment Parameter (μ)</h3>
            <p className="text-sm text-text-muted">
              Ground-truth friction coefficient for this run (hidden from model inputs)
            </p>
          </div>
          <div className="text-right">
            <span
              className={`text-xs px-2 py-1 rounded ${
                meta.env_mode === 'linear'
                  ? 'bg-accent-green/20 text-accent-green'
                  : 'bg-accent-purple/20 text-accent-purple'
              }`}
            >
              {meta.env_mode === 'linear' ? 'Linear Friction' : 'Nonlinear Friction'}
            </span>
          </div>
        </div>

        {/* Large μ display */}
        <div className="flex items-center gap-8 mb-6">
          <div className="text-center">
            <span className="text-6xl font-bold font-mono text-accent-purple">μ</span>
            <span className="text-4xl font-bold font-mono text-text-primary ml-2">
              = {meta.mu.toFixed(4)}
            </span>
          </div>

          {/* μ slider visualization */}
          <div className="flex-1">
            <div className="relative h-8 bg-surface-100 rounded-full overflow-hidden">
              <div
                className="absolute left-0 top-0 h-full bg-gradient-to-r from-accent-green/30 to-accent-purple/30"
                style={{ width: `${muNormalized}%` }}
              />
              <div
                className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-accent-purple border-2 border-white shadow-lg"
                style={{ left: `calc(${muNormalized}% - 8px)` }}
              />
            </div>
            <div className="flex justify-between text-xs text-text-muted mt-1">
              <span>μ_min = {muMin}</span>
              <span>μ_max = {muMax}</span>
            </div>
          </div>
        </div>

        {/* Physics explanation */}
        <div className="bg-surface-100 rounded-lg p-4 text-sm">
          <p className="text-text-secondary mb-2">
            <strong className="text-text-primary">Environment model:</strong> 2D point mass with friction
          </p>
          <code className="block bg-surface rounded p-2 font-mono text-xs text-accent-blue">
            vel_new = (1 - μ) × vel_old + acceleration
          </code>

          {meta.env_mode === 'nonlinear' && (
            <div className="mt-3 p-3 border border-accent-purple/30 rounded bg-accent-purple/5">
              <p className="text-accent-purple font-medium mb-2">Nonlinear Mode Active</p>
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <span className="text-text-muted">Velocity threshold: </span>
                  <span className="font-mono">{(meta.mu * thresholdScale).toFixed(4)}</span>
                </div>
                <div>
                  <span className="text-text-muted">Static μ (low speed): </span>
                  <span className="font-mono">{Math.min(0.95, meta.mu * staticMult).toFixed(4)}</span>
                </div>
                <div>
                  <span className="text-text-muted">Dynamic μ (high speed): </span>
                  <span className="font-mono">{(meta.mu * dynamicMult).toFixed(4)}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Trajectory Visualization */}
      <TrajectoryPlot trajectory={trajectory} height={400} />

      {/* State Time Series */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-4">State Evolution</h3>

        <div className="grid grid-cols-2 gap-4">
          {/* Position */}
          <div>
            <h4 className="text-xs text-text-muted mb-2">Position (x, y)</h4>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={stateData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                <XAxis dataKey="t" stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 10 }} />
                <YAxis stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#161b22',
                    border: '1px solid #30363d',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line type="monotone" dataKey="x" stroke="#58a6ff" strokeWidth={1.5} dot={false} name="x" />
                <Line type="monotone" dataKey="y" stroke="#39d353" strokeWidth={1.5} dot={false} name="y" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Velocity */}
          <div>
            <h4 className="text-xs text-text-muted mb-2">Velocity (vx, vy)</h4>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={stateData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                <XAxis dataKey="t" stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 10 }} />
                <YAxis stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#161b22',
                    border: '1px solid #30363d',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line type="monotone" dataKey="vx" stroke="#f0883e" strokeWidth={1.5} dot={false} name="vx" />
                <Line type="monotone" dataKey="vy" stroke="#a371f7" strokeWidth={1.5} dot={false} name="vy" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Key insight box */}
      <div className="bg-accent-blue/10 border border-accent-blue/30 rounded-lg p-4">
        <p className="text-sm text-text-primary">
          <strong className="text-accent-blue">Key Insight:</strong> The model must infer the hidden μ value
          from observing how the dynamics behave, then encode that knowledge into its plastic weights.
          Different μ values produce different velocity decay patterns, and a well-trained TTT model
          will adapt its predictions accordingly.
        </p>
      </div>
    </div>
  );
}
