import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { RunData } from '../../types';
import { formatNumber } from '../../utils/formatting';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
} from 'recharts';

interface RunComparisonPanelProps {
  runs: RunData[];
  sessionId: string;
}

const RUN_COLORS = ['#58a6ff', '#39d353', '#a371f7', '#f0883e', '#da3633', '#d29922'];

export function RunComparisonPanel({ runs, sessionId }: RunComparisonPanelProps) {
  // Prepare MSE trajectory data
  const mseTrajectoryData = useMemo(() => {
    if (runs.length === 0) return [];

    const maxSteps = Math.max(...runs.map(r => r.perStep.length));
    return Array.from({ length: maxSteps }, (_, t) => {
      const point: Record<string, number> = { t };
      runs.forEach(run => {
        const step = run.perStep[t];
        if (step) {
          point[`base_${run.run_id}`] = step.base_mse;
          point[`session_${run.run_id}`] = step.session_start_mse;
          point[`adaptive_${run.run_id}`] = step.adaptive_mse;
        }
      });
      return point;
    });
  }, [runs]);

  // Prepare comparison metrics
  const comparisonMetrics = useMemo(() => {
    return runs.map(run => ({
      run_id: run.run_id,
      seed: run.seed,
      steps: run.steps,
      base_mse: run.metrics.base_mse_last100_mean,
      session_start_mse: run.metrics.session_no_update_last100_mean,
      adaptive_mse: run.metrics.adaptive_last100_mean,
      persistent_learning: run.metrics.base_mse_last100_mean > 0
        ? ((run.metrics.base_mse_last100_mean - run.metrics.session_no_update_last100_mean) / run.metrics.base_mse_last100_mean) * 100
        : 0,
      online_learning: run.metrics.session_no_update_last100_mean > 0
        ? ((run.metrics.session_no_update_last100_mean - run.metrics.adaptive_last100_mean) / run.metrics.session_no_update_last100_mean) * 100
        : 0,
      total_improvement: run.metrics.base_mse_last100_mean > 0
        ? ((run.metrics.base_mse_last100_mean - run.metrics.adaptive_last100_mean) / run.metrics.base_mse_last100_mean) * 100
        : 0,
      updates_committed: run.metrics.updates_committed,
      updates_attempted: run.metrics.updates_attempted,
      updates_rolled_back: run.metrics.updates_rolled_back,
      commit_rate: run.metrics.updates_attempted > 0
        ? (run.metrics.updates_committed / run.metrics.updates_attempted) * 100
        : 0,
    }));
  }, [runs]);

  // Prepare bar chart data for final metrics
  const finalMetricsBarData = useMemo(() => {
    return runs.map((run, idx) => ({
      run_id: run.run_id.split('_').pop() || run.run_id,
      base: run.metrics.base_mse_last100_mean,
      session_start: run.metrics.session_no_update_last100_mean,
      adaptive: run.metrics.adaptive_last100_mean,
      color: RUN_COLORS[idx % RUN_COLORS.length],
    }));
  }, [runs]);

  // Prepare improvement comparison data
  const improvementBarData = useMemo(() => {
    return comparisonMetrics.map((m, idx) => ({
      run_id: m.run_id.split('_').pop() || m.run_id,
      persistent: m.persistent_learning,
      online: m.online_learning,
      total: m.total_improvement,
      color: RUN_COLORS[idx % RUN_COLORS.length],
    }));
  }, [comparisonMetrics]);

  // Calculate deltas (compare each run to the first run)
  const deltas = useMemo(() => {
    if (comparisonMetrics.length < 2) return [];

    const baseline = comparisonMetrics[0];
    return comparisonMetrics.slice(1).map(run => ({
      run_id: run.run_id,
      mse_delta: run.adaptive_mse - baseline.adaptive_mse,
      mse_delta_pct: baseline.adaptive_mse > 0
        ? ((run.adaptive_mse - baseline.adaptive_mse) / baseline.adaptive_mse) * 100
        : 0,
      improvement_delta: run.total_improvement - baseline.total_improvement,
      commit_rate_delta: run.commit_rate - baseline.commit_rate,
      rollback_delta: run.updates_rolled_back - baseline.updates_rolled_back,
    }));
  }, [comparisonMetrics]);

  if (runs.length === 0) {
    return (
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <p className="text-sm text-text-muted">No runs selected for comparison</p>
      </div>
    );
  }

  if (runs.length === 1) {
    const run = runs[0];
    const metrics = comparisonMetrics[0];

    return (
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4 space-y-4">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">Single Run View</h3>
          <p className="text-xs text-text-muted mt-1">
            Run: <span className="font-mono text-accent-blue">{run.run_id}</span>
          </p>
        </div>

        <div className="grid grid-cols-5 gap-3">
          <div className="bg-surface-100 rounded-lg p-3">
            <div className="text-xs text-text-muted">Seed</div>
            <div className="text-xl font-bold font-mono text-text-primary mt-1">{run.seed}</div>
          </div>
          <div className="bg-surface-100 rounded-lg p-3">
            <div className="text-xs text-text-muted">Steps</div>
            <div className="text-xl font-bold font-mono text-accent-blue mt-1">{run.steps}</div>
          </div>
          <div className="bg-surface-100 rounded-lg p-3">
            <div className="text-xs text-text-muted">Commits</div>
            <div className="text-xl font-bold font-mono text-accent-green mt-1">
              {metrics.updates_committed}
            </div>
            <div className="text-xs text-text-secondary mt-1">
              {metrics.commit_rate.toFixed(1)}% rate
            </div>
          </div>
          <div className="bg-surface-100 rounded-lg p-3">
            <div className="text-xs text-text-muted">Rollbacks</div>
            <div className="text-xl font-bold font-mono text-accent-orange mt-1">
              {metrics.updates_rolled_back}
            </div>
          </div>
          <div className="bg-accent-purple/10 border border-accent-purple/30 rounded-lg p-3">
            <div className="text-xs text-text-muted">Total Improvement</div>
            <div className="text-xl font-bold font-mono text-accent-purple mt-1">
              {metrics.total_improvement >= 0 ? '+' : ''}{metrics.total_improvement.toFixed(1)}%
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-surface-50 border border-surface-200 rounded-lg p-4 space-y-6"
    >
      <div>
        <h3 className="text-sm font-semibold text-text-primary">Run Comparison</h3>
        <p className="text-xs text-text-muted mt-1">
          Comparing {runs.length} runs from session: <span className="font-mono text-accent-blue">{sessionId}</span>
        </p>
      </div>

      {/* Side-by-side metrics table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-surface-100 text-text-muted text-xs">
            <tr>
              <th className="px-3 py-2 text-left">Metric</th>
              {runs.map((run, idx) => (
                <th key={run.run_id} className="px-3 py-2 text-center">
                  <div className="flex items-center justify-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: RUN_COLORS[idx % RUN_COLORS.length] }}
                    />
                    <span className="font-mono">{run.run_id.split('_').pop()}</span>
                  </div>
                  <div className="text-xs text-text-secondary mt-1">seed: {run.seed}</div>
                </th>
              ))}
              {deltas.length > 0 && <th className="px-3 py-2 text-center">Δ vs Run 1</th>}
            </tr>
          </thead>
          <tbody className="divide-y divide-surface-200">
            {/* Steps */}
            <tr className="hover:bg-surface-100">
              <td className="px-3 py-2 text-text-muted">Steps</td>
              {comparisonMetrics.map(m => (
                <td key={m.run_id} className="px-3 py-2 text-center font-mono text-text-primary">
                  {m.steps}
                </td>
              ))}
              {deltas.length > 0 && <td className="px-3 py-2"></td>}
            </tr>

            {/* Base MSE */}
            <tr className="hover:bg-surface-100">
              <td className="px-3 py-2 text-text-muted">Base MSE</td>
              {comparisonMetrics.map(m => (
                <td key={m.run_id} className="px-3 py-2 text-center font-mono text-text-secondary">
                  {formatNumber(m.base_mse, 3)}
                </td>
              ))}
              {deltas.length > 0 && <td className="px-3 py-2"></td>}
            </tr>

            {/* Session Start MSE */}
            <tr className="hover:bg-surface-100">
              <td className="px-3 py-2 text-text-muted">Session Start MSE</td>
              {comparisonMetrics.map(m => (
                <td key={m.run_id} className="px-3 py-2 text-center font-mono text-accent-blue">
                  {formatNumber(m.session_start_mse, 3)}
                </td>
              ))}
              {deltas.length > 0 && <td className="px-3 py-2"></td>}
            </tr>

            {/* Adaptive MSE */}
            <tr className="hover:bg-surface-100">
              <td className="px-3 py-2 text-text-muted">Adaptive MSE</td>
              {comparisonMetrics.map(m => (
                <td key={m.run_id} className="px-3 py-2 text-center font-mono text-accent-green font-bold">
                  {formatNumber(m.adaptive_mse, 3)}
                </td>
              ))}
              {deltas.length > 0 && (
                <td className="px-3 py-2">
                  {deltas.map((d) => (
                    <div key={d.run_id} className="text-center mb-1">
                      <span className={`font-mono text-xs ${d.mse_delta < 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {d.mse_delta >= 0 ? '+' : ''}{formatNumber(d.mse_delta, 3)}
                        {' '}({d.mse_delta_pct >= 0 ? '+' : ''}{d.mse_delta_pct.toFixed(1)}%)
                      </span>
                    </div>
                  ))}
                </td>
              )}
            </tr>

            {/* Persistent Learning */}
            <tr className="hover:bg-surface-100">
              <td className="px-3 py-2 text-text-muted">Persistent Learning</td>
              {comparisonMetrics.map(m => (
                <td key={m.run_id} className="px-3 py-2 text-center font-mono text-accent-blue">
                  {m.persistent_learning >= 0 ? '+' : ''}{m.persistent_learning.toFixed(1)}%
                </td>
              ))}
              {deltas.length > 0 && <td className="px-3 py-2"></td>}
            </tr>

            {/* Online Learning */}
            <tr className="hover:bg-surface-100">
              <td className="px-3 py-2 text-text-muted">Online Learning</td>
              {comparisonMetrics.map(m => (
                <td key={m.run_id} className="px-3 py-2 text-center font-mono text-accent-green">
                  {m.online_learning >= 0 ? '+' : ''}{m.online_learning.toFixed(1)}%
                </td>
              ))}
              {deltas.length > 0 && <td className="px-3 py-2"></td>}
            </tr>

            {/* Total Improvement */}
            <tr className="hover:bg-surface-100 bg-accent-purple/5">
              <td className="px-3 py-2 text-text-primary font-semibold">Total Improvement</td>
              {comparisonMetrics.map(m => (
                <td key={m.run_id} className="px-3 py-2 text-center font-mono text-accent-purple font-bold">
                  {m.total_improvement >= 0 ? '+' : ''}{m.total_improvement.toFixed(1)}%
                </td>
              ))}
              {deltas.length > 0 && (
                <td className="px-3 py-2">
                  {deltas.map((d) => (
                    <div key={d.run_id} className="text-center mb-1">
                      <span className={`font-mono text-xs font-bold ${d.improvement_delta >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {d.improvement_delta >= 0 ? '+' : ''}{d.improvement_delta.toFixed(1)}pp
                      </span>
                    </div>
                  ))}
                </td>
              )}
            </tr>

            {/* Commits */}
            <tr className="hover:bg-surface-100">
              <td className="px-3 py-2 text-text-muted">Updates Committed</td>
              {comparisonMetrics.map(m => (
                <td key={m.run_id} className="px-3 py-2 text-center font-mono text-accent-green">
                  {m.updates_committed}
                </td>
              ))}
              {deltas.length > 0 && <td className="px-3 py-2"></td>}
            </tr>

            {/* Rollbacks */}
            <tr className="hover:bg-surface-100">
              <td className="px-3 py-2 text-text-muted">Updates Rolled Back</td>
              {comparisonMetrics.map(m => (
                <td key={m.run_id} className="px-3 py-2 text-center font-mono text-accent-orange">
                  {m.updates_rolled_back}
                </td>
              ))}
              {deltas.length > 0 && (
                <td className="px-3 py-2">
                  {deltas.map((d) => (
                    <div key={d.run_id} className="text-center mb-1">
                      <span className={`font-mono text-xs ${d.rollback_delta > 0 ? 'text-accent-red' : 'text-accent-green'}`}>
                        {d.rollback_delta >= 0 ? '+' : ''}{d.rollback_delta}
                      </span>
                    </div>
                  ))}
                </td>
              )}
            </tr>

            {/* Commit Rate */}
            <tr className="hover:bg-surface-100">
              <td className="px-3 py-2 text-text-muted">Commit Rate</td>
              {comparisonMetrics.map(m => (
                <td key={m.run_id} className="px-3 py-2 text-center font-mono text-text-primary">
                  {m.commit_rate.toFixed(1)}%
                </td>
              ))}
              {deltas.length > 0 && (
                <td className="px-3 py-2">
                  {deltas.map((d) => (
                    <div key={d.run_id} className="text-center mb-1">
                      <span className={`font-mono text-xs ${d.commit_rate_delta >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {d.commit_rate_delta >= 0 ? '+' : ''}{d.commit_rate_delta.toFixed(1)}pp
                      </span>
                    </div>
                  ))}
                </td>
              )}
            </tr>
          </tbody>
        </table>
      </div>

      {/* MSE Trajectory Comparison */}
      <div>
        <h4 className="text-xs font-semibold text-text-primary mb-2">MSE Trajectory Comparison</h4>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={mseTrajectoryData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis
              dataKey="t"
              stroke="#8b949e"
              tick={{ fill: '#8b949e', fontSize: 11 }}
              label={{ value: 'Step', position: 'insideBottom', offset: -10, style: { fill: '#8b949e', fontSize: 11 } }}
            />
            <YAxis
              stroke="#8b949e"
              tick={{ fill: '#8b949e', fontSize: 11 }}
              label={{ value: 'MSE', angle: -90, position: 'insideLeft', style: { fill: '#8b949e', fontSize: 11 } }}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '6px', fontSize: '11px' }}
            />
            <Legend wrapperStyle={{ fontSize: '11px' }} />
            {runs.map((run, idx) => (
              <Line
                key={`adaptive_${run.run_id}`}
                type="monotone"
                dataKey={`adaptive_${run.run_id}`}
                stroke={RUN_COLORS[idx % RUN_COLORS.length]}
                strokeWidth={2}
                dot={false}
                name={`Run ${run.run_id.split('_').pop()} (seed ${run.seed})`}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Final Metrics Bar Chart */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <h4 className="text-xs font-semibold text-text-primary mb-2">Final MSE Comparison</h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={finalMetricsBarData} margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis
                dataKey="run_id"
                stroke="#8b949e"
                tick={{ fill: '#8b949e', fontSize: 10 }}
                label={{ value: 'Run', position: 'insideBottom', offset: -10, style: { fill: '#8b949e', fontSize: 10 } }}
              />
              <YAxis
                stroke="#8b949e"
                tick={{ fill: '#8b949e', fontSize: 10 }}
                label={{ value: 'MSE', angle: -90, position: 'insideLeft', style: { fill: '#8b949e', fontSize: 10 } }}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '6px', fontSize: '11px' }}
              />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Bar dataKey="base" fill="#6e7681" name="Base" />
              <Bar dataKey="session_start" fill="#58a6ff" name="Session Start" />
              <Bar dataKey="adaptive" fill="#39d353" name="Adaptive" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div>
          <h4 className="text-xs font-semibold text-text-primary mb-2">Learning Breakdown</h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={improvementBarData} margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis
                dataKey="run_id"
                stroke="#8b949e"
                tick={{ fill: '#8b949e', fontSize: 10 }}
                label={{ value: 'Run', position: 'insideBottom', offset: -10, style: { fill: '#8b949e', fontSize: 10 } }}
              />
              <YAxis
                stroke="#8b949e"
                tick={{ fill: '#8b949e', fontSize: 10 }}
                label={{ value: 'Improvement %', angle: -90, position: 'insideLeft', style: { fill: '#8b949e', fontSize: 10 } }}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '6px', fontSize: '11px' }}
              />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Bar dataKey="persistent" fill="#58a6ff" name="Persistent" />
              <Bar dataKey="online" fill="#39d353" name="Online" />
              <Bar dataKey="total" fill="#a371f7" name="Total" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Summary */}
      {deltas.length > 0 && (
        <div className="bg-accent-blue/10 border border-accent-blue/30 rounded-lg p-3">
          <div className="text-xs font-semibold text-accent-blue mb-2">Comparison Summary</div>
          <div className="text-xs text-text-secondary space-y-1">
            {deltas.map((d, idx) => {
              const betterOrWorse = d.improvement_delta >= 0 ? 'better' : 'worse';
              const color = d.improvement_delta >= 0 ? 'text-accent-green' : 'text-accent-red';
              return (
                <div key={d.run_id}>
                  Run {idx + 2} is <span className={`font-mono ${color} font-semibold`}>
                    {Math.abs(d.improvement_delta).toFixed(1)}pp {betterOrWorse}
                  </span> than Run 1
                  {d.rollback_delta !== 0 && (
                    <span className="ml-2">
                      ({d.rollback_delta > 0 ? '+' : ''}{d.rollback_delta} rollbacks)
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </motion.div>
  );
}
