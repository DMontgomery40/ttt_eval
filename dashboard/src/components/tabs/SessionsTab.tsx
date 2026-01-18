import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../../store';
import { truncateHash, formatImprovement } from '../../utils/formatting';
import { RunComparisonPanel } from '../cards/RunComparisonPanel';
import { RunData } from '../../types';
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

const SESSION_COLORS = ['#58a6ff', '#39d353', '#a371f7', '#f0883e', '#da3633'];

// Fork modal component
function ForkModal({
  parentSessionId,
  onClose,
  onConfirm
}: {
  parentSessionId: string;
  onClose: () => void;
  onConfirm: (newId: string) => void;
}) {
  const [newSessionId, setNewSessionId] = useState(`${parentSessionId}_fork_${Date.now() % 10000}`);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="bg-surface-50 border border-surface-200 rounded-lg p-6 w-96"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="text-lg font-semibold text-text-primary mb-4">Fork Session</h3>
        <p className="text-sm text-text-secondary mb-4">
          Create a new session branched from <span className="font-mono text-accent-blue">{parentSessionId}</span>
        </p>

        <label className="block text-sm text-text-muted mb-2">New Session ID</label>
        <input
          type="text"
          value={newSessionId}
          onChange={(e) => setNewSessionId(e.target.value)}
          className="w-full bg-surface-100 border border-surface-200 rounded px-3 py-2 font-mono text-sm mb-4"
        />

        <div className="flex gap-2 justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm bg-surface-200 hover:bg-surface-300 rounded transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              onConfirm(newSessionId);
              onClose();
            }}
            className="px-4 py-2 text-sm bg-accent-blue hover:bg-accent-blue/80 text-white rounded transition-colors"
          >
            Create Fork
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}

export function SessionsTab() {
  const {
    sessions,
    currentSession,
    sessionIndex,
    selectedSessionIds,
    toggleSessionSelection,
    setCurrentSession,
    setActiveTab,
    forkSession
  } = useDashboardStore();

  const [forkingSession, setForkingSession] = useState<string | null>(null);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [showRunComparison, setShowRunComparison] = useState(false);

  // Calculate lineage depth for each session
  const getLineageDepth = (sessionId: string): number => {
    let depth = 0;
    let current = sessionIndex.sessions[sessionId];
    while (current?.parent_session_id) {
      depth++;
      current = sessionIndex.sessions[current.parent_session_id];
    }
    return depth;
  };

  // Get total runs for a session
  const getTotalRuns = (sessionId: string): number => {
    return sessionIndex.sessions[sessionId]?.total_runs ?? 0;
  };

  // Toggle run selection
  const toggleRunSelection = (runId: string) => {
    setSelectedRunIds(prev =>
      prev.includes(runId)
        ? prev.filter(id => id !== runId)
        : [...prev, runId]
    );
  };

  // Get selected runs
  const selectedRuns = useMemo((): RunData[] => {
    if (selectedRunIds.length === 0) return [];

    // Find all runs across all sessions
    const allRuns: RunData[] = [];
    sessions.forEach(session => {
      if (session.runs) {
        allRuns.push(...session.runs);
      }
    });

    return allRuns.filter(run => selectedRunIds.includes(run.run_id));
  }, [sessions, selectedRunIds]);

  // Get session for run comparison
  const runComparisonSessionId = useMemo(() => {
    if (selectedRuns.length === 0) return '';
    // All selected runs should be from the same session
    const session = sessions.find(s =>
      s.runs?.some(r => selectedRunIds.includes(r.run_id))
    );
    return session?.meta.session_id || '';
  }, [sessions, selectedRuns, selectedRunIds]);

  // Comparison data
  const comparisonData = useMemo(() => {
    if (selectedSessionIds.length < 2) return null;

    const selectedSessions = sessions.filter((s) => selectedSessionIds.includes(s.meta.session_id));

    // Align by timestep for line chart
    const maxSteps = Math.max(...selectedSessions.map((s) => s.perStep.length));
    const lineData = Array.from({ length: maxSteps }, (_, t) => {
      const point: Record<string, number> = { t };
      selectedSessions.forEach((session) => {
        const step = session.perStep[t];
        if (step) {
          point[session.meta.session_id] = step.adaptive_mse;
        }
      });
      return point;
    });

    // Bar chart data for final metrics
    const barData = selectedSessions.map((s) => ({
      session_id: s.meta.session_id,
      mu: s.meta.mu,
      baseline: s.metrics.base_mse_last100_mean,
      adaptive: s.metrics.adaptive_last100_mean,
      improvement:
        ((s.metrics.base_mse_last100_mean - s.metrics.adaptive_last100_mean) /
          s.metrics.base_mse_last100_mean) *
        100,
    }));

    return { lineData, barData, sessions: selectedSessions };
  }, [sessions, selectedSessionIds]);

  // Handle fork confirmation
  const handleForkConfirm = (newSessionId: string) => {
    if (forkingSession) {
      forkSession(forkingSession, newSessionId);
    }
  };

  return (
    <div className="space-y-6">
      {/* Session List */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg">
        <div className="p-4 border-b border-surface-200 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-text-primary">Available Sessions</h3>
            <p className="text-xs text-text-muted mt-1">
              Select 2+ sessions to compare, or click to load. Fork to create branches.
            </p>
          </div>
          <button
            onClick={() => setActiveTab('session-tree')}
            className="text-xs text-accent-blue hover:underline"
          >
            View as Tree
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-surface-100 text-text-muted text-xs uppercase">
              <tr>
                <th className="px-4 py-3 text-left w-8">
                  <input
                    type="checkbox"
                    checked={selectedSessionIds.length === sessions.length}
                    onChange={() => {
                      if (selectedSessionIds.length === sessions.length) {
                        sessions.forEach((s) => toggleSessionSelection(s.meta.session_id));
                      } else {
                        sessions
                          .filter((s) => !selectedSessionIds.includes(s.meta.session_id))
                          .forEach((s) => toggleSessionSelection(s.meta.session_id));
                      }
                    }}
                    className="rounded"
                  />
                </th>
                <th className="px-4 py-3 text-left">Session ID</th>
                <th className="px-4 py-3 text-left">Parent</th>
                <th className="px-4 py-3 text-center">Depth</th>
                <th className="px-4 py-3 text-left">μ Value</th>
                <th className="px-4 py-3 text-left">Mode</th>
                <th className="px-4 py-3 text-center">Runs</th>
                <th className="px-4 py-3 text-right">Commits</th>
                <th className="px-4 py-3 text-right">Improvement</th>
                <th className="px-4 py-3 text-center">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-200">
              {sessions.map((session, idx) => {
                const isSelected = selectedSessionIds.includes(session.meta.session_id);
                const isCurrent = currentSession.meta.session_id === session.meta.session_id;
                const depth = getLineageDepth(session.meta.session_id);
                const totalRuns = getTotalRuns(session.meta.session_id);

                return (
                  <motion.tr
                    key={session.meta.session_id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: idx * 0.05 }}
                    className={`
                      transition-colors
                      ${isCurrent ? 'bg-accent-blue/10' : 'hover:bg-surface-100'}
                    `}
                  >
                    <td className="px-4 py-3">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleSessionSelection(session.meta.session_id)}
                        className="rounded"
                      />
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        {/* Depth indicator */}
                        {depth > 0 && (
                          <span className="text-text-muted">{'└'.repeat(Math.min(depth, 2))}</span>
                        )}
                        <span className="font-mono">{session.meta.session_id}</span>
                        {isCurrent && (
                          <span className="text-xs bg-accent-blue/20 text-accent-blue px-1.5 py-0.5 rounded">
                            current
                          </span>
                        )}
                        {session.meta.parent_session_id === null && (
                          <span className="text-xs bg-accent-gold/20 text-accent-gold px-1.5 py-0.5 rounded">
                            root
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-text-muted">
                      {session.meta.parent_session_id || 'base'}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className={`px-1.5 py-0.5 rounded text-xs ${
                        depth === 0 ? 'bg-accent-gold/20 text-accent-gold' : 'bg-surface-200 text-text-muted'
                      }`}>
                        {depth}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className="font-mono text-accent-purple">{session.meta.mu.toFixed(4)}</span>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`text-xs px-1.5 py-0.5 rounded ${
                          session.meta.env_mode === 'linear'
                            ? 'bg-accent-green/20 text-accent-green'
                            : 'bg-accent-purple/20 text-accent-purple'
                        }`}
                      >
                        {session.meta.env_mode}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center font-mono text-accent-blue">
                      {totalRuns || session.runs?.length || 1}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      <span className="text-accent-green">{session.metrics.updates_committed}</span>
                      <span className="text-text-muted"> / </span>
                      <span>{session.metrics.updates_attempted}</span>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className="font-mono text-accent-green">
                        {formatImprovement(
                          session.metrics.base_mse_last100_mean,
                          session.metrics.adaptive_last100_mean
                        )}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <div className="flex items-center justify-center gap-1">
                        <button
                          onClick={() => setCurrentSession(session)}
                          disabled={isCurrent}
                          className={`px-2 py-1 text-xs rounded transition-colors ${
                            isCurrent
                              ? 'bg-surface-200 text-text-muted cursor-not-allowed'
                              : 'bg-accent-blue/20 text-accent-blue hover:bg-accent-blue/30'
                          }`}
                        >
                          {isCurrent ? 'Loaded' : 'Load'}
                        </button>
                        <button
                          onClick={() => setForkingSession(session.meta.session_id)}
                          className="px-2 py-1 text-xs rounded bg-surface-200 hover:bg-surface-300 text-text-secondary transition-colors"
                        >
                          Fork
                        </button>
                      </div>
                    </td>
                  </motion.tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Run Comparison Section */}
      {currentSession.runs && currentSession.runs.length > 1 && (
        <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-text-primary">Compare Runs</h3>
              <p className="text-xs text-text-muted mt-1">
                This session has {currentSession.runs.length} runs. Select 2+ to compare.
              </p>
            </div>
            {selectedRunIds.length >= 2 && (
              <button
                onClick={() => setShowRunComparison(!showRunComparison)}
                className="px-3 py-1.5 text-xs bg-accent-blue text-white rounded hover:bg-accent-blue/80 transition-colors"
              >
                {showRunComparison ? 'Hide Comparison' : `Compare ${selectedRunIds.length} Runs`}
              </button>
            )}
          </div>

          {/* Run selection checkboxes */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
            {currentSession.runs.map(run => {
              const isSelected = selectedRunIds.includes(run.run_id);
              return (
                <label
                  key={run.run_id}
                  className={`flex items-center gap-2 p-2 rounded border transition-colors cursor-pointer ${
                    isSelected
                      ? 'bg-accent-blue/10 border-accent-blue'
                      : 'bg-surface-100 border-surface-200 hover:border-surface-300'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => toggleRunSelection(run.run_id)}
                    className="rounded"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="font-mono text-xs text-text-primary truncate">
                      {run.run_id.split('_').pop()}
                    </div>
                    <div className="text-xs text-text-muted">
                      seed: {run.seed}, {run.steps} steps
                    </div>
                    <div className="text-xs text-accent-green font-mono">
                      {formatImprovement(run.metrics.base_mse_last100_mean, run.metrics.adaptive_last100_mean)}
                    </div>
                  </div>
                </label>
              );
            })}
          </div>

          {/* Clear selection button */}
          {selectedRunIds.length > 0 && (
            <div className="mt-3 flex items-center justify-between">
              <span className="text-xs text-text-muted">
                {selectedRunIds.length} run{selectedRunIds.length !== 1 ? 's' : ''} selected
              </span>
              <button
                onClick={() => setSelectedRunIds([])}
                className="text-xs text-accent-red hover:underline"
              >
                Clear Selection
              </button>
            </div>
          )}
        </div>
      )}

      {/* Run Comparison Panel */}
      {showRunComparison && selectedRuns.length >= 2 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <RunComparisonPanel runs={selectedRuns} sessionId={runComparisonSessionId} />
        </motion.div>
      )}

      {/* Compare with Parent */}
      {currentSession.meta.parent_session_id && (
        <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-text-primary">Parent Comparison</h3>
            <span className="text-xs text-text-muted">
              Comparing with <span className="font-mono text-accent-blue">{currentSession.meta.parent_session_id}</span>
            </span>
          </div>

          <div className="grid grid-cols-4 gap-4">
            <div className="bg-surface-100 rounded-lg p-3">
              <div className="text-xs text-text-muted mb-1">Lineage Depth</div>
              <div className="font-mono text-lg font-bold text-accent-purple">
                {getLineageDepth(currentSession.meta.session_id)}
              </div>
            </div>
            <div className="bg-surface-100 rounded-lg p-3">
              <div className="text-xs text-text-muted mb-1">Root Session</div>
              <div className="font-mono text-sm text-text-primary">
                {currentSession.meta.root_session_id}
              </div>
            </div>
            <div className="bg-surface-100 rounded-lg p-3">
              <div className="text-xs text-text-muted mb-1">Parent Session</div>
              <div className="font-mono text-sm text-text-primary">
                {currentSession.meta.parent_session_id}
              </div>
            </div>
            <div className="bg-surface-100 rounded-lg p-3">
              <div className="text-xs text-text-muted mb-1">Total Runs (This Branch)</div>
              <div className="font-mono text-lg font-bold text-accent-blue">
                {currentSession.runs?.length || 1}
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-accent-blue/10 border border-accent-blue/30 rounded text-xs text-text-secondary">
            <strong className="text-accent-blue">Branch Info:</strong> This session was forked from{' '}
            <span className="font-mono">{currentSession.meta.parent_session_id}</span> and inherits its learned weights.
            Any updates in this session build upon the parent's learning.
          </div>
        </div>
      )}

      {/* Comparison Charts */}
      {comparisonData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-4">
              MSE Comparison ({selectedSessionIds.length} sessions)
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={comparisonData.lineData}>
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
                {comparisonData.sessions.map((s, i) => (
                  <Line
                    key={s.meta.session_id}
                    type="monotone"
                    dataKey={s.meta.session_id}
                    stroke={SESSION_COLORS[i % SESSION_COLORS.length]}
                    strokeWidth={1.5}
                    dot={false}
                    name={`${s.meta.session_id} (μ=${s.meta.mu.toFixed(3)})`}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Final Performance Comparison</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={comparisonData.barData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                <XAxis dataKey="session_id" stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 10 }} />
                <YAxis stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#161b22',
                    border: '1px solid #30363d',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Bar dataKey="baseline" fill="#6e7681" name="Base MSE" />
                <Bar dataKey="adaptive" fill="#39d353" name="Adaptive MSE" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      )}

      {/* Current Session Resume Info */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Session Resume Info</h3>

        <div className="bg-surface-100 rounded-lg p-4 font-mono text-sm">
          <p className="text-text-muted mb-2">Resume this session with:</p>
          <code className="block bg-surface rounded p-2 text-accent-blue break-all">
            python phase0_muon.py --resume_session_dir=./artifacts/sessions/{currentSession.meta.session_id}
          </code>
        </div>

        <div className="mt-4 bg-surface-100 rounded-lg p-4 font-mono text-sm">
          <p className="text-text-muted mb-2">Fork this session:</p>
          <code className="block bg-surface rounded p-2 text-accent-green break-all">
            python phase0_muon.py --fork_from=./artifacts/sessions/{currentSession.meta.session_id} --new_session_id=my_fork
          </code>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4">
          <div>
            <span className="text-xs text-text-muted block mb-1">Model Signature</span>
            <code className="text-xs font-mono text-text-secondary break-all">
              {truncateHash(currentSession.meta.model_signature, 16)}
            </code>
          </div>
          <div>
            <span className="text-xs text-text-muted block mb-1">Base Checkpoint Hash</span>
            <code className="text-xs font-mono text-text-secondary break-all">
              {truncateHash(currentSession.meta.base_ckpt_hash, 16)}
            </code>
          </div>
        </div>

        <div className="mt-4 p-3 bg-accent-blue/10 border border-accent-blue/30 rounded text-xs text-text-secondary">
          <strong className="text-accent-blue">Note:</strong> All sessions with matching model_signature
          are compatible for comparison. The signature ensures identical model architecture and base weights.
        </div>
      </div>

      {/* Fork modal */}
      {forkingSession && (
        <ForkModal
          parentSessionId={forkingSession}
          onClose={() => setForkingSession(null)}
          onConfirm={handleForkConfirm}
        />
      )}
    </div>
  );
}
