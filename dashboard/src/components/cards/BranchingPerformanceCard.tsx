import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { SessionData } from '../../types';

interface BranchingPerformanceCardProps {
  sessions: SessionData[];
}

interface ForkPerformance {
  sessionId: string;
  parentId: string | null;
  rootId: string;
  depth: number;
  baseMse: number;
  adaptiveMse: number;
  improvement: number;
  improvementVsParent: number | null;
  totalRuns: number;
  commits: number;
  rollbacks: number;
  createdAt: number;
  lastRunAt: number | null;
}

export function BranchingPerformanceCard({ sessions }: BranchingPerformanceCardProps) {
  // Calculate performance metrics for each session
  const performances = useMemo((): ForkPerformance[] => {
    return sessions.map(session => {
      const baseMse = session.metrics.base_mse_last100_mean;
      const adaptiveMse = session.metrics.adaptive_last100_mean;
      const improvement = baseMse > 0
        ? ((baseMse - adaptiveMse) / baseMse) * 100
        : 0;

      // Calculate depth
      let depth = 0;
      let current = session;
      const sessionMap = new Map(sessions.map(s => [s.meta.session_id, s]));
      while (current.meta.parent_session_id) {
        depth++;
        const parent = sessionMap.get(current.meta.parent_session_id);
        if (!parent) break;
        current = parent;
      }

      // Calculate improvement vs parent
      let improvementVsParent: number | null = null;
      if (session.meta.parent_session_id) {
        const parent = sessionMap.get(session.meta.parent_session_id);
        if (parent) {
          const parentBaseMse = parent.metrics.base_mse_last100_mean;
          const parentAdaptiveMse = parent.metrics.adaptive_last100_mean;
          const parentImprovement = parentBaseMse > 0
            ? ((parentBaseMse - parentAdaptiveMse) / parentBaseMse) * 100
            : 0;
          improvementVsParent = improvement - parentImprovement;
        }
      }

      return {
        sessionId: session.meta.session_id,
        parentId: session.meta.parent_session_id,
        rootId: session.meta.root_session_id,
        depth,
        baseMse,
        adaptiveMse,
        improvement,
        improvementVsParent,
        totalRuns: session.runs?.length || 1,
        commits: session.metrics.updates_committed,
        rollbacks: session.metrics.updates_rolled_back,
        createdAt: session.meta.created_at_unix,
        lastRunAt: session.meta.last_run_at_unix,
      };
    });
  }, [sessions]);

  // Build parent-child relationship data for heatmap
  const forkData = useMemo(() => {
    const parents = new Set<string>();
    const children = new Map<string, ForkPerformance[]>();

    performances.forEach(perf => {
      if (perf.parentId) {
        parents.add(perf.parentId);
        if (!children.has(perf.parentId)) {
          children.set(perf.parentId, []);
        }
        children.get(perf.parentId)!.push(perf);
      }
    });

    return {
      parents: Array.from(parents).sort(),
      children,
    };
  }, [performances]);

  // Leaderboard: best-performing forks
  const leaderboard = useMemo(() => {
    return performances
      .filter(p => p.improvementVsParent !== null)
      .sort((a, b) => b.improvementVsParent! - a.improvementVsParent!)
      .slice(0, 10);
  }, [performances]);

  // Get color for performance delta
  const getPerformanceColor = (delta: number | null): string => {
    if (delta === null) return '#6e7681';
    if (delta > 5) return '#39d353'; // Excellent
    if (delta > 2) return '#58a6ff'; // Good
    if (delta > 0) return '#a371f7'; // Slight improvement
    if (delta > -2) return '#f0883e'; // Slight regression
    return '#da3633'; // Regression
  };

  // Get background color with opacity
  const getBackgroundColor = (delta: number | null): string => {
    if (delta === null) return 'bg-surface-100';
    if (delta > 5) return 'bg-accent-green/20';
    if (delta > 2) return 'bg-accent-blue/20';
    if (delta > 0) return 'bg-accent-purple/20';
    if (delta > -2) return 'bg-accent-orange/20';
    return 'bg-accent-red/20';
  };

  if (sessions.length === 0) {
    return (
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-2">Branching Performance</h3>
        <p className="text-xs text-text-muted">No sessions available</p>
      </div>
    );
  }

  if (forkData.parents.length === 0) {
    return (
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-2">Branching Performance</h3>
        <p className="text-xs text-text-muted">
          No forks detected. Use the "Fork" button in the Sessions tab to create branches.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-surface-50 border border-surface-200 rounded-lg p-4 space-y-6">
      <div>
        <h3 className="text-sm font-semibold text-text-primary">Branching Performance Analysis</h3>
        <p className="text-xs text-text-muted mt-1">
          Comparing performance of forked sessions vs their parents
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-surface-100 rounded-lg p-3">
          <div className="text-xs text-text-muted">Total Sessions</div>
          <div className="text-2xl font-bold font-mono text-text-primary mt-1">
            {sessions.length}
          </div>
        </div>
        <div className="bg-surface-100 rounded-lg p-3">
          <div className="text-xs text-text-muted">Parent Sessions</div>
          <div className="text-2xl font-bold font-mono text-accent-blue mt-1">
            {forkData.parents.length}
          </div>
        </div>
        <div className="bg-surface-100 rounded-lg p-3">
          <div className="text-xs text-text-muted">Total Forks</div>
          <div className="text-2xl font-bold font-mono text-accent-purple mt-1">
            {performances.filter(p => p.parentId !== null).length}
          </div>
        </div>
        <div className="bg-surface-100 rounded-lg p-3">
          <div className="text-xs text-text-muted">Max Depth</div>
          <div className="text-2xl font-bold font-mono text-accent-gold mt-1">
            {Math.max(...performances.map(p => p.depth))}
          </div>
        </div>
      </div>

      {/* Parent-Child Performance Heatmap */}
      <div>
        <h4 className="text-xs font-semibold text-text-primary mb-3">Fork Performance Heatmap</h4>
        <div className="space-y-3">
          {forkData.parents.map(parentId => {
            const childPerfs = forkData.children.get(parentId) || [];
            const parentPerf = performances.find(p => p.sessionId === parentId);

            return (
              <div key={parentId} className="bg-surface-100 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs text-text-muted">Parent:</span>
                  <span className="font-mono text-sm text-accent-blue">{parentId}</span>
                  {parentPerf && (
                    <span className="text-xs text-text-secondary">
                      ({parentPerf.improvement >= 0 ? '+' : ''}{parentPerf.improvement.toFixed(1)}% improvement)
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                  {childPerfs.map(child => {
                    const delta = child.improvementVsParent;
                    const bgColor = getBackgroundColor(delta);
                    const textColor = getPerformanceColor(delta);

                    return (
                      <motion.div
                        key={child.sessionId}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className={`${bgColor} border border-surface-300 rounded p-2`}
                      >
                        <div className="font-mono text-xs text-text-primary truncate mb-1">
                          {child.sessionId}
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-text-muted">Δ:</span>
                          <span
                            className="font-mono text-sm font-bold"
                            style={{ color: textColor }}
                          >
                            {delta !== null ? (
                              <>{delta >= 0 ? '+' : ''}{delta.toFixed(1)}pp</>
                            ) : (
                              'N/A'
                            )}
                          </span>
                        </div>
                        <div className="flex items-center justify-between mt-1">
                          <span className="text-xs text-text-muted">Total:</span>
                          <span className="text-xs font-mono text-accent-green">
                            {child.improvement >= 0 ? '+' : ''}{child.improvement.toFixed(1)}%
                          </span>
                        </div>
                        <div className="text-xs text-text-secondary mt-1">
                          {child.totalRuns} run{child.totalRuns !== 1 ? 's' : ''}
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Leaderboard */}
      {leaderboard.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-text-primary mb-2">
            Top Performing Forks
          </h4>
          <div className="bg-surface-100 rounded-lg overflow-hidden">
            <table className="w-full text-xs">
              <thead className="bg-surface-200 text-text-muted uppercase">
                <tr>
                  <th className="px-3 py-2 text-left">Rank</th>
                  <th className="px-3 py-2 text-left">Session ID</th>
                  <th className="px-3 py-2 text-left">Parent</th>
                  <th className="px-3 py-2 text-center">Δ vs Parent</th>
                  <th className="px-3 py-2 text-center">Total Improvement</th>
                  <th className="px-3 py-2 text-center">Depth</th>
                  <th className="px-3 py-2 text-center">Runs</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-200">
                {leaderboard.map((perf, idx) => {
                  const medal = idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : '';
                  const rankColor = idx === 0 ? 'text-accent-gold' : idx === 1 ? 'text-text-muted' : idx === 2 ? 'text-accent-orange' : 'text-text-secondary';

                  return (
                    <tr key={perf.sessionId} className="hover:bg-surface-50">
                      <td className={`px-3 py-2 font-semibold ${rankColor}`}>
                        {medal} #{idx + 1}
                      </td>
                      <td className="px-3 py-2">
                        <span className="font-mono text-text-primary">{perf.sessionId}</span>
                      </td>
                      <td className="px-3 py-2">
                        <span className="font-mono text-text-secondary text-xs">{perf.parentId}</span>
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span
                          className="font-mono font-bold"
                          style={{ color: getPerformanceColor(perf.improvementVsParent) }}
                        >
                          {perf.improvementVsParent !== null ? (
                            <>{perf.improvementVsParent >= 0 ? '+' : ''}{perf.improvementVsParent.toFixed(1)}pp</>
                          ) : (
                            'N/A'
                          )}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className="font-mono text-accent-green">
                          {perf.improvement >= 0 ? '+' : ''}{perf.improvement.toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className="text-accent-purple font-mono">{perf.depth}</span>
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className="text-accent-blue font-mono">{perf.totalRuns}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="bg-accent-blue/10 border border-accent-blue/30 rounded-lg p-3">
        <div className="text-xs font-semibold text-accent-blue mb-2">Performance Scale</div>
        <div className="grid grid-cols-5 gap-2 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-accent-green" />
            <span className="text-text-secondary">&gt;+5pp: Excellent</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-accent-blue" />
            <span className="text-text-secondary">+2 to +5pp: Good</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-accent-purple" />
            <span className="text-text-secondary">0 to +2pp: Slight+</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-accent-orange" />
            <span className="text-text-secondary">0 to -2pp: Slight−</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-accent-red" />
            <span className="text-text-secondary">&lt;-2pp: Regression</span>
          </div>
        </div>
        <p className="text-xs text-text-secondary mt-2">
          <strong>Δ vs Parent:</strong> Percentage point (pp) change in total improvement compared to parent session.
          Positive = fork performs better than parent.
        </p>
      </div>
    </div>
  );
}
