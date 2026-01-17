import { useState, useMemo, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../../store';
import { formatRelativeTime } from '../../utils/formatting';
import type { SessionTreeNode } from '../../types';

// Tree node colors based on lineage
const ROOT_COLORS = ['#58a6ff', '#39d353', '#a371f7', '#f0883e', '#da3633'];

function getNodeColor(_node: SessionTreeNode, rootIndex: number): string {
  return ROOT_COLORS[rootIndex % ROOT_COLORS.length];
}

interface TreeNodeProps {
  node: SessionTreeNode;
  rootIndex: number;
  selectedSessionId: string;
  onSelect: (sessionId: string) => void;
  onFork: (sessionId: string) => void;
  collapsed: Set<string>;
  toggleCollapse: (sessionId: string) => void;
  level: number;
}

function TreeNode({
  node,
  rootIndex,
  selectedSessionId,
  onSelect,
  onFork,
  collapsed,
  toggleCollapse,
  level
}: TreeNodeProps) {
  const isCollapsed = collapsed.has(node.session.session_id);
  const hasChildren = node.children.length > 0;
  const color = getNodeColor(node, rootIndex);
  const isSelected = selectedSessionId === node.session.session_id;

  return (
    <div className="tree-node">
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: level * 0.05 }}
        className="flex items-start"
      >
        {/* Connector lines */}
        <div className="flex items-center" style={{ width: level * 24 }}>
          {level > 0 && (
            <div className="flex items-center h-full">
              {Array.from({ length: level }).map((_, i) => (
                <div
                  key={i}
                  className="w-6 h-full flex items-center justify-center"
                >
                  {i === level - 1 && (
                    <div className="w-4 h-px bg-surface-300" />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Expand/collapse button */}
        <button
          onClick={() => hasChildren && toggleCollapse(node.session.session_id)}
          className={`w-5 h-5 flex items-center justify-center text-xs flex-shrink-0 ${
            hasChildren ? 'cursor-pointer hover:bg-surface-200 rounded' : 'cursor-default'
          }`}
        >
          {hasChildren && (isCollapsed ? '+' : '-')}
        </button>

        {/* Node content */}
        <motion.button
          onClick={() => onSelect(node.session.session_id)}
          className={`
            flex-1 flex items-center gap-3 p-3 rounded-lg border-2 transition-all
            ${isSelected
              ? 'border-accent-blue bg-accent-blue/10 ring-2 ring-accent-blue/30'
              : 'border-surface-200 hover:border-surface-300 bg-surface-50'
            }
          `}
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
        >
          {/* Node indicator */}
          <div
            className="w-3 h-3 rounded-full flex-shrink-0"
            style={{ backgroundColor: color }}
          />

          {/* Session info */}
          <div className="flex-1 text-left">
            <div className="flex items-center gap-2">
              <span className="font-mono text-sm font-medium text-text-primary">
                {node.session.session_id}
              </span>
              {node.session.parent_session_id === null && (
                <span className="text-xs px-1.5 py-0.5 bg-accent-gold/20 text-accent-gold rounded">
                  root
                </span>
              )}
            </div>
            <div className="flex items-center gap-3 mt-1 text-xs text-text-muted">
              <span className="font-mono text-accent-purple">
                μ={node.session.mu.toFixed(3)}
              </span>
              <span className={node.session.env_mode === 'linear' ? 'text-accent-green' : 'text-accent-purple'}>
                {node.session.env_mode}
              </span>
              <span>
                {node.session.total_updates_committed} commits
              </span>
              <span className="text-accent-red">
                {node.session.total_updates_rolled_back} rollbacks
              </span>
            </div>
          </div>

          {/* Stats badge */}
          <div className="text-right flex-shrink-0">
            <div className="text-xs text-text-muted">
              {node.session.total_runs} runs
            </div>
            <div className="text-xs text-text-muted">
              {node.session.last_run_at_unix
                ? formatRelativeTime(node.session.last_run_at_unix)
                : 'never run'
              }
            </div>
          </div>

          {/* Fork button */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onFork(node.session.session_id);
            }}
            className="px-2 py-1 text-xs bg-surface-200 hover:bg-surface-300 rounded transition-colors flex-shrink-0"
            title="Fork from this session"
          >
            Fork
          </button>
        </motion.button>
      </motion.div>

      {/* Children */}
      {hasChildren && !isCollapsed && (
        <div className="ml-5 mt-1 border-l border-surface-200 pl-0">
          {node.children.map((child) => (
            <TreeNode
              key={child.session.session_id}
              node={child}
              rootIndex={rootIndex}
              selectedSessionId={selectedSessionId}
              onSelect={onSelect}
              onFork={onFork}
              collapsed={collapsed}
              toggleCollapse={toggleCollapse}
              level={level + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// Base checkpoint visual node
function BaseCheckpointNode() {
  return (
    <div className="flex items-center gap-3 p-4 rounded-lg border-2 border-dashed border-accent-gold/50 bg-accent-gold/5 mb-4">
      <div className="w-4 h-4 rounded-full bg-gradient-to-br from-yellow-400 to-orange-500" />
      <div>
        <div className="font-mono text-sm font-bold text-accent-gold">
          base_checkpoint
        </div>
        <div className="text-xs text-text-muted">
          Original pretrained weights (frozen)
        </div>
      </div>
    </div>
  );
}

// Fork modal
function ForkModal({
  parentSessionId,
  onClose,
  onConfirm
}: {
  parentSessionId: string;
  onClose: () => void;
  onConfirm: (newId: string) => void;
}) {
  const [newSessionId, setNewSessionId] = useState(`${parentSessionId}_fork_${Date.now()}`);

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
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-surface-50 border border-surface-200 rounded-lg p-6 w-96"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="text-lg font-semibold text-text-primary mb-4">
          Fork Session
        </h3>
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

export function SessionTreeTab() {
  const {
    currentSession,
    sessions,
    sessionTree,
    sessionIndex,
    setCurrentSession,
    setActiveTab,
    forkSession
  } = useDashboardStore();

  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [forkingSession, setForkingSession] = useState<string | null>(null);

  const toggleCollapse = useCallback((sessionId: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(sessionId)) {
        next.delete(sessionId);
      } else {
        next.add(sessionId);
      }
      return next;
    });
  }, []);

  const handleSelect = useCallback((sessionId: string) => {
    const session = sessions.find(s => s.meta.session_id === sessionId);
    if (session) {
      setCurrentSession(session);
      setActiveTab('overview');
    }
  }, [sessions, setCurrentSession, setActiveTab]);

  const handleFork = useCallback((sessionId: string) => {
    setForkingSession(sessionId);
  }, []);

  const handleForkConfirm = useCallback((newSessionId: string) => {
    if (forkingSession) {
      forkSession(forkingSession, newSessionId);
    }
  }, [forkingSession, forkSession]);

  // Stats summary
  const stats = useMemo(() => {
    const allSessions = Object.values(sessionIndex.sessions);
    return {
      totalSessions: allSessions.length,
      rootSessions: allSessions.filter(s => s.parent_session_id === null).length,
      totalRuns: allSessions.reduce((sum, s) => sum + s.total_runs, 0),
      totalCommits: allSessions.reduce((sum, s) => sum + s.total_updates_committed, 0),
      totalRollbacks: allSessions.reduce((sum, s) => sum + s.total_updates_rolled_back, 0)
    };
  }, [sessionIndex]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-text-primary">Session Tree</h2>
          <p className="text-sm text-text-muted">
            Git-like branching for neural network weight evolution
          </p>
        </div>

        {/* Stats summary */}
        <div className="flex items-center gap-6 text-sm">
          <div>
            <span className="text-text-muted">Sessions: </span>
            <span className="font-mono font-bold text-text-primary">{stats.totalSessions}</span>
          </div>
          <div>
            <span className="text-text-muted">Roots: </span>
            <span className="font-mono font-bold text-accent-gold">{stats.rootSessions}</span>
          </div>
          <div>
            <span className="text-text-muted">Total Runs: </span>
            <span className="font-mono font-bold text-accent-blue">{stats.totalRuns}</span>
          </div>
          <div>
            <span className="text-text-muted">Commits: </span>
            <span className="font-mono font-bold text-accent-green">{stats.totalCommits}</span>
          </div>
          <div>
            <span className="text-text-muted">Rollbacks: </span>
            <span className="font-mono font-bold text-accent-red">{stats.totalRollbacks}</span>
          </div>
        </div>
      </div>

      {/* Tree visualization */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-6">
        <BaseCheckpointNode />

        {/* Session trees */}
        <div className="space-y-4">
          {sessionTree.map((root, idx) => (
            <div key={root.session.session_id} className="session-tree-root">
              <TreeNode
                node={root}
                rootIndex={idx}
                selectedSessionId={currentSession.meta.session_id}
                onSelect={handleSelect}
                onFork={handleFork}
                collapsed={collapsed}
                toggleCollapse={toggleCollapse}
                level={0}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Current session info */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-text-primary">Current Session</h3>
          <button
            onClick={() => setActiveTab('overview')}
            className="px-3 py-1 text-sm bg-accent-blue/20 text-accent-blue hover:bg-accent-blue/30 rounded transition-colors"
          >
            View Details
          </button>
        </div>

        <div className="grid grid-cols-4 gap-4">
          <div className="bg-surface-100 rounded p-3">
            <div className="text-xs text-text-muted mb-1">Session ID</div>
            <div className="font-mono text-sm text-text-primary">{currentSession.meta.session_id}</div>
          </div>
          <div className="bg-surface-100 rounded p-3">
            <div className="text-xs text-text-muted mb-1">Parent</div>
            <div className="font-mono text-sm text-text-primary">
              {currentSession.meta.parent_session_id || 'base_checkpoint'}
            </div>
          </div>
          <div className="bg-surface-100 rounded p-3">
            <div className="text-xs text-text-muted mb-1">Runs</div>
            <div className="font-mono text-sm text-accent-blue">{currentSession.runs.length}</div>
          </div>
          <div className="bg-surface-100 rounded p-3">
            <div className="text-xs text-text-muted mb-1">Updates</div>
            <div className="font-mono text-sm">
              <span className="text-accent-green">{currentSession.metrics.updates_committed}</span>
              <span className="text-text-muted"> / </span>
              <span className="text-accent-red">{currentSession.metrics.updates_rolled_back}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-3">Legend</h3>
        <div className="flex items-center gap-6 text-xs text-text-secondary">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-br from-yellow-400 to-orange-500" />
            <span>Base checkpoint (pretrained)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-accent-blue" />
            <span>Root session (forked from base)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-accent-green" />
            <span>Child session</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-1.5 py-0.5 bg-accent-gold/20 text-accent-gold rounded">root</span>
            <span>First-level fork from base</span>
          </div>
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
