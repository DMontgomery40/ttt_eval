import { create } from 'zustand';
import type { SessionData, TabId, UpdateEvent, SessionIndex, SessionTreeNode, RunData } from '../types';
import { mockSession, mockSessions, mockSessionIndex, mockSessionTree } from '../data/mockData';
import { buildSessionTree, getSessionLineage } from '../utils/sessionTree';
import { fetchSessionIndex, fetchSessions, forkSession as forkSessionApi } from '../api/nanoApi';

interface DashboardState {
  // Current session
  currentSession: SessionData;
  sessions: SessionData[];

  // Phase 1: Session index and tree
  sessionIndex: SessionIndex;
  sessionTree: SessionTreeNode[];

  // Data source
  dataSource: 'mock' | 'api';
  isLoading: boolean;
  loadError: string | null;
  hasInitialized: boolean;
  initialize: () => Promise<void>;
  refreshFromApi: () => Promise<void>;

  // Navigation
  activeTab: TabId;
  setActiveTab: (tab: TabId) => void;

  // Time control
  currentTime: number;
  isPlaying: boolean;
  playbackSpeed: number;
  setCurrentTime: (t: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setPlaybackSpeed: (speed: number) => void;

  // Selection
  selectedUpdateEvent: UpdateEvent | null;
  setSelectedUpdateEvent: (event: UpdateEvent | null) => void;

  // Phase 1: Run selection
  selectedRunId: string | null;
  setSelectedRunId: (runId: string | null) => void;
  getCurrentRun: () => RunData | null;

  // View options
  logScale: boolean;
  smoothing: boolean;
  smoothingWindow: number;
  setLogScale: (log: boolean) => void;
  setSmoothing: (smooth: boolean) => void;
  setSmoothingWindow: (window: number) => void;

  // Phase 1: Weight comparison mode
  weightCompareMode: 'none' | 'parent' | 'base';
  setWeightCompareMode: (mode: 'none' | 'parent' | 'base') => void;

  // Session management
  selectedSessionIds: string[];
  toggleSessionSelection: (id: string) => void;
  setCurrentSession: (session: SessionData) => void;

  // Phase 1: Session lineage
  getSessionLineage: () => string[];

  // Phase 1: Fork session (mock or API)
  forkSession: (parentSessionId: string, newSessionId: string) => void;
}

function pickDefaultSession(sessions: SessionData[], index: SessionIndex): SessionData | null {
  if (sessions.length === 0) return null;

  let best: SessionData | null = null;
  let bestTs = -1;

  for (const s of sessions) {
    const summary = index.sessions[s.meta.session_id];
    const ts = (summary?.last_run_at_unix ?? summary?.created_at_unix ?? 0) as number;
    if (ts > bestTs) {
      best = s;
      bestTs = ts;
    }
  }

  return best || sessions[0];
}

function normalizeSessionForLatestRun(session: SessionData): SessionData {
  if (!session.runs || session.runs.length === 0) return session;
  const latestRun = [...session.runs].sort((a, b) => b.created_at_unix - a.created_at_unix)[0];
  if (!latestRun) return session;

  return {
    ...session,
    metrics: latestRun.metrics,
    perStep: latestRun.perStep,
    updateEvents: latestRun.updateEvents,
    trajectory: latestRun.trajectory ?? session.trajectory,
  };
}

export const useDashboardStore = create<DashboardState>((set, get) => ({
  // Initial state
  currentSession: mockSession,
  sessions: mockSessions,
  sessionIndex: mockSessionIndex,
  sessionTree: mockSessionTree,
  dataSource: 'mock',
  isLoading: false,
  loadError: null,
  hasInitialized: false,
  activeTab: 'session-tree',  // Phase 1: Start on session tree
  currentTime: 0,
  isPlaying: false,
  playbackSpeed: 1,
  selectedUpdateEvent: null,
  selectedRunId: null,
  logScale: false,
  smoothing: false,
  smoothingWindow: 10,
  weightCompareMode: 'none',
  selectedSessionIds: [],

  // Actions
  setActiveTab: (tab) => set({ activeTab: tab }),

  setCurrentTime: (t) => set({ currentTime: t }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),

  setSelectedUpdateEvent: (event) => set({ selectedUpdateEvent: event }),

  setSelectedRunId: (runId) => set((state) => {
    const chosen = runId
      ? state.currentSession.runs.find(r => r.run_id === runId)
      : [...state.currentSession.runs].sort((a, b) => b.created_at_unix - a.created_at_unix)[0];

    if (!chosen) return { selectedRunId: runId };

    return {
      selectedRunId: runId,
      currentSession: {
        ...state.currentSession,
        metrics: chosen.metrics,
        perStep: chosen.perStep,
        updateEvents: chosen.updateEvents,
        trajectory: chosen.trajectory ?? state.currentSession.trajectory,
      }
    };
  }),

  getCurrentRun: () => {
    const state = get();
    const runId = state.selectedRunId;
    if (!runId) return state.currentSession.runs[state.currentSession.runs.length - 1] || null;
    return state.currentSession.runs.find(r => r.run_id === runId) || null;
  },

  setLogScale: (log) => set({ logScale: log }),
  setSmoothing: (smooth) => set({ smoothing: smooth }),
  setSmoothingWindow: (window) => set({ smoothingWindow: window }),

  setWeightCompareMode: (mode) => set({ weightCompareMode: mode }),

  toggleSessionSelection: (id) =>
    set((state) => ({
      selectedSessionIds: state.selectedSessionIds.includes(id)
        ? state.selectedSessionIds.filter((i) => i !== id)
        : [...state.selectedSessionIds, id]
    })),

  setCurrentSession: (session) => set({
    currentSession: normalizeSessionForLatestRun(session),
    selectedRunId: null,  // Reset run selection when switching sessions
    selectedUpdateEvent: null
  }),

  getSessionLineage: () => {
    const state = get();
    return getSessionLineage(state.currentSession.meta.session_id, state.sessionIndex);
  },

  forkSession: (parentSessionId, newSessionId) => {
    const state = get();

    // Mock mode: local-only fork for UI prototyping
    if (state.dataSource === 'mock') {
      set((s) => {
        const parentSession = s.sessionIndex.sessions[parentSessionId];
        if (!parentSession) return s;

        const newSummary = {
          session_id: newSessionId,
          parent_session_id: parentSessionId,
          root_session_id: parentSession.root_session_id,
          created_at_unix: Math.floor(Date.now() / 1000),
          last_run_at_unix: null,
          env_mode: parentSession.env_mode,
          mu: parentSession.mu,
          model_signature: parentSession.model_signature,
          total_runs: 0,
          total_updates_committed: 0,
          total_updates_rolled_back: 0
        };

        const nextIndex: SessionIndex = {
          ...s.sessionIndex,
          sessions: {
            ...s.sessionIndex.sessions,
            [newSessionId]: newSummary
          }
        };

        return {
          sessionIndex: nextIndex,
          sessionTree: buildSessionTree(nextIndex)
        };
      });
      return;
    }

    // API mode: request a real fork and refresh index.
    void (async () => {
      try {
        set({ isLoading: true, loadError: null });
        const newSession = await forkSessionApi(parentSessionId, newSessionId);
        const index = await fetchSessionIndex();
        const tree = buildSessionTree(index);

        set((s) => ({
          dataSource: 'api',
          sessionIndex: index,
          sessionTree: tree,
          sessions: [
            ...s.sessions.filter(x => x.meta.session_id !== newSession.meta.session_id),
            newSession
          ].map(normalizeSessionForLatestRun),
          currentSession: normalizeSessionForLatestRun(newSession),
          selectedRunId: null,
          selectedUpdateEvent: null,
          isLoading: false,
          loadError: null,
        }));
      } catch (e: any) {
        set({ isLoading: false, loadError: e?.message || String(e) });
      }
    })();
  },

  initialize: async () => {
    const state = get();
    if (state.hasInitialized) return;
    set({ hasInitialized: true });

    try {
      set({ isLoading: true, loadError: null });
      const [index, sessions] = await Promise.all([fetchSessionIndex(), fetchSessions()]);
      if (!sessions.length) {
        set({
          isLoading: false,
          loadError: 'No sessions found in artifacts. Run phase1_branching_muon.py to create sessions.',
        });
        return;
      }
      const tree = buildSessionTree(index);
      const normalizedSessions = sessions.map(normalizeSessionForLatestRun);
      const defaultSession = pickDefaultSession(normalizedSessions, index) || normalizedSessions[0];

      set({
        dataSource: 'api',
        sessionIndex: index,
        sessionTree: tree,
        sessions: normalizedSessions,
        currentSession: defaultSession,
        selectedRunId: null,
        selectedUpdateEvent: null,
        isLoading: false,
        loadError: null,
      });
    } catch (e: any) {
      // API unavailable: keep mock data, but surface error for debugging.
      set({ isLoading: false, loadError: e?.message || String(e) });
    }
  },

  refreshFromApi: async () => {
    try {
      set({ isLoading: true, loadError: null });
      const [index, sessions] = await Promise.all([fetchSessionIndex(), fetchSessions()]);
      const tree = buildSessionTree(index);
      const normalizedSessions = sessions.map(normalizeSessionForLatestRun);

      const currentId = get().currentSession.meta.session_id;
      const nextCurrent = normalizedSessions.find(s => s.meta.session_id === currentId)
        || pickDefaultSession(normalizedSessions, index)
        || normalizedSessions[0];

      set({
        dataSource: 'api',
        sessionIndex: index,
        sessionTree: tree,
        sessions: normalizedSessions,
        currentSession: normalizeSessionForLatestRun(nextCurrent),
        selectedRunId: null,
        selectedUpdateEvent: null,
        isLoading: false,
        loadError: null,
      });
    } catch (e: any) {
      set({ isLoading: false, loadError: e?.message || String(e) });
    }
  },
}));
