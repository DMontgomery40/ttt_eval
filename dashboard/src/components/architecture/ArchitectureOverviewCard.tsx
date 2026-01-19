import { Arrow, DiagramBox } from './FlowDiagram';

export type ArchitectureRepoStats = {
  textModelsTotal?: number;
  textModelsSleepCandidates?: number;
  textSessionsTotal?: number;
  textRunsTotal?: number;
  latestTextModelId?: string;
  latestTextModelStatus?: string;
  apiError?: string | null;
};

function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 px-3 py-2 rounded bg-surface-100 border border-surface-200">
      <span className="text-xs text-text-muted">{label}</span>
      <span className="text-xs font-mono text-text-primary">{value}</span>
    </div>
  );
}

export function ArchitectureOverviewCard({
  stats,
}: {
  stats: ArchitectureRepoStats;
}) {
  return (
    <div className="bg-surface-50 border border-surface-200 rounded-lg p-6 space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-lg font-semibold text-text-primary">System Architecture (repo-wide)</h3>
          <p className="text-sm text-text-secondary mt-2 max-w-3xl">
            This repo is intentionally a single cohesive surface for a TTT×SSM hybrid: a slow/core recurrent model plus a
            fast plastic module treated as a weight-based context window, plus persistence, branching (Nano), a shared
            safety harness for fast writes (gate / rollback / SPFW), and a first-pass sleep consolidation path (Text).
          </p>
        </div>

        <div className="grid grid-cols-2 gap-2 min-w-[360px]">
          <StatPill label="text_models" value={String(stats.textModelsTotal ?? '—')} />
          <StatPill label="sleep_candidates" value={String(stats.textModelsSleepCandidates ?? '—')} />
          <StatPill label="text_sessions" value={String(stats.textSessionsTotal ?? '—')} />
          <StatPill label="text_runs" value={String(stats.textRunsTotal ?? '—')} />
        </div>
      </div>

      {stats.latestTextModelId ? (
        <div className="text-xs text-text-muted">
          Latest text model:{' '}
          <span className="font-mono text-text-primary">{stats.latestTextModelId}</span>
          {stats.latestTextModelStatus ? (
            <>
              {' '}
              <span className="text-text-muted">status</span>{' '}
              <span className="font-mono text-text-primary">{stats.latestTextModelStatus}</span>
            </>
          ) : null}
        </div>
      ) : null}

      {stats.apiError ? (
        <div className="text-xs text-accent-orange">
          API stats unavailable: <span className="font-mono">{stats.apiError}</span>
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-4">
        {/* Text loop */}
        <div className="bg-surface-100 border border-surface-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-text-primary">Text Loop (core + fast weights)</h4>
            <span className="text-xs text-text-muted">no attention / no transformers</span>
          </div>

          <div className="mt-3 space-y-3">
            <div className="text-xs text-text-muted">Offline (core model):</div>
            <div className="flex items-center gap-2 overflow-x-auto py-2">
              <DiagramBox label="training_data/" color="#8b949e" />
              <Arrow />
              <DiagramBox label="train" sublabel="Muon" color="#58a6ff" highlight />
              <Arrow />
              <DiagramBox label="text_models/" sublabel="checkpoint.pt" color="#39d353" />
            </div>

            <div className="text-xs text-text-muted">Online (fast context net):</div>
            <div className="flex items-center gap-2 overflow-x-auto py-2">
              <DiagramBox label="prompt" color="#8b949e" />
              <Arrow />
              <DiagramBox label="core SSM" sublabel="frozen" color="#a371f7" />
              <Arrow />
              <DiagramBox label="context net" sublabel="TTT" color="#39d353" plastic />
              <Arrow />
              <DiagramBox label="sample" sublabel="completion" color="#58a6ff" />
              <Arrow />
              <DiagramBox label="text_sessions/" sublabel="state + trace" color="#8b949e" />
            </div>

            <div className="text-xs text-text-muted">Sleep (prototype):</div>
            <div className="flex items-center gap-2 overflow-x-auto py-2">
              <DiagramBox label="trace.jsonl" color="#8b949e" />
              <Arrow />
              <DiagramBox label="sleep" sublabel="Muon" color="#39d353" highlight />
              <Arrow />
              <DiagramBox label="candidate" sublabel="text_models/" color="#58a6ff" />
            </div>
          </div>
        </div>

        {/* Nano loop */}
        <div className="bg-surface-100 border border-surface-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-text-primary">Nano Loop (branching plastic weights)</h4>
            <span className="text-xs text-text-muted">Phase 1</span>
          </div>

          <div className="mt-3 space-y-3">
            <div className="text-xs text-text-muted">Artifacts and branching:</div>
            <div className="flex items-center gap-2 overflow-x-auto py-2">
              <DiagramBox label="base" sublabel="checkpoint" color="#8b949e" />
              <Arrow />
              <DiagramBox label="session" sublabel="plastic_state" color="#39d353" plastic />
              <Arrow />
              <DiagramBox label="run" sublabel="update_events" color="#58a6ff" />
              <Arrow />
              <DiagramBox label="fork" sublabel="branch" color="#a371f7" />
            </div>

            <div className="text-xs text-text-muted">
              Phase 1 is constrained by design: only a few 2D matrices are plastic, and the stability parameter is
              frozen to keep the recurrence stable while updates happen.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
