import { useEffect, useMemo, useState } from 'react';

import { useDashboardStore } from '../../store';
import { listTextRuns } from '../../api/textApi';
import { listTextSessions } from '../../api/textChatApi';
import { listTextModels } from '../../api/textLmApi';

import { ArchitectureOverviewCard, type ArchitectureRepoStats } from '../architecture/ArchitectureOverviewCard';
import { NanoArchitectureSection } from '../architecture/NanoArchitectureSection';
import { SafetyCoverageSection } from '../architecture/SafetyCoverageSection';
import { SleepArchitectureSection } from '../architecture/SleepArchitectureSection';
import { TextArchitectureSection } from '../architecture/TextArchitectureSection';

type ArchitectureView = 'all' | 'system' | 'nano' | 'text' | 'sleep' | 'safety';

function ViewButton({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-3 py-1.5 text-xs rounded border transition-colors ${
        active
          ? 'bg-surface border-surface-200 text-text-primary'
          : 'bg-surface-100 border-surface-200 text-text-secondary hover:text-text-primary hover:bg-surface-200'
      }`}
    >
      {label}
    </button>
  );
}

export function ArchitectureTab() {
  const { currentSession } = useDashboardStore();
  const { meta } = currentSession;

  const [view, setView] = useState<ArchitectureView>('all');
  const [stats, setStats] = useState<ArchitectureRepoStats>({});
  const [isRefreshing, setIsRefreshing] = useState(false);

  const refreshStats = async () => {
    setIsRefreshing(true);
    try {
      const [modelsRes, sessionsRes, runsRes] = await Promise.allSettled([
        listTextModels(),
        listTextSessions(500),
        listTextRuns(500),
      ]);

      const next: ArchitectureRepoStats = {};

      if (modelsRes.status === 'fulfilled') {
        const ms = modelsRes.value || [];
        next.textModelsTotal = ms.length;
        next.textModelsSleepCandidates = ms.filter((m) => {
          const s = String((m as any)?.status || '').toLowerCase().trim();
          return s === 'sleep_candidate' || s === 'candidate';
        }).length;

        const latest = ms[0];
        if (latest) {
          next.latestTextModelId = String((latest as any)?.model_id || '').trim() || undefined;
          next.latestTextModelStatus = String((latest as any)?.status || '').trim() || undefined;
        }
      }

      if (sessionsRes.status === 'fulfilled') {
        next.textSessionsTotal = (sessionsRes.value || []).length;
      }

      if (runsRes.status === 'fulfilled') {
        next.textRunsTotal = (runsRes.value || []).length;
      }

      const errors: string[] = [];
      if (modelsRes.status === 'rejected') errors.push(String(modelsRes.reason?.message || modelsRes.reason));
      if (sessionsRes.status === 'rejected') errors.push(String(sessionsRes.reason?.message || sessionsRes.reason));
      if (runsRes.status === 'rejected') errors.push(String(runsRes.reason?.message || runsRes.reason));
      next.apiError = errors.length ? errors[0] : null;

      setStats(next);
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    void refreshStats();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const showSystem = view === 'all' || view === 'system';
  const showNano = view === 'all' || view === 'nano';
  const showText = view === 'all' || view === 'text';
  const showSleep = view === 'all' || view === 'sleep';
  const showSafety = view === 'all' || view === 'safety';

  const nanoCfg = useMemo(() => {
    return {
      modelCfg: meta.model_cfg,
      plasticityCfg: meta.plasticity_cfg,
    };
  }, [meta.model_cfg, meta.plasticity_cfg]);

  return (
    <div className="space-y-6">
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4 flex items-center justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">Architecture</h3>
          <p className="text-xs text-text-muted mt-1">
            Repo-wide blueprint (Nano + Text) and what is actually wired today.
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap justify-end">
          <ViewButton active={view === 'all'} label="All" onClick={() => setView('all')} />
          <ViewButton active={view === 'system'} label="System" onClick={() => setView('system')} />
          <ViewButton active={view === 'nano'} label="Nano" onClick={() => setView('nano')} />
          <ViewButton active={view === 'text'} label="Text" onClick={() => setView('text')} />
          <ViewButton active={view === 'sleep'} label="Sleep" onClick={() => setView('sleep')} />
          <ViewButton active={view === 'safety'} label="Safety" onClick={() => setView('safety')} />
          <button
            type="button"
            onClick={() => void refreshStats()}
            className="px-3 py-1.5 text-xs rounded bg-surface-200 hover:bg-surface-300 transition-colors"
            disabled={isRefreshing}
          >
            {isRefreshing ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {showSystem ? <ArchitectureOverviewCard stats={stats} /> : null}
      {showNano ? <NanoArchitectureSection modelCfg={nanoCfg.modelCfg} plasticityCfg={nanoCfg.plasticityCfg} /> : null}
      {showText ? <TextArchitectureSection /> : null}
      {showSleep ? <SleepArchitectureSection /> : null}
      {showSafety ? <SafetyCoverageSection /> : null}
    </div>
  );
}

