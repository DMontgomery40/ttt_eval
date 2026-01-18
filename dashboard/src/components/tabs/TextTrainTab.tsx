import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
} from 'recharts';
import { formatNumber } from '../../utils/formatting';
import { listTextModels, type TextModelSummary } from '../../api/textLmApi';
import {
  cancelTraining,
  getTrainMetrics,
  getTrainStatus,
  listTrainJobs,
  startTraining,
  type TrainMetric,
  type TrainStatus,
} from '../../api/textTrainApi';

const DEFAULT_CORPUS = ['README.md', 'CLAUDE.md'];

export function TextTrainTab() {
  const [corpusText, setCorpusText] = useState(DEFAULT_CORPUS.join('\n'));
  const [tokenizerPath, setTokenizerPath] = useState('');

  const [vocabSize, setVocabSize] = useState(4096);
  const [dModel, setDModel] = useState(256);
  const [backbone, setBackbone] = useState<'gru' | 'ssm'>('ssm');

  const [seqLen, setSeqLen] = useState(128);
  const [batchSize, setBatchSize] = useState(32);
  const [steps, setSteps] = useState(2000);
  const [seed, setSeed] = useState(0);
  const [device, setDevice] = useState<'auto' | 'cpu' | 'mps'>('auto');

  const [lr, setLr] = useState(0.003);
  const [weightDecay, setWeightDecay] = useState(0.0);
  const [momentum, setMomentum] = useState(0.95);
  const [nsSteps, setNsSteps] = useState(5);

  const [logEvery, setLogEvery] = useState(20);
  const [saveEvery, setSaveEvery] = useState(200);

  const [activeModelId, setActiveModelId] = useState<string>('');
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [metrics, setMetrics] = useState<TrainMetric[]>([]);
  const [jobs, setJobs] = useState<any[]>([]);
  const [models, setModels] = useState<TextModelSummary[]>([]);

  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pollRef = useRef<number | null>(null);

  const corpusPaths = useMemo(() => {
    return corpusText
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean);
  }, [corpusText]);

  const canStart = !isStarting && corpusPaths.length > 0;

  const refreshJobs = async () => {
    try {
      const j = await listTrainJobs();
      setJobs(j || []);
    } catch {
      setJobs([]);
    }
  };

  const refreshModels = async () => {
    try {
      const ms = await listTextModels();
      setModels(ms || []);
    } catch {
      setModels([]);
    }
  };

  const refreshCatalog = async () => {
    await Promise.all([refreshJobs(), refreshModels()]);
  };

  const refreshStatusAndMetrics = async (modelId: string) => {
    const s = await getTrainStatus(modelId);
    setStatus(s);
    const m = await getTrainMetrics(modelId, 500);
    setMetrics(m || []);
  };

  const start = async () => {
    setIsStarting(true);
    setError(null);
    setStatus(null);
    setMetrics([]);

    try {
      const res = await startTraining({
        corpus_paths: corpusPaths,
        tokenizer_path: tokenizerPath.trim() ? tokenizerPath.trim() : null,
        vocab_size: vocabSize,
        d_model: dModel,
        backbone,
        seq_len: seqLen,
        batch_size: batchSize,
        steps,
        seed,
        device,
        lr,
        weight_decay: weightDecay,
        momentum,
        ns_steps: nsSteps,
        log_every: logEvery,
        save_every: saveEvery,
      });
      setActiveModelId(res.model_id);
      await refreshCatalog();
      await refreshStatusAndMetrics(res.model_id);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setIsStarting(false);
    }
  };

  const cancel = async () => {
    if (!activeModelId) return;
    setError(null);
    try {
      await cancelTraining(activeModelId);
      await refreshCatalog();
      await refreshStatusAndMetrics(activeModelId);
    } catch (e: any) {
      setError(e?.message || String(e));
    }
  };

  useEffect(() => {
    void refreshCatalog();
  }, []);

  useEffect(() => {
    if (!activeModelId) return;
    if (pollRef.current) window.clearInterval(pollRef.current);
    pollRef.current = window.setInterval(() => {
      void refreshStatusAndMetrics(activeModelId).catch(() => {});
      void refreshCatalog();
    }, 2000);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
      pollRef.current = null;
    };
  }, [activeModelId]);

  const latest = status?.latest || null;
  const selectOptions = useMemo(() => {
    const map = new Map<string, { label: string }>();

    for (const m of models) {
      const statusLabel = m.status ? ` (${m.status})` : '';
      map.set(m.model_id, { label: `${m.model_id}${statusLabel}` });
    }

    for (const j of jobs) {
      if (!j?.model_id) continue;
      if (!map.has(j.model_id)) {
        map.set(j.model_id, { label: `${j.model_id} (${j.status})` });
      }
    }

    return Array.from(map.entries()).map(([id, v]) => ({ id, label: v.label }));
  }, [models, jobs]);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        {/* Controls */}
        <div className="bg-surface-50 border border-surface-200 rounded-lg p-4 space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-text-primary">Train (Muon + BPE)</h3>
            <p className="text-xs text-text-muted mt-1">
              Starts a background training process and streams metrics from <span className="font-mono">artifacts/</span>.
            </p>
          </div>

          <label className="text-xs text-text-muted block">
            Corpus files (repo-relative, one per line)
            <textarea
              value={corpusText}
              onChange={(e) => setCorpusText(e.target.value)}
              className="mt-1 w-full h-24 bg-surface-100 border border-surface-200 rounded px-3 py-2 font-mono text-xs text-text-primary resize-y"
            />
          </label>

          <label className="text-xs text-text-muted block">
            Tokenizer path (optional, repo-relative)
            <input
              value={tokenizerPath}
              onChange={(e) => setTokenizerPath(e.target.value)}
              placeholder="artifacts/text_models/<id>/tokenizer.json"
              className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
            />
          </label>

          <div className="grid grid-cols-3 gap-3">
            <label className="text-xs text-text-muted">
              Vocab
              <input
                type="number"
                value={vocabSize}
                onChange={(e) => setVocabSize(Math.max(512, Math.min(65536, Number(e.target.value) || 4096)))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <label className="text-xs text-text-muted">
              d_model
              <input
                type="number"
                value={dModel}
                onChange={(e) => setDModel(Math.max(32, Math.min(4096, Number(e.target.value) || 256)))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <label className="text-xs text-text-muted">
              Backbone
              <select
                value={backbone}
                onChange={(e) => setBackbone(e.target.value as any)}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs"
              >
                <option value="ssm">SSM</option>
                <option value="gru">GRU</option>
              </select>
            </label>

            <label className="text-xs text-text-muted">
              Seq len
              <input
                type="number"
                value={seqLen}
                onChange={(e) => setSeqLen(Math.max(8, Math.min(4096, Number(e.target.value) || 128)))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <label className="text-xs text-text-muted">
              Batch
              <input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(Math.max(1, Math.min(4096, Number(e.target.value) || 32)))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <label className="text-xs text-text-muted">
              Steps
              <input
                type="number"
                value={steps}
                onChange={(e) => setSteps(Math.max(1, Number(e.target.value) || 2000))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>

            <label className="text-xs text-text-muted">
              Device
              <select
                value={device}
                onChange={(e) => setDevice(e.target.value as any)}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs"
              >
                <option value="auto">auto</option>
                <option value="mps">mps</option>
                <option value="cpu">cpu</option>
              </select>
            </label>
            <label className="text-xs text-text-muted">
              Seed
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <div />

            <label className="text-xs text-text-muted">
              LR
              <input
                type="number"
                step="0.0005"
                value={lr}
                onChange={(e) => setLr(Math.max(1e-6, Number(e.target.value) || 0.003))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <label className="text-xs text-text-muted">
              Weight decay
              <input
                type="number"
                step="0.0001"
                value={weightDecay}
                onChange={(e) => setWeightDecay(Math.max(0, Number(e.target.value) || 0))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <label className="text-xs text-text-muted">
              Momentum
              <input
                type="number"
                step="0.01"
                value={momentum}
                onChange={(e) => setMomentum(Math.max(0, Math.min(0.9999, Number(e.target.value) || 0.95)))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <label className="text-xs text-text-muted">
              NS steps
              <input
                type="number"
                value={nsSteps}
                onChange={(e) => setNsSteps(Math.max(1, Math.min(20, Number(e.target.value) || 5)))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>

            <label className="text-xs text-text-muted">
              Log every
              <input
                type="number"
                value={logEvery}
                onChange={(e) => setLogEvery(Math.max(1, Number(e.target.value) || 20))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <label className="text-xs text-text-muted">
              Save every
              <input
                type="number"
                value={saveEvery}
                onChange={(e) => setSaveEvery(Math.max(1, Number(e.target.value) || 200))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <div />
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={start}
              disabled={!canStart}
              className={`px-4 py-2 text-sm rounded font-medium transition-colors ${
                !canStart
                  ? 'bg-surface-200 text-text-muted cursor-not-allowed'
                  : 'bg-accent-blue text-white hover:bg-accent-blue/80'
              }`}
              type="button"
            >
              {isStarting ? 'Starting…' : 'Start training'}
            </button>
            <button
              onClick={cancel}
              disabled={!activeModelId || status?.status !== 'running'}
              className="px-4 py-2 text-sm rounded font-medium bg-surface-200 hover:bg-surface-300 transition-colors disabled:opacity-50"
              type="button"
            >
              Cancel
            </button>
            {error && <span className="text-xs text-accent-red">{error}</span>}
          </div>

          <div className="text-xs text-text-muted">
            Active model: <span className="font-mono text-text-primary">{activeModelId || '—'}</span>
          </div>
        </div>

        {/* Live status + chart */}
        <div className="bg-surface-50 border border-surface-200 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-text-primary">Training progress</h3>
              <p className="text-xs text-text-muted mt-1">
                Loss + grad-norm from <span className="font-mono">train_log.jsonl</span>
              </p>
            </div>
            <button
              onClick={() => void refreshCatalog()}
              className="px-3 py-1.5 text-xs bg-surface-200 hover:bg-surface-300 rounded transition-colors"
              type="button"
            >
              Refresh
            </button>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-text-muted">Jobs</span>
            <select
              value={activeModelId}
              onChange={async (e) => {
                const id = e.target.value;
                setActiveModelId(id);
                setError(null);
                setStatus(null);
                setMetrics([]);
                if (!id) return;
                try {
                  await refreshStatusAndMetrics(id);
                } catch (err: any) {
                  setError(err?.message || String(err));
                }
              }}
              className="flex-1 bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
            >
              <option value="">(select)</option>
              {selectOptions.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-4 gap-3 text-xs">
            <div>
              <div className="text-text-muted">status</div>
              <div className="font-mono text-text-primary">{status?.status ?? '—'}</div>
            </div>
            <div>
              <div className="text-text-muted">step</div>
              <div className="font-mono text-text-primary">{latest?.step ?? '—'}</div>
            </div>
            <div>
              <div className="text-text-muted">loss</div>
              <div className="font-mono text-text-primary">{latest?.loss == null ? '—' : formatNumber(latest.loss, 4)}</div>
            </div>
            <div>
              <div className="text-text-muted">grad</div>
              <div className="font-mono text-text-primary">
                {latest?.grad_norm == null ? '—' : formatNumber(latest.grad_norm, 4)}
              </div>
            </div>
          </div>

          {(status?.exit_code != null || status?.error) && (
            <div className="text-xs">
              {status?.exit_code != null && (
                <div className={status.exit_code === 0 ? 'text-text-muted' : 'text-accent-red'}>
                  exit_code: <span className="font-mono">{status.exit_code}</span>
                </div>
              )}
              {status?.error && <div className="text-accent-red font-mono break-words mt-1">{status.error}</div>}
            </div>
          )}

          <div className="h-64">
            {!metrics.length ? (
              <div className="text-sm text-text-muted flex items-center justify-center h-full">
                Start a run to see live metrics.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="step" stroke="#94a3b8" tick={{ fontSize: 12 }} />
                  <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8 }}
                    labelStyle={{ color: '#e2e8f0' }}
                    itemStyle={{ color: '#e2e8f0' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="loss" stroke="#60a5fa" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="grad_norm" stroke="#34d399" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      <div className="bg-accent-blue/10 border border-accent-blue/30 rounded-lg p-4 text-sm text-text-secondary">
        <strong className="text-accent-blue">Note:</strong> the server restricts corpus/tokenizer paths to files under this repo.
      </div>
    </div>
  );
}
