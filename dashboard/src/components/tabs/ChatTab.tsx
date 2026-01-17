import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { generateText, listTextModels, type TextModelSummary } from '../../api/textLmApi';

export function ChatTab() {
  const [models, setModels] = useState<TextModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');

  const [prompt, setPrompt] = useState('Hello! Write a short paragraph about test-time training.');
  const [output, setOutput] = useState<string>('');
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [maxNewTokens, setMaxNewTokens] = useState(160);
  const [temperature, setTemperature] = useState(0.9);
  const [topK, setTopK] = useState(50);

  const refreshModels = async () => {
    try {
      const ms = await listTextModels();
      setModels(ms || []);
      if (!selectedModelId && ms && ms.length > 0) {
        setSelectedModelId(ms[0].model_id);
      }
    } catch (e: any) {
      setModels([]);
      if (!selectedModelId) setSelectedModelId('');
    }
  };

  useEffect(() => {
    void refreshModels();
  }, []);

  const canRun = useMemo(() => {
    return !isRunning && prompt.trim().length > 0;
  }, [isRunning, prompt]);

  const run = async () => {
    setIsRunning(true);
    setError(null);
    setOutput('');

    try {
      const res = await generateText({
        prompt,
        model_id: selectedModelId || undefined,
        max_new_tokens: maxNewTokens,
        temperature,
        top_k: topK,
      });
      setOutput(res.text || '');
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        {/* Prompt */}
        <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-sm font-semibold text-text-primary">Generate</h3>
              <p className="text-xs text-text-muted mt-1">
                BPE tiny LM (trained locally) via <span className="font-mono">/api/text/generate</span>
              </p>
            </div>
            <button
              onClick={() => void refreshModels()}
              className="px-3 py-1.5 text-xs bg-surface-200 hover:bg-surface-300 rounded transition-colors"
              type="button"
            >
              Refresh models
            </button>
          </div>

          <label className="text-xs text-text-muted">
            Model
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
            >
              <option value="">(latest)</option>
              {models.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {m.model_id}
                </option>
              ))}
            </select>
          </label>

          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full h-56 bg-surface-100 border border-surface-200 rounded px-3 py-2 font-mono text-xs text-text-primary resize-y mt-3"
          />

          <div className="grid grid-cols-3 gap-3 mt-4">
            <label className="text-xs text-text-muted">
              Max new tokens
              <input
                type="number"
                value={maxNewTokens}
                onChange={(e) => setMaxNewTokens(Math.max(1, Math.min(2048, Number(e.target.value) || 160)))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <label className="text-xs text-text-muted">
              Temperature
              <input
                type="number"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(Math.max(0.05, Math.min(5, Number(e.target.value) || 0.9)))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <label className="text-xs text-text-muted">
              Top-k
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(Math.max(0, Math.min(5000, Number(e.target.value) || 50)))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
          </div>

          <div className="flex items-center gap-2 mt-4">
            <button
              onClick={run}
              disabled={!canRun}
              className={`px-4 py-2 text-sm rounded font-medium transition-colors ${
                !canRun
                  ? 'bg-surface-200 text-text-muted cursor-not-allowed'
                  : 'bg-accent-blue text-white hover:bg-accent-blue/80'
              }`}
              type="button"
            >
              {isRunning ? 'Generating…' : 'Generate'}
            </button>
            {error && <span className="text-xs text-accent-red">{error}</span>}
          </div>
        </div>

        {/* Output */}
        <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-sm font-semibold text-text-primary">Output</h3>
              <p className="text-xs text-text-muted mt-1">
                Train a model first with <span className="font-mono">python -m ttt.text_lm.train</span>
              </p>
            </div>
          </div>

          {!output ? (
            <div className="text-sm text-text-muted flex items-center justify-center h-64">
              {isRunning ? 'Generating…' : 'No output yet.'}
            </div>
          ) : (
            <motion.pre
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-surface-100 border border-surface-200 rounded-lg p-3 text-xs text-text-primary whitespace-pre-wrap font-mono max-h-[520px] overflow-y-auto"
            >
              {output}
            </motion.pre>
          )}
        </div>
      </div>
    </div>
  );
}

