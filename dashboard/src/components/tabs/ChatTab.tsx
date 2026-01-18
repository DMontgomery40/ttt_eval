import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { listTextModels, type TextModelSummary } from '../../api/textLmApi';
import {
  chatInSession,
  createTextSession,
  listTextSessions,
  resetTextSession,
  type TextSessionSummary,
} from '../../api/textChatApi';
import { ChatUpdateEvent } from '../../types';

export function ChatTab() {
  const [models, setModels] = useState<TextModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');

  const [sessions, setSessions] = useState<TextSessionSummary[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string>(() => {
    return window.localStorage.getItem('ttt_text_session_id') || '';
  });

  const [prompt, setPrompt] = useState('Hello! Write a short paragraph about test-time training.');
  const [output, setOutput] = useState<string>('');
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Latency tracking
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number>(0);

  // Response metadata
  const [updateEvents, setUpdateEvents] = useState<ChatUpdateEvent[]>([]);
  const [generatedTokens, setGeneratedTokens] = useState<number>(0);

  // Context-net (TTT) defaults for new sessions.
  const [contextLr, setContextLr] = useState(0.02);
  const [stepsPerMessage, setStepsPerMessage] = useState(1);
  const [chunkTokens, setChunkTokens] = useState(128);

  const [maxNewTokens, setMaxNewTokens] = useState(160);
  const [temperature, setTemperature] = useState(0.9);
  const [topK, setTopK] = useState(50);

  const refreshModels = async () => {
    try {
      const ms = await listTextModels();
      setModels(ms || []);
    } catch (e: any) {
      setModels([]);
      if (!selectedModelId) setSelectedModelId('');
    }
  };

  const refreshSessions = async () => {
    try {
      const ss = await listTextSessions(100);
      setSessions(ss || []);
      if (selectedSessionId && ss?.some((s) => s.session_id === selectedSessionId)) return;
      if (ss && ss.length > 0) {
        setSelectedSessionId(ss[0].session_id);
        window.localStorage.setItem('ttt_text_session_id', ss[0].session_id);
      }
    } catch {
      setSessions([]);
    }
  };

  useEffect(() => {
    void refreshModels();
    void refreshSessions();
  }, []);

  // Live timer while running
  useEffect(() => {
    if (!isRunning || !startTime) return;

    const interval = setInterval(() => {
      setElapsedMs(Date.now() - startTime);
    }, 50); // Update every 50ms for smooth animation

    return () => clearInterval(interval);
  }, [isRunning, startTime]);

  const canRun = useMemo(() => {
    return !isRunning && prompt.trim().length > 0;
  }, [isRunning, prompt]);

  const createSession = async () => {
    const res = await createTextSession({
      model_id: selectedModelId || null,
      lr: contextLr,
      steps_per_message: stepsPerMessage,
      chunk_tokens: chunkTokens,
    });
    const sid = String(res?.session_id || '').trim();
    if (!sid) throw new Error('Server did not return a session_id');
    setSelectedSessionId(sid);
    window.localStorage.setItem('ttt_text_session_id', sid);
    await refreshSessions();
    return sid;
  };

  const resetSession = async () => {
    if (!selectedSessionId) return;
    await resetTextSession(selectedSessionId);
  };

  const run = async () => {
    setIsRunning(true);
    setError(null);
    setOutput('');
    setUpdateEvents([]);
    setGeneratedTokens(0);

    // Start timer
    const start = Date.now();
    setStartTime(start);
    setElapsedMs(0);

    try {
      const sid = selectedSessionId || (await createSession());
      const res = await chatInSession({
        session_id: sid,
        prompt,
        max_new_tokens: maxNewTokens,
        temperature,
        top_k: topK,
      });

      // Stop timer
      const end = Date.now();
      setElapsedMs(end - start);

      // Capture response data
      setOutput(res.completion || res.text || '');
      setUpdateEvents(res.update_events || []);
      setGeneratedTokens((res.completion || res.text || '').split(/\s+/).filter(Boolean).length);
    } catch (e: any) {
      setError(e?.message || String(e));
      const end = Date.now();
      setElapsedMs(end - start);
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
                BPE tiny LM + fast context net (TTT) via <span className="font-mono">/api/text/sessions/*/chat</span>
              </p>
            </div>
            <button
              onClick={() => void Promise.all([refreshModels(), refreshSessions()])}
              className="px-3 py-1.5 text-xs bg-surface-200 hover:bg-surface-300 rounded transition-colors"
              type="button"
            >
              Refresh
            </button>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs text-text-muted">
              Core model
              <select
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              >
                <option value="">(latest usable)</option>
                {models.map((m) => (
                  <option key={m.model_id} value={m.model_id}>
                    {m.model_id}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-xs text-text-muted">
              Session
              <select
                value={selectedSessionId}
                onChange={(e) => {
                  const id = e.target.value;
                  setSelectedSessionId(id);
                  window.localStorage.setItem('ttt_text_session_id', id);
                }}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              >
                <option value="">(new)</option>
                {sessions.map((s) => (
                  <option key={s.session_id} value={s.session_id}>
                    {s.session_id}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="grid grid-cols-3 gap-3 mt-3">
            <label className="text-xs text-text-muted">
              Context LR
              <input
                type="number"
                step="0.001"
                value={contextLr}
                onChange={(e) => setContextLr(Math.max(1e-6, Number(e.target.value) || 0.02))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <label className="text-xs text-text-muted">
              Steps/msg
              <input
                type="number"
                value={stepsPerMessage}
                onChange={(e) => setStepsPerMessage(Math.max(1, Math.min(128, Number(e.target.value) || 1)))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
            <label className="text-xs text-text-muted">
              Chunk tokens
              <input
                type="number"
                value={chunkTokens}
                onChange={(e) => setChunkTokens(Math.max(8, Math.min(8192, Number(e.target.value) || 128)))}
                className="mt-1 w-full bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs font-mono"
              />
            </label>
          </div>

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
            <button
              onClick={() => void createSession().catch((e: any) => setError(e?.message || String(e)))}
              className="px-3 py-2 text-sm rounded font-medium bg-surface-200 hover:bg-surface-300 transition-colors"
              type="button"
            >
              New session
            </button>
            <button
              onClick={() => void resetSession().catch((e: any) => setError(e?.message || String(e)))}
              disabled={!selectedSessionId}
              className="px-3 py-2 text-sm rounded font-medium bg-surface-200 hover:bg-surface-300 transition-colors disabled:opacity-50"
              type="button"
            >
              Reset
            </button>
            {error && <span className="text-xs text-accent-red">{error}</span>}
          </div>
        </div>

        {/* Output + Stats */}
        <div className="bg-surface-50 border border-surface-200 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-text-primary">Output</h3>
              <p className="text-xs text-text-muted mt-1">
                Train a core model first with <span className="font-mono">python -m ttt.text_lm.train</span> (or the Train tab).
              </p>
            </div>
            {/* Live latency counter */}
            {isRunning && startTime && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-2 bg-accent-blue/20 px-3 py-1.5 rounded-lg"
              >
                <div className="w-2 h-2 bg-accent-blue rounded-full animate-pulse" />
                <span className="font-mono text-sm text-accent-blue font-bold">
                  {(elapsedMs / 1000).toFixed(2)}s
                </span>
              </motion.div>
            )}
          </div>

          {!output ? (
            <div className="text-sm text-text-muted flex flex-col items-center justify-center h-64 gap-3">
              {isRunning ? (
                <>
                  <div className="text-accent-blue font-medium">Generating response...</div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                  <div className="text-xs text-text-muted">
                    Elapsed: <span className="font-mono text-accent-blue">{(elapsedMs / 1000).toFixed(2)}s</span>
                  </div>
                </>
              ) : (
                'No output yet.'
              )}
            </div>
          ) : (
            <>
              {/* Output text */}
              <motion.pre
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-surface-100 border border-surface-200 rounded-lg p-3 text-xs text-text-primary whitespace-pre-wrap font-mono max-h-64 overflow-y-auto"
              >
                {output}
              </motion.pre>

              {/* Comprehensive statistics */}
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="space-y-3"
              >
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-semibold text-text-primary">Generation Statistics</h4>
                  <span className="text-xs text-text-muted">
                    Total latency: <span className="font-mono text-accent-green font-bold">{(elapsedMs / 1000).toFixed(3)}s</span>
                  </span>
                </div>

                {/* Stats grid */}
                <div className="grid grid-cols-3 gap-2">
                  {/* Latency */}
                  <div className="bg-surface-100 rounded-lg p-2">
                    <div className="text-xs text-text-muted mb-1">Total Time</div>
                    <div className="font-mono text-sm text-accent-blue">
                      {(elapsedMs / 1000).toFixed(3)}s
                    </div>
                  </div>

                  {/* Words generated */}
                  <div className="bg-surface-100 rounded-lg p-2">
                    <div className="text-xs text-text-muted mb-1">Words</div>
                    <div className="font-mono text-sm text-accent-purple">
                      {generatedTokens}
                    </div>
                  </div>

                  {/* Speed */}
                  <div className="bg-surface-100 rounded-lg p-2">
                    <div className="text-xs text-text-muted mb-1">Words/sec</div>
                    <div className="font-mono text-sm text-accent-green">
                      {elapsedMs > 0 ? ((generatedTokens / elapsedMs) * 1000).toFixed(1) : '—'}
                    </div>
                  </div>

                  {/* TTT updates */}
                  <div className="bg-surface-100 rounded-lg p-2">
                    <div className="text-xs text-text-muted mb-1">TTT Updates</div>
                    <div className="font-mono text-sm text-accent-gold">
                      {updateEvents.length}
                    </div>
                  </div>

                  {/* Avg loss */}
                  <div className="bg-surface-100 rounded-lg p-2">
                    <div className="text-xs text-text-muted mb-1">Avg Loss</div>
                    <div className="font-mono text-sm text-text-primary">
                      {updateEvents.length > 0
                        ? (updateEvents.reduce((sum, e) => sum + (e.loss || 0), 0) / updateEvents.length).toFixed(4)
                        : '—'}
                    </div>
                  </div>

                  {/* Avg grad norm */}
                  <div className="bg-surface-100 rounded-lg p-2">
                    <div className="text-xs text-text-muted mb-1">Avg Grad</div>
                    <div className="font-mono text-sm text-accent-blue">
                      {updateEvents.length > 0
                        ? (updateEvents.reduce((sum, e) => sum + (e.grad_norm || 0), 0) / updateEvents.length).toFixed(4)
                        : '—'}
                    </div>
                  </div>
                </div>

                {/* TTT Update Events Table */}
                {updateEvents.length > 0 && (
                  <div className="bg-surface-100 rounded-lg p-3">
                    <div className="text-xs font-semibold text-text-primary mb-2">TTT Adaptation Details</div>
                    <div className="max-h-32 overflow-y-auto">
                      <table className="w-full text-xs">
                        <thead className="bg-surface-200 text-text-muted sticky top-0">
                          <tr>
                            <th className="px-2 py-1 text-left">Chunk</th>
                            <th className="px-2 py-1 text-left">Tokens</th>
                            <th className="px-2 py-1 text-right">Step</th>
                            <th className="px-2 py-1 text-right">Loss</th>
                            <th className="px-2 py-1 text-right">Grad</th>
                            <th className="px-2 py-1 text-right">Update</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-surface-200">
                          {updateEvents.map((evt, idx) => (
                            <tr key={idx} className="hover:bg-surface-50">
                              <td className="px-2 py-1 font-mono text-accent-gold">#{evt.chunk_index}</td>
                              <td className="px-2 py-1 font-mono text-text-secondary">
                                {evt.token_start}–{evt.token_end}
                              </td>
                              <td className="px-2 py-1 font-mono text-right text-accent-blue">{evt.step_in_chunk}</td>
                              <td className="px-2 py-1 font-mono text-right text-accent-purple">
                                {evt.loss?.toFixed(4) || '—'}
                              </td>
                              <td className="px-2 py-1 font-mono text-right text-accent-green">
                                {evt.grad_norm?.toFixed(4) || '—'}
                              </td>
                              <td className="px-2 py-1 font-mono text-right text-accent-orange">
                                {evt.update_norm?.toFixed(4) || '—'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </motion.div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
