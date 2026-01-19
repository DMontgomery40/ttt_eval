import { Arrow, DiagramBox } from './FlowDiagram';

function InlineCode({ children }: { children: string }) {
  return <span className="font-mono text-xs text-text-primary">{children}</span>;
}

export function TextArchitectureSection() {
  return (
    <div className="space-y-6">
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-text-primary">Text Domain (two codepaths)</h3>
        <p className="text-sm text-text-secondary mt-2">
          The repo has two parallel text tracks with different models/tokenizers. They now share the same safety harness
          primitives (gate / rollback / SPFW) and a largely aligned event schema, but they are still separate model
          implementations.
        </p>

        <div className="grid grid-cols-2 gap-4 mt-6">
          {/* World A */}
          <div className="bg-surface-100 border border-surface-200 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-text-primary">World A — TTT Sentry</h4>
              <span className="text-xs text-text-muted">ToyTTTModel + monitor</span>
            </div>

            <p className="text-xs text-text-secondary">
              Purpose-built safety eval harness: gate + rollback + directional signals, instrumented per-chunk.
            </p>

            <div className="text-xs text-text-muted">
              Entry points: <InlineCode>run_monitor.py</InlineCode>, <InlineCode>/api/text/runs</InlineCode>
            </div>

            <div className="text-xs text-text-muted">Model path:</div>
            <div className="flex items-center gap-2 overflow-x-auto py-2">
              <DiagramBox label="regex/hash" sublabel="tokenize" color="#8b949e" />
              <Arrow />
              <DiagramBox label="embed" color="#58a6ff" />
              <Arrow />
              <DiagramBox label="backbone" sublabel="SSM|GRU" color="#a371f7" />
              <Arrow />
              <DiagramBox label="LN" color="#8b949e" />
              <Arrow />
              <DiagramBox label="adapter" sublabel="plastic" color="#39d353" plastic />
              <Arrow />
              <DiagramBox label="head" sublabel="logits" color="#58a6ff" />
            </div>

            <div className="text-xs text-text-muted">Update loop (per chunk):</div>
            <ul className="text-xs text-text-secondary list-disc pl-5 space-y-1">
              <li>Compute AR/MLM loss + adapter grad norm (“write pressure”).</li>
              <li>Gate decision (entropy/diversity/blob/override/OOD+heavy-write).</li>
              <li>Apply update (or project grads via SPFW then step).</li>
              <li>Canary probe and rollback on regression (“transaction semantics”).</li>
            </ul>
          </div>

          {/* World B */}
          <div className="bg-surface-100 border border-surface-200 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-text-primary">World B — TinyLM + fast weights</h4>
              <span className="text-xs text-text-muted">Train + Chat</span>
            </div>

            <p className="text-xs text-text-secondary">
              Runnable tiny LM from scratch (BPE) plus a per-session fast context module updated online (TTT).
            </p>

            <div className="text-xs text-text-muted">
              Entry points: <InlineCode>ttt.text_lm.train</InlineCode>, <InlineCode>/api/text/sessions/*/chat</InlineCode>
            </div>

            <div className="text-xs text-text-muted">Core model (no attention):</div>
            <div className="flex items-center gap-2 overflow-x-auto py-2">
              <DiagramBox label="BPE" sublabel="tokenize" color="#8b949e" />
              <Arrow />
              <DiagramBox label="embed" color="#58a6ff" />
              <Arrow />
              <DiagramBox label="backbone" sublabel="SSM|GRU" color="#a371f7" />
              <Arrow />
              <DiagramBox label="LN" color="#8b949e" />
              <Arrow />
              <DiagramBox label="head" sublabel="logits" color="#58a6ff" />
            </div>

            <div className="text-xs text-text-muted">Fast context net (chat):</div>
            <div className="flex items-center gap-2 overflow-x-auto py-2">
              <DiagramBox label="hidden h" sublabel="core" color="#8b949e" />
              <Arrow />
              <DiagramBox label="+ context(h)" sublabel="TTT (fast weights)" color="#39d353" plastic />
              <Arrow />
              <DiagramBox label="logits" sublabel="sample" color="#58a6ff" />
              <Arrow />
              <DiagramBox label="text_sessions/" sublabel="state+trace" color="#8b949e" />
            </div>

            <div className="text-xs text-text-muted">Update loop (per chunk, chat):</div>
            <ul className="text-xs text-text-secondary list-disc pl-5 space-y-1">
              <li>Compute a fast-weight loss (CE for linear adapter, or associative loss for fast memory).</li>
              <li>Optional gate: block means “skip write” (generation still runs).</li>
              <li>Optional SPFW: project context gradients into a safe subspace before Muon step.</li>
              <li>Optional rollback: probe canary before/after and restore previous state on regression.</li>
            </ul>

            <div className="text-xs text-text-muted">What updates where:</div>
            <ul className="text-xs text-text-secondary list-disc pl-5 space-y-1">
              <li>Train tab updates the core model (Muon; offline).</li>
              <li>Chat tab updates the fast context net only (Muon; online; core frozen).</li>
              <li>Sleep replay updates backbone+LN only and writes a new candidate checkpoint.</li>
              <li>Muon uses a fallback path when parameters are not 2D (e.g. LayerNorm and SSM diagonals).</li>
            </ul>

            <div className="bg-surface-50 border border-surface-200 rounded p-3 text-xs text-text-secondary">
              Safety note: gate / rollback / SPFW are available for chat fast-weight updates (all opt-in; defaults off per
              session). Canary texts are used for both rollback probing and SPFW constraints.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
