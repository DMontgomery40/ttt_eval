import { Arrow, DiagramBox } from './FlowDiagram';

function InlineCode({ children }: { children: string }) {
  return <span className="font-mono text-xs text-text-primary">{children}</span>;
}

export function SleepArchitectureSection() {
  return (
    <div className="bg-surface-50 border border-surface-200 rounded-lg p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-text-primary">Sleep Consolidation (prototype)</h3>
        <span className="text-xs text-text-muted">fast → slow</span>
      </div>

      <p className="text-sm text-text-secondary">
        Sleep is the first pass at “end-of-day consolidation”: replaying chat traces into the slow/core model and writing
        a new candidate checkpoint. This is intentionally conservative and not yet the end-goal “selective delta
        transfer”.
      </p>

      <div className="flex items-center gap-2 overflow-x-auto py-2">
        <DiagramBox label="text_sessions/*" sublabel="trace.jsonl" color="#8b949e" />
        <Arrow />
        <DiagramBox label="sleep" sublabel="Muon" color="#39d353" highlight />
        <Arrow />
        <DiagramBox label="text_models/*" sublabel="sleep_candidate" color="#58a6ff" />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-surface-100 border border-surface-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-text-primary mb-2">What it reads</h4>
          <ul className="text-xs text-text-secondary list-disc pl-5 space-y-1">
            <li>
              Chat traces: <InlineCode>artifacts/text_sessions/&lt;session_id&gt;/trace.jsonl</InlineCode>
            </li>
            <li>
              Base checkpoint: <InlineCode>artifacts/text_models/&lt;base_model_id&gt;/checkpoint.pt</InlineCode>
            </li>
            <li>
              Optional “core” text: <InlineCode>--core_path</InlineCode> (either a single <InlineCode>.txt</InlineCode> or
              a <InlineCode>.jsonl</InlineCode> with <InlineCode>{"{\"text\": ...}"}</InlineCode>)
            </li>
          </ul>
        </div>

        <div className="bg-surface-100 border border-surface-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-text-primary mb-2">What it writes</h4>
          <ul className="text-xs text-text-secondary list-disc pl-5 space-y-1">
            <li>
              New model dir: <InlineCode>artifacts/text_models/&lt;sleep_id&gt;/</InlineCode>
            </li>
            <li>
              Candidate checkpoint: <InlineCode>checkpoint.pt</InlineCode> (updated weights)
            </li>
            <li>
              Manifest: <InlineCode>sleep_manifest.json</InlineCode> (pre/post loss on core + mix)
            </li>
            <li>
              Index record: <InlineCode>status="sleep_candidate"</InlineCode> (excluded from chat auto-selection)
            </li>
          </ul>
        </div>
      </div>

      <div className="bg-surface-100 border border-surface-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-text-primary mb-2">Current training scope</h4>
        <ul className="text-xs text-text-secondary list-disc pl-5 space-y-1">
          <li>Only `TinyLm.backbone` and `TinyLm.ln` are trained (embed + head remain frozen).</li>
          <li>Muon is used at a very low LR by default.</li>
          <li>This is trace replay consolidation; selective delta transfer is a future step.</li>
        </ul>
      </div>
    </div>
  );
}

