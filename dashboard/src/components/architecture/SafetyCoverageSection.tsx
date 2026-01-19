function Badge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
        ok ? 'bg-accent-green/15 text-accent-green border border-accent-green/30' : 'bg-surface-200 text-text-muted'
      }`}
    >
      {label}
    </span>
  );
}

type Row = {
  mechanism: string;
  sentry: boolean;
  chat: boolean;
  sleep: boolean;
  nano: boolean;
  notes: string;
};

const rows: Row[] = [
  {
    mechanism: 'Online fast-weight updates (TTT)',
    sentry: true,
    chat: true,
    sleep: false,
    nano: true,
    notes: 'Adapter in Sentry; context net in Chat; plastic matrices in Nano.',
  },
  {
    mechanism: 'Rule-based pre-update gate',
    sentry: true,
    chat: true,
    sleep: false,
    nano: false,
    notes: 'Same gate logic is available in Chat as an opt-in per-session toggle (defaults off).',
  },
  {
    mechanism: 'Rollback canary probe',
    sentry: true,
    chat: true,
    sleep: false,
    nano: true,
    notes: 'Chat can probe canary loss pre/post update and rollback (opt-in); Nano logs commit/rollback events.',
  },
  {
    mechanism: 'Directional canary signals',
    sentry: true,
    chat: true,
    sleep: false,
    nano: true,
    notes: 'Chat computes canary gradients when SPFW is enabled; Sentry can monitor alignment and/or constrain writes.',
  },
  {
    mechanism: 'SPFW projection',
    sentry: true,
    chat: true,
    sleep: false,
    nano: false,
    notes: 'Half-space projection based on canary gradients (supports multi-canary; includes a stall guard).',
  },
  {
    mechanism: 'Sleep consolidation (fast→slow)',
    sentry: false,
    chat: false,
    sleep: true,
    nano: false,
    notes: 'Trace replay into backbone+LN only; writes a sleep-candidate checkpoint.',
  },
];

export function SafetyCoverageSection() {
  return (
    <div className="bg-surface-50 border border-surface-200 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-text-primary">Safety Coverage (current wiring)</h3>
      <p className="text-sm text-text-secondary mt-2">
        This table is intentionally “honest wiring”, not aspirational design. Some mechanisms exist only in specific
        tracks today.
      </p>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="text-xs text-text-muted">
              <th className="py-2 pr-4">Mechanism</th>
              <th className="py-2 pr-4">TTT Sentry</th>
              <th className="py-2 pr-4">Chat TTT</th>
              <th className="py-2 pr-4">Sleep</th>
              <th className="py-2 pr-4">Nano</th>
              <th className="py-2">Notes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.mechanism} className="border-t border-surface-200">
                <td className="py-3 pr-4 text-text-primary">{r.mechanism}</td>
                <td className="py-3 pr-4">
                  <Badge ok={r.sentry} label={r.sentry ? 'yes' : 'no'} />
                </td>
                <td className="py-3 pr-4">
                  <Badge ok={r.chat} label={r.chat ? 'yes' : 'no'} />
                </td>
                <td className="py-3 pr-4">
                  <Badge ok={r.sleep} label={r.sleep ? 'yes' : 'no'} />
                </td>
                <td className="py-3 pr-4">
                  <Badge ok={r.nano} label={r.nano ? 'yes' : 'no'} />
                </td>
                <td className="py-3 text-text-secondary">{r.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
