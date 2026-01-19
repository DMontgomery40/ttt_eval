import { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

import type { ModelConfig, PlasticityConfig } from '../../types';
import { Arrow, DiagramBox } from './FlowDiagram';

export function NanoArchitectureSection({
  modelCfg,
  plasticityCfg,
}: {
  modelCfg: ModelConfig;
  plasticityCfg: PlasticityConfig;
}) {
  const paramBreakdown = useMemo(
    () => [
      {
        name: 'W_u',
        shape: `[${modelCfg.u_dim}, ${modelCfg.z_dim + modelCfg.act_dim}]`,
        count: modelCfg.u_dim * (modelCfg.z_dim + modelCfg.act_dim),
        plastic: true,
        color: '#58a6ff',
      },
      {
        name: 'B',
        shape: `[${modelCfg.n_state}, ${modelCfg.u_dim}]`,
        count: modelCfg.n_state * modelCfg.u_dim,
        plastic: true,
        color: '#39d353',
      },
      {
        name: 'W_o',
        shape: `[${modelCfg.z_dim}, ${modelCfg.n_state}]`,
        count: modelCfg.z_dim * modelCfg.n_state,
        plastic: true,
        color: '#a371f7',
      },
      {
        name: 'a_raw',
        shape: `[${modelCfg.n_state}]`,
        count: modelCfg.n_state,
        plastic: false,
        color: '#6e7681',
      },
    ],
    [modelCfg]
  );

  const totalPlastic = paramBreakdown.filter((p) => p.plastic).reduce((a, b) => a + b.count, 0);
  const totalFrozen = paramBreakdown.filter((p) => !p.plastic).reduce((a, b) => a + b.count, 0);

  return (
    <div className="space-y-6">
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-text-primary mb-2">Nano — Diagonal Stable SSM (Phase 1)</h3>
        <p className="text-sm text-text-secondary">
          Phase 1 Nano is a constrained, interpretable online-learning SSM: a stability-guaranteed recurrence plus a small
          set of plastic matrices updated with Muon.
        </p>

        <div className="flex items-center justify-center gap-2 overflow-x-auto py-6">
          <DiagramBox label="[z_t, a_t]" sublabel={`[${modelCfg.z_dim + modelCfg.act_dim}]`} color="#8b949e" />
          <Arrow />
          <DiagramBox label="W_u" sublabel="input proj" color="#58a6ff" plastic />
          <Arrow />
          <DiagramBox label="u_t" sublabel={`[${modelCfg.u_dim}]`} color="#8b949e" />
          <Arrow />
          <DiagramBox label="B" sublabel="state update" color="#39d353" plastic />
          <Arrow />
          <div className="bg-surface-100 border-2 border-accent-orange rounded-lg p-4 text-center">
            <div className="font-mono text-sm text-accent-orange mb-1">SSM Core</div>
            <div className="text-xs text-text-muted">decay ⊙ h + B u</div>
            <div className="text-xs text-text-muted mt-1">decay = exp(A·dt)</div>
          </div>
          <Arrow />
          <DiagramBox label="h_{t+1}" sublabel={`[${modelCfg.n_state}]`} color="#8b949e" />
          <Arrow />
          <DiagramBox label="W_o" sublabel="output proj" color="#a371f7" plastic />
          <Arrow />
          <DiagramBox label="ẑ_{t+1}" sublabel={`[${modelCfg.z_dim}]`} color="#39d353" highlight />
        </div>

        <div className="flex items-center justify-center gap-6 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded border-2 border-dashed border-accent-green" />
            <span className="text-text-muted">Plastic</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-surface-100 border border-surface-200" />
            <span className="text-text-muted">Frozen</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded border-2 border-accent-orange" />
            <span className="text-text-muted">Stability-guaranteed core</span>
          </div>
        </div>
      </div>

      <div className="bg-surface-50 border border-surface-200 rounded-lg p-6">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Stability guarantee</h3>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <p className="text-sm text-text-secondary mb-4">
              The recurrence uses a diagonal transition where the decay is constrained into (0, 1). Phase 1 freezes the
              stability parameter so online updates can’t destabilize the core dynamics.
            </p>

            <div className="bg-surface-100 rounded-lg p-4 font-mono text-sm space-y-2">
              <div>
                <span className="text-text-muted">A</span>
                <span className="text-text-primary mx-2">=</span>
                <span className="text-accent-orange">-softplus(a_raw)</span>
                <span className="text-text-muted ml-2">{'// A < 0 always'}</span>
              </div>
              <div>
                <span className="text-text-muted">decay</span>
                <span className="text-text-primary mx-2">=</span>
                <span className="text-accent-green">exp(A × dt)</span>
                <span className="text-text-muted ml-2">{'// decay ∈ (0, 1)'}</span>
              </div>
            </div>
          </div>

          <div className="bg-accent-green/10 border border-accent-green/30 rounded-lg p-4">
            <p className="text-accent-green font-medium mb-2">Why this matters</p>
            <p className="text-sm text-text-secondary">
              With |decay| {'<'} 1 for all hidden dimensions, the hidden state naturally contracts. That keeps the system
              from exploding even when plastic weights are being updated online.
            </p>
          </div>
        </div>
      </div>

      <div className="bg-surface-50 border border-surface-200 rounded-lg p-6">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Parameter distribution</h3>

        <div className="grid grid-cols-2 gap-6">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={paramBreakdown} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
              <XAxis type="number" stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 11 }} />
              <YAxis type="category" dataKey="name" stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#161b22',
                  border: '1px solid #30363d',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => [value.toLocaleString(), 'Parameters']}
              />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {paramBreakdown.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div className="space-y-4">
            {paramBreakdown.map((param) => (
              <div key={param.name} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: param.color }} />
                  <span className="font-mono">{param.name}</span>
                  <span className="text-xs text-text-muted">{param.shape}</span>
                </div>
                <div className="text-right">
                  <span className="font-mono font-bold">{param.count.toLocaleString()}</span>
                  <span
                    className={`ml-2 text-xs px-1.5 py-0.5 rounded ${
                      param.plastic ? 'bg-accent-green/20 text-accent-green' : 'bg-surface-200 text-text-muted'
                    }`}
                  >
                    {param.plastic ? 'plastic' : 'frozen'}
                  </span>
                </div>
              </div>
            ))}

            <div className="pt-4 border-t border-surface-200">
              <div className="flex justify-between text-sm">
                <span className="text-text-muted">Total plastic:</span>
                <span className="font-mono font-bold text-accent-green">{totalPlastic.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-sm mt-1">
                <span className="text-text-muted">Total frozen:</span>
                <span className="font-mono font-bold text-text-secondary">{totalFrozen.toLocaleString()}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-surface-50 border border-surface-200 rounded-lg p-6">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Muon optimizer (Phase 1 plasticity)</h3>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <p className="text-sm text-text-secondary mb-4">
              Muon orthogonalizes gradient matrices via Newton–Schulz iteration before applying updates. This is used for
              online plastic matrices to reduce directional collapse and keep updates stable.
            </p>

            <div className="bg-surface-100 rounded-lg p-4 space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-text-muted">Learning rate</span>
                <span className="font-mono text-accent-blue">{plasticityCfg.lr}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Momentum</span>
                <span className="font-mono text-accent-blue">{plasticityCfg.momentum}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Nesterov</span>
                <span className="font-mono text-accent-blue">{plasticityCfg.nesterov ? 'Yes' : 'No'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">NS iterations</span>
                <span className="font-mono text-accent-blue">{plasticityCfg.ns_steps}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Weight decay</span>
                <span className="font-mono text-accent-blue">{plasticityCfg.weight_decay}</span>
              </div>
            </div>
          </div>

          <div className="bg-accent-blue/10 border border-accent-blue/30 rounded-lg p-4">
            <p className="text-accent-blue font-medium mb-2">Newton–Schulz orthogonalization</p>
            <p className="text-sm text-text-secondary mb-3">
              Each gradient matrix is iteratively transformed to approximate an orthogonal matrix direction.
            </p>
            <code className="block bg-surface rounded p-2 font-mono text-xs text-text-primary">
              X ← a·X + b·(X·X^T)·X + c·(X·X^T)^2·X
            </code>
            <p className="text-xs text-text-muted mt-2">Repeated for {plasticityCfg.ns_steps} iterations per update.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
