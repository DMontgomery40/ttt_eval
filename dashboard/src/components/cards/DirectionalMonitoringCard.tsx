import { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, ReferenceLine, Legend } from 'recharts';
import { MonitorEvent } from '../../types';
import { formatNumber } from '../../utils/formatting';

interface DirectionalMonitoringCardProps {
  events: MonitorEvent[];
}

export function DirectionalMonitoringCard({ events }: DirectionalMonitoringCardProps) {
  // Filter events that have canary gradient alignment data
  const eventsWithAlignment = useMemo(() => {
    return events.filter(e => e.grad_canary_cos !== null && e.grad_canary_cos !== undefined);
  }, [events]);

  // Prepare scatter plot data
  const scatterData = useMemo(() => {
    return eventsWithAlignment.map(e => ({
      grad_norm: e.grad_norm,
      cos: e.grad_canary_cos!,
      chunk_index: e.chunk_index,
      preview: e.chunk_preview,
      canary_grad_norm: e.canary_grad_norm || 0,
      gate_blocked: e.update_skipped,
      rollback: e.rollback_triggered,
    }));
  }, [eventsWithAlignment]);

  // Prepare timeline data
  const timelineData = useMemo(() => {
    return eventsWithAlignment.map(e => ({
      chunk: e.chunk_index,
      cos: e.grad_canary_cos!,
      dot: e.grad_canary_dot || 0,
      rollback: e.rollback_triggered,
    }));
  }, [eventsWithAlignment]);

  // Calculate summary stats
  const stats: {
    harmful: number;
    benign: number;
    neutral: number;
    maxHarmful: number | null;
    maxHarmfulChunk: MonitorEvent | null;
  } = useMemo(() => {
    if (eventsWithAlignment.length === 0) {
      return { harmful: 0, benign: 0, neutral: 0, maxHarmful: null, maxHarmfulChunk: null };
    }

    let harmful = 0;
    let benign = 0;
    let neutral = 0;
    let maxCos = -2;
    let maxChunk: MonitorEvent | null = null;

    eventsWithAlignment.forEach(e => {
      const cos = e.grad_canary_cos!;
      if (cos > 0.3) {
        harmful++;
        if (cos > maxCos) {
          maxCos = cos;
          maxChunk = e;
        }
      } else if (cos < -0.3) {
        benign++;
      } else {
        neutral++;
      }
    });

    return {
      harmful,
      benign,
      neutral,
      maxHarmful: maxChunk ? maxCos : null,
      maxHarmfulChunk: maxChunk,
    };
  }, [eventsWithAlignment]);

  // Color function for scatter points
  const getPointColor = (cos: number, blocked: boolean, rollback: boolean) => {
    if (rollback) return '#f85149'; // red - rollback
    if (blocked) return '#f0883e'; // orange - blocked
    if (cos > 0.3) return '#d29922'; // yellow/gold - harmful alignment
    if (cos < -0.3) return '#3fb950'; // green - benign
    return '#848d97'; // gray - neutral
  };

  if (eventsWithAlignment.length === 0) {
    return (
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-2">Directional Monitoring</h3>
        <p className="text-xs text-text-muted">No canary gradient alignment data available. Enable with --enable_canary_grad flag.</p>
      </div>
    );
  }

  return (
    <div className="bg-surface-50 border border-surface-200 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">Directional Monitoring</h3>
          <p className="text-xs text-text-muted mt-1">
            Canary gradient alignment detects harmful update directions
          </p>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-surface-100 rounded-lg p-3">
          <div className="text-xs text-text-muted">Harmful (cos &gt; 0.3)</div>
          <div className="text-2xl font-bold font-mono text-accent-red mt-1">
            {stats.harmful}
          </div>
          <div className="text-xs text-text-secondary mt-1">
            {events.length > 0 ? ((stats.harmful / eventsWithAlignment.length) * 100).toFixed(1) : 0}% of chunks
          </div>
        </div>

        <div className="bg-surface-100 rounded-lg p-3">
          <div className="text-xs text-text-muted">Benign (cos &lt; -0.3)</div>
          <div className="text-2xl font-bold font-mono text-accent-green mt-1">
            {stats.benign}
          </div>
          <div className="text-xs text-text-secondary mt-1">
            {eventsWithAlignment.length > 0 ? ((stats.benign / eventsWithAlignment.length) * 100).toFixed(1) : 0}% of chunks
          </div>
        </div>

        <div className="bg-surface-100 rounded-lg p-3">
          <div className="text-xs text-text-muted">Neutral (|cos| &lt; 0.3)</div>
          <div className="text-2xl font-bold font-mono text-text-secondary mt-1">
            {stats.neutral}
          </div>
          <div className="text-xs text-text-secondary mt-1">
            {eventsWithAlignment.length > 0 ? ((stats.neutral / eventsWithAlignment.length) * 100).toFixed(1) : 0}% of chunks
          </div>
        </div>

        <div className="bg-accent-red/10 border border-accent-red/30 rounded-lg p-3">
          <div className="text-xs text-text-muted">Max Harmful Alignment</div>
          <div className="text-2xl font-bold font-mono text-accent-red mt-1">
            {stats.maxHarmful !== null ? formatNumber(stats.maxHarmful, 3) : '—'}
          </div>
          <div className="text-xs text-text-secondary mt-1">
            {stats.maxHarmfulChunk ? `chunk ${stats.maxHarmfulChunk.chunk_index}` : '—'}
          </div>
        </div>
      </div>

      {/* Scatter Plot: grad_norm vs canary_cos */}
      <div>
        <div className="text-xs font-semibold text-text-primary mb-2">
          Gradient Magnitude vs Canary Alignment
        </div>
        <ResponsiveContainer width="100%" height={250}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis
              type="number"
              dataKey="grad_norm"
              name="Grad Norm"
              stroke="#8b949e"
              tick={{ fill: '#8b949e', fontSize: 11 }}
              label={{ value: 'Gradient Norm', position: 'insideBottom', offset: -10, style: { fill: '#8b949e', fontSize: 11 } }}
            />
            <YAxis
              type="number"
              dataKey="cos"
              name="Canary Cos"
              stroke="#8b949e"
              tick={{ fill: '#8b949e', fontSize: 11 }}
              label={{ value: 'Canary Alignment (cos)', angle: -90, position: 'insideLeft', style: { fill: '#8b949e', fontSize: 11 } }}
              domain={[-1, 1]}
            />
            <ReferenceLine y={0.3} stroke="#d29922" strokeDasharray="3 3" />
            <ReferenceLine y={-0.3} stroke="#3fb950" strokeDasharray="3 3" />
            <ReferenceLine y={0} stroke="#484f58" strokeDasharray="2 2" />
            <Tooltip
              contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '6px', fontSize: '11px' }}
              labelStyle={{ color: '#c9d1d9' }}
              formatter={(value: any, name: string) => {
                if (name === 'Grad Norm' || name === 'Canary Cos') {
                  return formatNumber(Number(value), 3);
                }
                return value;
              }}
              content={({ active, payload }) => {
                if (!active || !payload || payload.length === 0) return null;
                const data = payload[0].payload;
                return (
                  <div className="bg-surface-100 border border-surface-200 rounded p-2 text-xs">
                    <div className="font-mono mb-1">Chunk {data.chunk_index}</div>
                    <div className="text-text-muted">Grad: {formatNumber(data.grad_norm, 3)}</div>
                    <div className="text-text-muted">Cos: {formatNumber(data.cos, 3)}</div>
                    {data.gate_blocked && <div className="text-accent-orange mt-1">⚠ Blocked</div>}
                    {data.rollback && <div className="text-accent-red mt-1">↶ Rollback</div>}
                    <div className="text-text-secondary mt-1 max-w-xs truncate">{data.preview}</div>
                  </div>
                );
              }}
            />
            <Scatter
              data={scatterData}
              fill="#58a6ff"
              shape={(props: any) => {
                const { cx, cy, payload } = props;
                const color = getPointColor(payload.cos, payload.gate_blocked, payload.rollback);
                const size = Math.min(8, Math.max(3, Math.sqrt(payload.canary_grad_norm) * 2));
                return <circle cx={cx} cy={cy} r={size} fill={color} fillOpacity={0.7} stroke={color} strokeWidth={1} />;
              }}
            />
          </ScatterChart>
        </ResponsiveContainer>
        <div className="flex items-center justify-center gap-4 mt-2 text-xs text-text-muted">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-accent-green" />
            <span>Benign (&lt; -0.3)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-text-muted" />
            <span>Neutral</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-accent-gold" />
            <span>Harmful (&gt; 0.3)</span>
          </div>
        </div>
      </div>

      {/* Timeline: Canary Alignment over chunks */}
      <div>
        <div className="text-xs font-semibold text-text-primary mb-2">
          Canary Alignment Timeline
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={timelineData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis
              dataKey="chunk"
              stroke="#8b949e"
              tick={{ fill: '#8b949e', fontSize: 11 }}
              label={{ value: 'Chunk Index', position: 'insideBottom', offset: -10, style: { fill: '#8b949e', fontSize: 11 } }}
            />
            <YAxis
              stroke="#8b949e"
              tick={{ fill: '#8b949e', fontSize: 11 }}
              domain={[-1, 1]}
              label={{ value: 'Cosine Similarity', angle: -90, position: 'insideLeft', style: { fill: '#8b949e', fontSize: 11 } }}
            />
            <ReferenceLine y={0.3} stroke="#d29922" strokeDasharray="3 3" label={{ value: 'harmful', position: 'right', fill: '#d29922', fontSize: 10 }} />
            <ReferenceLine y={-0.3} stroke="#3fb950" strokeDasharray="3 3" label={{ value: 'benign', position: 'right', fill: '#3fb950', fontSize: 10 }} />
            <ReferenceLine y={0} stroke="#484f58" strokeDasharray="2 2" />
            <Tooltip
              contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '6px', fontSize: '11px' }}
              formatter={(value: any) => formatNumber(Number(value), 3)}
            />
            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
            <Line
              type="monotone"
              dataKey="cos"
              stroke="#58a6ff"
              strokeWidth={2}
              dot={(props: any) => {
                const { cx, cy, payload } = props;
                if (payload.rollback) {
                  return <circle cx={cx} cy={cy} r={4} fill="#f85149" stroke="#f85149" strokeWidth={2} />;
                }
                return <circle cx={cx} cy={cy} r={2} fill="#58a6ff" />;
              }}
              name="Canary Alignment"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Worst offender */}
      {stats.maxHarmfulChunk && (
        <div className="bg-accent-red/10 border border-accent-red/30 rounded-lg p-3">
          <div className="text-xs font-semibold text-accent-red mb-2">
            🚨 Highest Harmful Alignment Detected
          </div>
          <div className="font-mono text-xs text-text-secondary mb-1">
            Chunk {stats.maxHarmfulChunk.chunk_index}: cos = {formatNumber(stats.maxHarmful!, 3)}
          </div>
          <div className="text-xs text-text-secondary">
            {stats.maxHarmfulChunk.chunk_preview}
          </div>
        </div>
      )}
    </div>
  );
}
