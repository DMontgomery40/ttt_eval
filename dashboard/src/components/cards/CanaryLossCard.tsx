import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend, BarChart, Bar } from 'recharts';
import { MonitorEvent } from '../../types';
import { formatNumber } from '../../utils/formatting';

interface CanaryLossCardProps {
  events: MonitorEvent[];
  rollbackAbsCanaryDelta?: number;
}

export function CanaryLossCard({ events, rollbackAbsCanaryDelta = 1.0 }: CanaryLossCardProps) {
  // Filter events with canary loss data
  const eventsWithCanary = useMemo(() => {
    return events.filter(e =>
      e.canary_loss_before !== null &&
      e.canary_loss_before !== undefined
    );
  }, [events]);

  // Prepare timeline data
  const timelineData = useMemo(() => {
    return eventsWithCanary.map(e => ({
      chunk: e.chunk_index,
      before: e.canary_loss_before!,
      after: e.canary_loss_after !== null ? e.canary_loss_after : e.canary_loss_before!,
      delta: e.canary_delta || 0,
      rollback: e.rollback_triggered,
    }));
  }, [eventsWithCanary]);

  // Prepare histogram data for canary delta distribution
  const histogramData = useMemo(() => {
    const deltas = eventsWithCanary
      .filter(e => e.canary_delta !== null)
      .map(e => e.canary_delta!);

    if (deltas.length === 0) return [];

    // Create bins from min to max
    const min = Math.min(...deltas, -0.5);
    const max = Math.max(...deltas, 2.0);
    const binCount = 20;
    const binSize = (max - min) / binCount;

    const bins: { bin: string; count: number; binStart: number }[] = [];
    for (let i = 0; i < binCount; i++) {
      const binStart = min + i * binSize;
      const binEnd = binStart + binSize;
      const count = deltas.filter(d => d >= binStart && d < binEnd).length;
      bins.push({
        bin: `${binStart.toFixed(2)}`,
        binStart,
        count,
      });
    }

    return bins;
  }, [eventsWithCanary]);

  // Calculate summary stats
  const stats = useMemo(() => {
    if (eventsWithCanary.length === 0) {
      return {
        totalRollbacks: 0,
        avgDelta: 0,
        maxDelta: 0,
        maxDeltaChunk: null,
        corruptionRate: 0,
      };
    }

    const rollbacks = eventsWithCanary.filter(e => e.rollback_triggered);
    const deltas = eventsWithCanary
      .filter(e => e.canary_delta !== null)
      .map(e => e.canary_delta!);

    const avgDelta = deltas.length > 0
      ? deltas.reduce((sum, d) => sum + d, 0) / deltas.length
      : 0;

    const maxDelta = deltas.length > 0 ? Math.max(...deltas) : 0;
    const maxDeltaEvent = eventsWithCanary.find(e => e.canary_delta === maxDelta);

    const corruptionRate = eventsWithCanary.length > 0
      ? (deltas.filter(d => d > rollbackAbsCanaryDelta).length / eventsWithCanary.length) * 100
      : 0;

    return {
      totalRollbacks: rollbacks.length,
      avgDelta,
      maxDelta,
      maxDeltaChunk: maxDeltaEvent,
      corruptionRate,
    };
  }, [eventsWithCanary, rollbackAbsCanaryDelta]);

  if (eventsWithCanary.length === 0) {
    return (
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-2">Canary Loss Tracking</h3>
        <p className="text-xs text-text-muted">No canary loss data available. Enable with --enable_rollback flag.</p>
      </div>
    );
  }

  return (
    <div className="bg-surface-50 border border-surface-200 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">Canary Loss Tracking</h3>
          <p className="text-xs text-text-muted mt-1">
            Measures model corruption on fixed reference text after updates
          </p>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-surface-100 rounded-lg p-3">
          <div className="text-xs text-text-muted">Total Rollbacks</div>
          <div className="text-2xl font-bold font-mono text-accent-orange mt-1">
            {stats.totalRollbacks}
          </div>
          <div className="text-xs text-text-secondary mt-1">
            {eventsWithCanary.length > 0
              ? ((stats.totalRollbacks / eventsWithCanary.length) * 100).toFixed(1)
              : 0}% of chunks
          </div>
        </div>

        <div className="bg-surface-100 rounded-lg p-3">
          <div className="text-xs text-text-muted">Avg Canary Δ</div>
          <div className="text-2xl font-bold font-mono text-text-secondary mt-1">
            {formatNumber(stats.avgDelta, 3)}
          </div>
          <div className="text-xs text-text-secondary mt-1">
            Mean corruption
          </div>
        </div>

        <div className="bg-surface-100 rounded-lg p-3">
          <div className="text-xs text-text-muted">Max Canary Δ</div>
          <div className="text-2xl font-bold font-mono text-accent-red mt-1">
            {formatNumber(stats.maxDelta, 3)}
          </div>
          <div className="text-xs text-text-secondary mt-1">
            {stats.maxDeltaChunk ? `chunk ${stats.maxDeltaChunk.chunk_index}` : '—'}
          </div>
        </div>

        <div className="bg-accent-orange/10 border border-accent-orange/30 rounded-lg p-3">
          <div className="text-xs text-text-muted">Corruption Rate</div>
          <div className="text-2xl font-bold font-mono text-accent-orange mt-1">
            {stats.corruptionRate.toFixed(1)}%
          </div>
          <div className="text-xs text-text-secondary mt-1">
            Δ &gt; {rollbackAbsCanaryDelta}
          </div>
        </div>
      </div>

      {/* Dual-line chart: before vs after */}
      <div>
        <div className="text-xs font-semibold text-text-primary mb-2">
          Canary Loss Before/After Updates
        </div>
        <ResponsiveContainer width="100%" height={220}>
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
              label={{ value: 'Canary Loss', angle: -90, position: 'insideLeft', style: { fill: '#8b949e', fontSize: 11 } }}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '6px', fontSize: '11px' }}
              formatter={(value: any) => formatNumber(Number(value), 3)}
            />
            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
            <Line
              type="monotone"
              dataKey="before"
              stroke="#58a6ff"
              strokeWidth={2}
              dot={{ r: 2, fill: '#58a6ff' }}
              name="Before Update"
            />
            <Line
              type="monotone"
              dataKey="after"
              stroke="#3fb950"
              strokeWidth={2}
              dot={(props: any) => {
                const { cx, cy, payload } = props;
                if (payload.rollback) {
                  return <circle cx={cx} cy={cy} r={5} fill="#f85149" stroke="#f85149" strokeWidth={2} />;
                }
                return <circle cx={cx} cy={cy} r={2} fill="#3fb950" />;
              }}
              name="After Update"
            />
          </LineChart>
        </ResponsiveContainer>
        <div className="flex items-center justify-center gap-4 mt-2 text-xs text-text-muted">
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-accent-blue" />
            <span>Before</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-accent-green" />
            <span>After (committed)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-accent-red" />
            <span>Rollback triggered</span>
          </div>
        </div>
      </div>

      {/* Canary Delta Distribution Histogram */}
      <div>
        <div className="text-xs font-semibold text-text-primary mb-2">
          Canary Delta Distribution
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={histogramData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis
              dataKey="bin"
              stroke="#8b949e"
              tick={{ fill: '#8b949e', fontSize: 10 }}
              label={{ value: 'Canary Delta', position: 'insideBottom', offset: -10, style: { fill: '#8b949e', fontSize: 11 } }}
              interval={4}
            />
            <YAxis
              stroke="#8b949e"
              tick={{ fill: '#8b949e', fontSize: 11 }}
              label={{ value: 'Count', angle: -90, position: 'insideLeft', style: { fill: '#8b949e', fontSize: 11 } }}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '6px', fontSize: '11px' }}
              formatter={(value: any) => value}
              labelFormatter={(label) => `Delta: ${label}`}
            />
            <ReferenceLine
              x={rollbackAbsCanaryDelta.toFixed(2)}
              stroke="#f0883e"
              strokeDasharray="3 3"
              label={{ value: 'rollback threshold', position: 'top', fill: '#f0883e', fontSize: 10 }}
            />
            <Bar
              dataKey="count"
              fill="#58a6ff"
              shape={(props: any) => {
                const { x, y, width, height, payload } = props;
                const isBelowThreshold = payload.binStart < rollbackAbsCanaryDelta;
                const fill = isBelowThreshold ? '#3fb950' : '#f0883e';
                return <rect x={x} y={y} width={width} height={height} fill={fill} fillOpacity={0.8} />;
              }}
            />
          </BarChart>
        </ResponsiveContainer>
        <div className="flex items-center justify-center gap-4 mt-2 text-xs text-text-muted">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-accent-green" />
            <span>Safe (Δ &lt; {rollbackAbsCanaryDelta})</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-accent-orange" />
            <span>Corrupt (Δ ≥ {rollbackAbsCanaryDelta})</span>
          </div>
        </div>
      </div>

      {/* Worst corruption event */}
      {stats.maxDeltaChunk && stats.maxDelta > rollbackAbsCanaryDelta && (
        <div className="bg-accent-red/10 border border-accent-red/30 rounded-lg p-3">
          <div className="text-xs font-semibold text-accent-red mb-2">
            🔥 Maximum Corruption Detected
          </div>
          <div className="font-mono text-xs text-text-secondary mb-1">
            Chunk {stats.maxDeltaChunk.chunk_index}: Δ = {formatNumber(stats.maxDelta, 3)}
            {stats.maxDeltaChunk.rollback_triggered && ' (rolled back)'}
          </div>
          <div className="text-xs text-text-secondary">
            {stats.maxDeltaChunk.chunk_preview}
          </div>
        </div>
      )}
    </div>
  );
}
