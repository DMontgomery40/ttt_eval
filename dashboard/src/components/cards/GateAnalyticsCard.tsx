import { useMemo } from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ReferenceLine, LineChart, Line, Legend } from 'recharts';
import { MonitorEvent } from '../../types';
import { formatNumber } from '../../utils/formatting';

interface GateAnalyticsCardProps {
  events: MonitorEvent[];
  minEntropyThreshold?: number;
  minDiversityThreshold?: number;
}

export function GateAnalyticsCard({
  events,
  minEntropyThreshold = 1.0,
  minDiversityThreshold = 0.1,
}: GateAnalyticsCardProps) {
  // Gate rejection reason breakdown
  const rejectionReasons = useMemo(() => {
    const reasons: Record<string, number> = {};
    let totalRejected = 0;

    events.forEach(e => {
      if (!e.gate_allowed || e.update_skipped) {
        totalRejected++;
        e.gate_reasons.forEach(reason => {
          reasons[reason] = (reasons[reason] || 0) + 1;
        });
      }
    });

    const pieData = Object.entries(reasons).map(([name, value]) => ({
      name,
      value,
    }));

    return { pieData, totalRejected };
  }, [events]);

  // Scatter plot data: entropy vs diversity
  const scatterData = useMemo(() => {
    return events.map(e => ({
      entropy: e.token_entropy,
      diversity: e.token_diversity,
      chunk_index: e.chunk_index,
      gate_allowed: e.gate_allowed,
      gate_blocked: e.update_skipped,
      preview: e.chunk_preview,
    }));
  }, [events]);

  // Timeline data
  const timelineData = useMemo(() => {
    return events.map(e => ({
      chunk: e.chunk_index,
      entropy: e.token_entropy,
      diversity: e.token_diversity,
      blocked: e.update_skipped,
    }));
  }, [events]);

  // Summary stats
  const stats = useMemo(() => {
    const blocked = events.filter(e => e.update_skipped);
    const lowEntropy = events.filter(e => e.token_entropy < minEntropyThreshold);
    const lowDiversity = events.filter(e => e.token_diversity < minDiversityThreshold);

    return {
      totalBlocked: blocked.length,
      lowEntropy: lowEntropy.length,
      lowDiversity: lowDiversity.length,
      blockRate: events.length > 0 ? (blocked.length / events.length) * 100 : 0,
    };
  }, [events, minEntropyThreshold, minDiversityThreshold]);

  // Color palette for pie chart
  const COLORS = ['#f85149', '#f0883e', '#d29922', '#58a6ff', '#8b949e'];

  if (events.length === 0) {
    return (
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-2">Gate Analytics</h3>
        <p className="text-xs text-text-muted">No gate decision data available.</p>
      </div>
    );
  }

  return (
    <div className="bg-surface-50 border border-surface-200 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">Gate Analytics</h3>
          <p className="text-xs text-text-muted mt-1">
            Pre-update gate blocks suspicious inputs before they write to weights
          </p>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-surface-100 rounded-lg p-3">
          <div className="text-xs text-text-muted">Total Blocked</div>
          <div className="text-2xl font-bold font-mono text-accent-orange mt-1">
            {stats.totalBlocked}
          </div>
          <div className="text-xs text-text-secondary mt-1">
            {stats.blockRate.toFixed(1)}% of chunks
          </div>
        </div>

        <div className="bg-surface-100 rounded-lg p-3">
          <div className="text-xs text-text-muted">Low Entropy</div>
          <div className="text-2xl font-bold font-mono text-text-secondary mt-1">
            {stats.lowEntropy}
          </div>
          <div className="text-xs text-text-secondary mt-1">
            &lt; {minEntropyThreshold}
          </div>
        </div>

        <div className="bg-surface-100 rounded-lg p-3">
          <div className="text-xs text-text-muted">Low Diversity</div>
          <div className="text-2xl font-bold font-mono text-text-secondary mt-1">
            {stats.lowDiversity}
          </div>
          <div className="text-xs text-text-secondary mt-1">
            &lt; {(minDiversityThreshold * 100).toFixed(0)}%
          </div>
        </div>

        <div className="bg-accent-orange/10 border border-accent-orange/30 rounded-lg p-3">
          <div className="text-xs text-text-muted">Unique Reasons</div>
          <div className="text-2xl font-bold font-mono text-accent-orange mt-1">
            {rejectionReasons.pieData.length}
          </div>
          <div className="text-xs text-text-secondary mt-1">
            Rejection types
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Pie Chart: Rejection Reasons */}
        <div>
          <div className="text-xs font-semibold text-text-primary mb-2">
            Gate Rejection Reasons
          </div>
          {rejectionReasons.pieData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={rejectionReasons.pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={(entry) => `${entry.name} (${entry.value})`}
                    outerRadius={70}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {rejectionReasons.pieData.map((_entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '6px', fontSize: '11px' }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-2 space-y-1">
                {rejectionReasons.pieData.map((item, idx) => (
                  <div key={item.name} className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
                      <span className="text-text-secondary">{item.name}</span>
                    </div>
                    <span className="font-mono text-text-muted">{item.value}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="h-[200px] flex items-center justify-center text-xs text-text-muted">
              No gate rejections detected
            </div>
          )}
        </div>

        {/* Scatter Plot: Entropy vs Diversity */}
        <div>
          <div className="text-xs font-semibold text-text-primary mb-2">
            Token Entropy vs Diversity
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis
                type="number"
                dataKey="entropy"
                name="Entropy"
                stroke="#8b949e"
                tick={{ fill: '#8b949e', fontSize: 10 }}
                label={{ value: 'Token Entropy', position: 'insideBottom', offset: -10, style: { fill: '#8b949e', fontSize: 10 } }}
              />
              <YAxis
                type="number"
                dataKey="diversity"
                name="Diversity"
                stroke="#8b949e"
                tick={{ fill: '#8b949e', fontSize: 10 }}
                label={{ value: 'Token Diversity', angle: -90, position: 'insideLeft', style: { fill: '#8b949e', fontSize: 10 } }}
                domain={[0, 1]}
              />
              <ReferenceLine x={minEntropyThreshold} stroke="#f0883e" strokeDasharray="3 3" />
              <ReferenceLine y={minDiversityThreshold} stroke="#f0883e" strokeDasharray="3 3" />
              <Tooltip
                contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '6px', fontSize: '11px' }}
                formatter={(value: any) => formatNumber(Number(value), 3)}
                content={({ active, payload }) => {
                  if (!active || !payload || payload.length === 0) return null;
                  const data = payload[0].payload;
                  return (
                    <div className="bg-surface-100 border border-surface-200 rounded p-2 text-xs">
                      <div className="font-mono mb-1">Chunk {data.chunk_index}</div>
                      <div className="text-text-muted">Entropy: {formatNumber(data.entropy, 3)}</div>
                      <div className="text-text-muted">Diversity: {formatNumber(data.diversity, 3)}</div>
                      {data.gate_blocked && <div className="text-accent-orange mt-1">⚠ Blocked</div>}
                    </div>
                  );
                }}
              />
              <Scatter
                data={scatterData}
                fill="#58a6ff"
                shape={(props: any) => {
                  const { cx, cy, payload } = props;
                  const color = payload.gate_blocked ? '#f0883e' : '#3fb950';
                  return <circle cx={cx} cy={cy} r={3} fill={color} fillOpacity={0.7} stroke={color} strokeWidth={1} />;
                }}
              />
            </ScatterChart>
          </ResponsiveContainer>
          <div className="flex items-center justify-center gap-4 mt-2 text-xs text-text-muted">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-accent-green" />
              <span>Allowed</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-accent-orange" />
              <span>Blocked</span>
            </div>
          </div>
        </div>
      </div>

      {/* Timeline: Entropy and Diversity */}
      <div>
        <div className="text-xs font-semibold text-text-primary mb-2">
          Entropy & Diversity Timeline
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
              yAxisId="left"
              stroke="#8b949e"
              tick={{ fill: '#8b949e', fontSize: 11 }}
              label={{ value: 'Entropy', angle: -90, position: 'insideLeft', style: { fill: '#8b949e', fontSize: 11 } }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#8b949e"
              tick={{ fill: '#8b949e', fontSize: 11 }}
              label={{ value: 'Diversity', angle: 90, position: 'insideRight', style: { fill: '#8b949e', fontSize: 11 } }}
              domain={[0, 1]}
            />
            <ReferenceLine yAxisId="left" y={minEntropyThreshold} stroke="#f0883e" strokeDasharray="3 3" />
            <ReferenceLine yAxisId="right" y={minDiversityThreshold} stroke="#d29922" strokeDasharray="3 3" />
            <Tooltip
              contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '6px', fontSize: '11px' }}
              formatter={(value: any) => formatNumber(Number(value), 3)}
            />
            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="entropy"
              stroke="#58a6ff"
              strokeWidth={2}
              dot={(props: any) => {
                const { cx, cy, payload } = props;
                if (payload.blocked) {
                  return <circle cx={cx} cy={cy} r={4} fill="#f0883e" stroke="#f0883e" strokeWidth={2} />;
                }
                return <circle cx={cx} cy={cy} r={2} fill="#58a6ff" />;
              }}
              name="Entropy"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="diversity"
              stroke="#a371f7"
              strokeWidth={2}
              dot={{ r: 2, fill: '#a371f7' }}
              name="Diversity"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
