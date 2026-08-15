import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { predictionHex, labelPrediction, formatNumber } from '../../utils/formatters';
import styles from './Charts.module.css';

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className={styles.tooltip}>
      <span className={styles.tooltipLabel}>{label}</span>
      <span className={styles.tooltipValue}>{formatNumber(payload[0].value)} invoices</span>
    </div>
  );
}

/**
 * Horizontal bar chart showing the 4 prediction categories.
 */
export default function PredictionBreakdown({ results }) {
  // Count predictions
  const counts = {};
  results.forEach(r => {
    const key = r.Prediction ?? 'Unknown';
    counts[key] = (counts[key] ?? 0) + 1;
  });

  const data = Object.entries(counts)
    .map(([prediction, count]) => ({
      name: labelPrediction(prediction),
      value: count,
      fill: predictionHex(prediction),
    }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className={styles.chartCard} id="prediction-breakdown-chart">
      <div className={styles.chartHeader}>
        <h3 className={styles.chartTitle}>Prediction Breakdown</h3>
        <span className={styles.chartSub}>By category</span>
      </div>
      <div className={styles.chartBody}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
            <XAxis type="number" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis
              type="category"
              dataKey="name"
              width={110}
              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={22} animationDuration={800}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
