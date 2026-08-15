import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { riskHex, formatNumber } from '../../utils/formatters';
import styles from './Charts.module.css';

const RADIAN = Math.PI / 180;

function CustomLabel({ cx, cy, midAngle, innerRadius, outerRadius, value, percent }) {
  if (percent < 0.05) return null;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);
  return (
    <text x={x} y={y} fill="#fff" textAnchor="middle" dominantBaseline="central"
      fontSize={11} fontWeight={700} fontFamily="Inter, sans-serif">
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0];
  return (
    <div className={styles.tooltip}>
      <span className={styles.tooltipDot} style={{ background: d.payload.fill }} />
      <span className={styles.tooltipLabel}>{d.name}</span>
      <span className={styles.tooltipValue}>{formatNumber(d.value)}</span>
    </div>
  );
}

/**
 * Donut chart showing HIGH / MEDIUM / LOW risk breakdown.
 */
export default function RiskDonut({ summary }) {
  const data = [
    { name: 'High Risk',    value: summary.high_risk,   fill: riskHex('HIGH') },
    { name: 'Medium Risk',  value: summary.medium_risk,  fill: riskHex('MEDIUM') },
    { name: 'Low Risk',     value: summary.low_risk,     fill: riskHex('LOW') },
  ].filter(d => d.value > 0);

  const total = data.reduce((sum, d) => sum + d.value, 0);

  return (
    <div className={styles.chartCard} id="risk-donut-chart">
      <div className={styles.chartHeader}>
        <h3 className={styles.chartTitle}>Risk Distribution</h3>
        <span className={styles.chartSub}>By risk level</span>
      </div>
      <div className={styles.chartBody}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={55}
              outerRadius={85}
              paddingAngle={3}
              dataKey="value"
              labelLine={false}
              label={CustomLabel}
              animationBegin={0}
              animationDuration={900}
            >
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.fill} strokeWidth={0} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend
              formatter={(value) => (
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>{value}</span>
              )}
            />
          </PieChart>
        </ResponsiveContainer>

        {/* Center label */}
        <div className={styles.donutCenter}>
          <span className={styles.donutTotal}>{formatNumber(total)}</span>
          <span className={styles.donutLabel}>Total</span>
        </div>
      </div>
    </div>
  );
}
