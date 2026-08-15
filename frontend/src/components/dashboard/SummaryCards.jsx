import { formatCurrency, formatNumber } from '../../utils/formatters';
import styles from './SummaryCards.module.css';

const CARDS = (summary) => [
  {
    id: 'total-invoices',
    label: 'Total Invoices',
    value: formatNumber(summary.total_invoices),
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M9 12h6m-6 4h6M7 8h10M5 4h14a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
    colorKey: 'info',
    sub: 'Records analyzed',
  },
  {
    id: 'leakage-detected',
    label: 'Leakage Detected',
    value: formatNumber(summary.leakage_detected),
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M12 9v4M12 17h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    colorKey: 'danger',
    sub: 'Confirmed revenue leakage',
    pct: summary.total_invoices > 0
      ? `${((summary.leakage_detected / summary.total_invoices) * 100).toFixed(1)}%`
      : null,
  },
  {
    id: 'high-risk',
    label: 'High Risk',
    value: formatNumber(summary.high_risk),
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5" />
        <path d="M12 8v4M12 15v1" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </svg>
    ),
    colorKey: 'warning',
    sub: 'Immediate attention needed',
  },
  {
    id: 'total-leakage',
    label: 'Total Leakage Amount',
    value: formatCurrency(summary.total_potential_leakage, 0),
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    colorKey: 'danger',
    sub: 'Potential revenue at risk',
    large: true,
  },
];

/**
 * 4-card KPI summary row shown at the top of the dashboard.
 */
export default function SummaryCards({ summary }) {
  return (
    <div className={styles.grid} role="list">
      {CARDS(summary).map((card, i) => (
        <article
          key={card.id}
          id={card.id}
          className={`${styles.card} ${styles[card.colorKey]}`}
          style={{ animationDelay: `${i * 0.08}s` }}
          role="listitem"
        >
          <div className={styles.top}>
            <div className={styles.iconWrap}>{card.icon}</div>
            {card.pct && <span className={styles.pctBadge}>{card.pct}</span>}
          </div>
          <div className={styles.value} aria-label={`${card.label}: ${card.value}`}>
            {card.value}
          </div>
          <div className={styles.label}>{card.label}</div>
          <div className={styles.sub}>{card.sub}</div>
        </article>
      ))}
    </div>
  );
}
