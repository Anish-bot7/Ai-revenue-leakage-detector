import styles from './FilterBar.module.css';

const RISK_OPTIONS    = ['ALL', 'HIGH', 'MEDIUM', 'LOW'];
const PREDICT_OPTIONS = [
  { value: 'ALL',                             label: 'All Predictions' },
  { value: 'Revenue Leakage Detected',        label: 'Leakage Detected' },
  { value: 'Potential Revenue Leakage',       label: 'Potential Leakage' },
  { value: 'Potential Anomaly - Review',      label: 'Needs Review' },
  { value: 'No Revenue Leakage',              label: 'Clean' },
];

/**
 * Filter bar for the results table.
 */
export default function FilterBar({ filters, onChange, totalVisible, totalAll }) {
  return (
    <div className={styles.bar}>
      <div className={styles.left}>
        {/* Risk filter */}
        <div className={styles.filterGroup}>
          <span className={styles.filterLabel}>Risk</span>
          <div className={styles.toggles} role="group" aria-label="Filter by risk level">
            {RISK_OPTIONS.map(opt => (
              <button
                key={opt}
                id={`filter-risk-${opt.toLowerCase()}`}
                className={`${styles.toggle} ${filters.risk === opt ? styles.active : ''} ${opt !== 'ALL' ? styles[opt.toLowerCase()] : ''}`}
                onClick={() => onChange({ ...filters, risk: opt })}
                type="button"
                aria-pressed={filters.risk === opt}
              >
                {opt}
              </button>
            ))}
          </div>
        </div>

        {/* Prediction filter */}
        <div className={styles.filterGroup}>
          <span className={styles.filterLabel}>Type</span>
          <select
            id="filter-prediction"
            className={styles.select}
            value={filters.prediction}
            onChange={e => onChange({ ...filters, prediction: e.target.value })}
            aria-label="Filter by prediction type"
          >
            {PREDICT_OPTIONS.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Search */}
      <div className={styles.right}>
        <div className={styles.searchWrap}>
          <svg className={styles.searchIcon} width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2" />
            <path d="M21 21l-4.35-4.35" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          <input
            id="filter-search"
            type="search"
            className={styles.search}
            placeholder="Search invoices…"
            value={filters.search}
            onChange={e => onChange({ ...filters, search: e.target.value })}
            aria-label="Search invoices"
          />
        </div>

        <span className={styles.count}>
          Showing <strong>{totalVisible}</strong> of <strong>{totalAll}</strong>
        </span>
      </div>
    </div>
  );
}
