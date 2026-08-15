import styles from './Header.module.css';

/**
 * App header with logo and branding.
 */
export default function Header({ onReset, showReset = false }) {
  return (
    <header className={styles.header}>
      <div className={`container ${styles.inner}`}>
        {/* Logo mark */}
        <button
          className={styles.logo}
          onClick={onReset}
          aria-label="Go to home — AI Revenue Leakage Detector"
          type="button"
        >
          <div className={styles.logoIcon} aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L2 7l10 5 10-5-10-5Z" stroke="var(--accent-light)" strokeWidth="1.5" strokeLinejoin="round" />
              <path d="M2 17l10 5 10-5" stroke="var(--accent-light)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M2 12l10 5 10-5" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <div className={styles.logoText}>
            <span className={styles.logoTitle}>Revenue Leakage AI</span>
            <span className={styles.logoSub}>Powered by XGBoost + Business Rules</span>
          </div>
        </button>

        <div className={styles.right}>
          {/* Status dot */}
          <div className={styles.statusDot} title="System online">
            <span className={styles.pulse} />
            <span>Live</span>
          </div>

          {showReset && (
            <button className={styles.resetBtn} onClick={onReset} type="button">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M3 12a9 9 0 1 0 2.6-6.4M3 3v5h5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Analyze Another
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
