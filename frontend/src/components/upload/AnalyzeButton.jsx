import styles from './AnalyzeButton.module.css';

/**
 * Primary CTA button for triggering analysis.
 */
export default function AnalyzeButton({ onClick, disabled, isLoading }) {
  return (
    <button
      id="analyze-btn"
      className={`${styles.btn} ${isLoading ? styles.loading : ''}`}
      onClick={onClick}
      disabled={disabled || isLoading}
      type="button"
      aria-label="Analyze file for revenue leakage"
      aria-busy={isLoading}
    >
      {/* Glow effect */}
      <span className={styles.glow} aria-hidden="true" />

      {isLoading ? (
        <>
          <span className={styles.spinner} aria-hidden="true" />
          <span>Analyzing…</span>
        </>
      ) : (
        <>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M9.663 17h4.673M12 3v1m6.364 1.636-.707.707M21 12h-1M4 12H3m3.343-5.657-.707-.707m2.828 9.9a5 5 0 1 1 7.072 0l-.548.547A3.374 3.374 0 0 0 14 18.469V19a2 2 0 0 1-4 0v-.531a3.374 3.374 0 0 0-.988-2.386l-.548-.547Z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span>Detect Revenue Leakage</span>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true" className={styles.arrow}>
            <path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </>
      )}
    </button>
  );
}
