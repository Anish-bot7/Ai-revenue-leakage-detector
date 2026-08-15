import styles from './LoadingSpinner.module.css';

/**
 * Centered animated loading spinner with optional message.
 */
export default function LoadingSpinner({ message = 'Analyzing your data…', progress = null }) {
  return (
    <div className={styles.wrapper} role="status" aria-live="polite">
      <div className={styles.orbContainer}>
        <div className={styles.orb1} />
        <div className={styles.orb2} />
        <div className={styles.orb3} />
        <div className={styles.core}>
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
            <path d="M14 2a12 12 0 1 1 0 24A12 12 0 0 1 14 2Z" stroke="var(--accent)" strokeWidth="2" strokeDasharray="60" strokeDashoffset="20" strokeLinecap="round" />
          </svg>
        </div>
      </div>

      <h3 className={styles.title}>Analyzing</h3>
      <p className={styles.message}>{message}</p>

      {progress !== null && (
        <div className={styles.progressBar} role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
          <div className={styles.progressFill} style={{ width: `${progress}%` }} />
          <span className={styles.progressLabel}>{progress}%</span>
        </div>
      )}

      <div className={styles.steps}>
        {['Uploading file', 'Preprocessing data', 'Running ML model', 'Generating report'].map((step, i) => (
          <div key={step} className={styles.step} style={{ animationDelay: `${i * 0.4}s` }}>
            <span className={styles.stepDot} />
            <span>{step}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
