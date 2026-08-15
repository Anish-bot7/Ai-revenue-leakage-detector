import styles from './ErrorAlert.module.css';

/**
 * Error state display with retry button.
 */
export default function ErrorAlert({ message, onRetry }) {
  return (
    <div className={styles.wrapper} role="alert" aria-live="assertive">
      <div className={styles.iconWrap}>
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="10" stroke="var(--danger)" strokeWidth="1.5" />
          <path d="M12 7v5M12 16v1" stroke="var(--danger)" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </div>
      <div className={styles.content}>
        <h3 className={styles.title}>Analysis Failed</h3>
        <p className={styles.message}>{message}</p>
      </div>
      {onRetry && (
        <button className={styles.retryBtn} onClick={onRetry} type="button">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M3 12a9 9 0 1 0 2.6-6.4M3 3v5h5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Try Again
        </button>
      )}
    </div>
  );
}
