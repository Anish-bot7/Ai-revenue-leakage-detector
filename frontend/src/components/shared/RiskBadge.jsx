import styles from './RiskBadge.module.css';

const CONFIG = {
  HIGH:   { label: 'HIGH',   cls: styles.high   },
  MEDIUM: { label: 'MEDIUM', cls: styles.medium  },
  LOW:    { label: 'LOW',    cls: styles.low     },
};

/**
 * Colored badge for risk level: HIGH / MEDIUM / LOW
 */
export default function RiskBadge({ level }) {
  const cfg = CONFIG[level] ?? { label: level, cls: styles.low };
  return (
    <span className={`${styles.badge} ${cfg.cls}`} aria-label={`Risk: ${cfg.label}`}>
      <span className={styles.dot} />
      {cfg.label}
    </span>
  );
}
