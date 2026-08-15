import styles from './ExportButton.module.css';
import { exportToCsv } from '../../utils/exportCsv';

/**
 * Button to export current (filtered) results as CSV.
 */
export default function ExportButton({ results, filename }) {
  const handleExport = () => {
    if (!results?.length) return;
    exportToCsv(results, filename ?? 'revenue-leakage-results');
  };

  return (
    <button
      id="export-csv-btn"
      className={styles.btn}
      onClick={handleExport}
      disabled={!results?.length}
      type="button"
      aria-label={`Export ${results?.length ?? 0} results as CSV`}
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      Export CSV
      {results?.length > 0 && (
        <span className={styles.count}>{results.length}</span>
      )}
    </button>
  );
}
