import { useState, useMemo } from 'react';
import RiskBadge from '../shared/RiskBadge';
import FilterBar from './FilterBar';
import ExportButton from './ExportButton';
import { formatCurrency, formatDecimal, labelPrediction } from '../../utils/formatters';
import styles from './ResultsTable.module.css';

const PAGE_SIZE = 50;

const PREDICTION_COLOR = {
  'Revenue Leakage Detected':   'danger',
  'Potential Revenue Leakage':  'warning',
  'Potential Anomaly - Review': 'info',
  'No Revenue Leakage':         'success',
};

/**
 * Filterable, sortable, paginated data table showing per-invoice predictions.
 */
export default function ResultsTable({ results, filename }) {
  const [filters, setFilters] = useState({ risk: 'ALL', prediction: 'ALL', search: '' });
  const [sort, setSort]       = useState({ key: 'Risk_Level', dir: 'desc' });
  const [page, setPage]       = useState(1);
  const [expanded, setExpanded] = useState(null);

  // ---- Filtering ----
  const filtered = useMemo(() => {
    let rows = results;

    if (filters.risk !== 'ALL') {
      rows = rows.filter(r => r.Risk_Level === filters.risk);
    }
    if (filters.prediction !== 'ALL') {
      rows = rows.filter(r => r.Prediction === filters.prediction);
    }
    if (filters.search.trim()) {
      const q = filters.search.trim().toLowerCase();
      rows = rows.filter(r =>
        Object.values(r).some(v => String(v ?? '').toLowerCase().includes(q))
      );
    }

    return rows;
  }, [results, filters]);

  // ---- Sorting ----
  const RISK_ORDER = { HIGH: 0, MEDIUM: 1, LOW: 2 };
  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const { key, dir } = sort;
      let va = a[key] ?? '';
      let vb = b[key] ?? '';
      if (key === 'Risk_Level') {
        va = RISK_ORDER[va] ?? 3;
        vb = RISK_ORDER[vb] ?? 3;
      }
      if (typeof va === 'number' && typeof vb === 'number') {
        return dir === 'asc' ? va - vb : vb - va;
      }
      return dir === 'asc'
        ? String(va).localeCompare(String(vb))
        : String(vb).localeCompare(String(va));
    });
  }, [filtered, sort]);

  // ---- Pagination ----
  const totalPages  = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const pageRows    = sorted.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const handleSort = (key) => {
    setSort(prev =>
      prev.key === key
        ? { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' }
        : { key, dir: 'desc' }
    );
    setPage(1);
  };

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
    setPage(1);
  };

  const SortIcon = ({ col }) => {
    if (sort.key !== col) return <span className={styles.sortIconInactive}>↕</span>;
    return <span className={styles.sortIconActive}>{sort.dir === 'asc' ? '↑' : '↓'}</span>;
  };

  // Determine which columns actually exist in data
  const hasBilledAmount  = results.some(r => r.Billed_Amount != null);
  const hasExpected      = results.some(r => r.Expected_Amount != null);

  return (
    <section className={styles.section} id="results-table-section" aria-label="Prediction results table">
      {/* Header */}
      <div className={styles.header}>
        <div>
          <h2 className={styles.title}>Invoice Results</h2>
          <p className={styles.sub}>Detailed prediction per invoice record</p>
        </div>
        <div className={styles.headerActions}>
          <ExportButton results={sorted} filename={filename?.replace(/\.[^.]+$/, '') ?? 'results'} />
        </div>
      </div>

      {/* Filter bar */}
      <FilterBar
        filters={filters}
        onChange={handleFilterChange}
        totalVisible={sorted.length}
        totalAll={results.length}
      />

      {/* Table */}
      <div className={styles.tableWrap}>
        <table className={styles.table} aria-label="Revenue leakage predictions">
          <thead>
            <tr>
              <th className={styles.th} onClick={() => handleSort('Risk_Level')} aria-sort={sort.key === 'Risk_Level' ? sort.dir : 'none'}>
                Risk <SortIcon col="Risk_Level" />
              </th>
              <th className={styles.th}>Prediction</th>
              <th className={styles.th} onClick={() => handleSort('ML_Anomaly_Score')} aria-sort={sort.key === 'ML_Anomaly_Score' ? sort.dir : 'none'}>
                ML Score <SortIcon col="ML_Anomaly_Score" />
              </th>
              <th className={styles.th} onClick={() => handleSort('Business_Evidence_Score')} aria-sort={sort.key === 'Business_Evidence_Score' ? sort.dir : 'none'}>
                Evidence <SortIcon col="Business_Evidence_Score" />
              </th>
              {hasExpected && (
                <th className={styles.th} onClick={() => handleSort('Expected_Amount')}>
                  Expected <SortIcon col="Expected_Amount" />
                </th>
              )}
              {hasBilledAmount && (
                <th className={styles.th} onClick={() => handleSort('Billed_Amount')}>
                  Billed <SortIcon col="Billed_Amount" />
                </th>
              )}
              <th className={styles.th} onClick={() => handleSort('Potential_Leakage')} aria-sort={sort.key === 'Potential_Leakage' ? sort.dir : 'none'}>
                Leakage ₹ <SortIcon col="Potential_Leakage" />
              </th>
              <th className={styles.th}>Risk Factors</th>
            </tr>
          </thead>
          <tbody>
            {pageRows.length === 0 ? (
              <tr>
                <td colSpan={8} className={styles.empty}>
                  <div className={styles.emptyInner}>
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <circle cx="11" cy="11" r="8" stroke="var(--text-muted)" strokeWidth="1.5" />
                      <path d="M21 21l-4.35-4.35" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" />
                      <path d="M8 11h6" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" />
                    </svg>
                    <span>No records match your filters</span>
                  </div>
                </td>
              </tr>
            ) : (
              pageRows.map((row, idx) => {
                const globalIdx = (page - 1) * PAGE_SIZE + idx;
                const isExpanded = expanded === globalIdx;
                const predColor = PREDICTION_COLOR[row.Prediction] ?? 'info';
                return (
                  <>
                    <tr
                      key={`row-${globalIdx}`}
                      className={`${styles.tr} ${styles[`row-${predColor}`]}`}
                      onClick={() => setExpanded(isExpanded ? null : globalIdx)}
                      aria-expanded={isExpanded}
                      style={{ cursor: 'pointer' }}
                    >
                      <td className={styles.td}>
                        <RiskBadge level={row.Risk_Level} />
                      </td>
                      <td className={styles.td}>
                        <span className={`${styles.predLabel} ${styles[predColor]}`}>
                          {labelPrediction(row.Prediction)}
                        </span>
                      </td>
                      <td className={styles.td}>
                        <div className={styles.scoreBar}>
                          <div className={styles.scoreTrack}>
                            <div
                              className={`${styles.scoreFill} ${styles[`score-${predColor}`]}`}
                              style={{ width: `${row.ML_Anomaly_Score ?? 0}%` }}
                            />
                          </div>
                          <span className={styles.scoreNum}>
                            {formatDecimal(row.ML_Anomaly_Score, 1)}
                          </span>
                        </div>
                      </td>
                      <td className={styles.td}>
                        <span className={styles.evidenceScore}>{row.Business_Evidence_Score ?? '—'}</span>
                      </td>
                      {hasExpected && (
                        <td className={styles.td + ' ' + styles.mono}>
                          {formatCurrency(row.Expected_Amount, 0)}
                        </td>
                      )}
                      {hasBilledAmount && (
                        <td className={styles.td + ' ' + styles.mono}>
                          {formatCurrency(row.Billed_Amount, 0)}
                        </td>
                      )}
                      <td className={`${styles.td} ${styles.mono} ${row.Potential_Leakage > 0 ? styles.leakageCell : ''}`}>
                        {row.Potential_Leakage > 0 ? formatCurrency(row.Potential_Leakage, 0) : '—'}
                      </td>
                      <td className={styles.td}>
                        <span className={styles.factors} title={row.Risk_Factors}>
                          {row.Risk_Factors
                            ? row.Risk_Factors.split(';')[0].trim()
                            : '—'}
                        </span>
                      </td>
                    </tr>
                    {isExpanded && (
                      <tr key={`expand-${globalIdx}`} className={styles.expandRow}>
                        <td colSpan={8} className={styles.expandCell}>
                          <div className={styles.expandContent}>
                            <div className={styles.expandGrid}>
                              {Object.entries(row).map(([k, v]) => (
                                <div key={k} className={styles.expandItem}>
                                  <span className={styles.expandKey}>{k.replace(/_/g, ' ')}</span>
                                  <span className={styles.expandVal}>{v ?? '—'}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className={styles.pagination} aria-label="Table pagination">
          <button
            className={styles.pageBtn}
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            type="button"
            aria-label="Previous page"
          >
            ← Prev
          </button>
          <div className={styles.pageNumbers}>
            {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
              const p = i + 1;
              return (
                <button
                  key={p}
                  className={`${styles.pageBtn} ${page === p ? styles.pageBtnActive : ''}`}
                  onClick={() => setPage(p)}
                  type="button"
                  aria-label={`Page ${p}`}
                  aria-current={page === p ? 'page' : undefined}
                >
                  {p}
                </button>
              );
            })}
          </div>
          <button
            className={styles.pageBtn}
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            type="button"
            aria-label="Next page"
          >
            Next →
          </button>
        </div>
      )}
    </section>
  );
}
