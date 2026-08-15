import { useFileUpload } from './hooks/useFileUpload';
import { useAnalysis }   from './hooks/useAnalysis';

import Header              from './components/layout/Header';
import UploadZone          from './components/upload/UploadZone';
import AnalyzeButton       from './components/upload/AnalyzeButton';
import LoadingSpinner      from './components/shared/LoadingSpinner';
import ErrorAlert          from './components/shared/ErrorAlert';
import SummaryCards        from './components/dashboard/SummaryCards';
import RiskDonut           from './components/dashboard/RiskDonut';
import PredictionBreakdown from './components/dashboard/PredictionBreakdown';
import ResultsTable        from './components/results/ResultsTable';

import styles from './App.module.css';

export default function App() {
  const upload   = useFileUpload();
  const analysis = useAnalysis();

  const handleAnalyze = () => {
    if (upload.file) {
      analysis.analyze(upload.file);
    }
  };

  const handleReset = () => {
    upload.clearFile();
    analysis.reset();
  };

  return (
    <>
      <Header
        onReset={handleReset}
        showReset={analysis.isSuccess}
      />

      <main className={styles.main}>
        {/* ============ UPLOAD VIEW ============ */}
        {!analysis.isSuccess && !analysis.isLoading && (
          <div className={`container ${styles.uploadView}`}>
            {/* Hero */}
            <div className={styles.hero}>
              <div className={styles.heroBadge}>
                <span className={styles.heroBadgeDot} />
                AI-Powered Detection
              </div>
              <h1 className={styles.heroTitle}>
                Detect Revenue{' '}
                <span className={styles.heroAccent}>Leakage</span>
                {' '}Instantly
              </h1>
              <p className={styles.heroDesc}>
                Upload your billing data and our hybrid XGBoost + business rules engine
                will identify revenue leakage, anomalies, and risk levels — in seconds.
              </p>

              {/* Feature pills */}
              <div className={styles.featurePills}>
                {[
                  { icon: '⚡', label: 'Real-time Analysis' },
                  { icon: '🎯', label: 'Hybrid ML + Rules' },
                  { icon: '📊', label: 'Risk Scoring' },
                  { icon: '💰', label: 'Leakage Quantified' },
                ].map(f => (
                  <span key={f.label} className={styles.pill}>
                    <span aria-hidden="true">{f.icon}</span> {f.label}
                  </span>
                ))}
              </div>
            </div>

            {/* Upload card */}
            <div className={styles.uploadCard}>
              <div className={styles.uploadCardHeader}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke="var(--accent-light)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span>Upload Billing Data</span>
              </div>

              <UploadZone
                file={upload.file}
                isDragging={upload.isDragging}
                error={upload.error}
                onDragEnter={upload.onDragEnter}
                onDragLeave={upload.onDragLeave}
                onDragOver={upload.onDragOver}
                onDrop={upload.onDrop}
                onInputChange={upload.onInputChange}
                onClear={upload.clearFile}
              />

              <AnalyzeButton
                onClick={handleAnalyze}
                disabled={!upload.file || !!upload.error}
                isLoading={analysis.isLoading}
              />

              {/* Accepted formats hint */}
              <p className={styles.formatHint}>
                Accepts CSV, XLSX, XLS · Supports flexible column naming
              </p>
            </div>

            {/* Error from analysis */}
            {analysis.isError && (
              <ErrorAlert message={analysis.error} onRetry={handleReset} />
            )}
          </div>
        )}

        {/* ============ LOADING VIEW ============ */}
        {analysis.isLoading && (
          <div className={`container ${styles.centered}`}>
            <LoadingSpinner
              message="Running ML model and business rule engine…"
              progress={analysis.progress > 0 ? analysis.progress : null}
            />
          </div>
        )}

        {/* ============ RESULTS DASHBOARD ============ */}
        {analysis.isSuccess && analysis.data && (
          <div className={styles.dashboard}>
            {/* Dashboard header */}
            <div className={`container ${styles.dashboardHeader}`}>
              <div>
                <h2 className={styles.dashTitle}>Analysis Complete</h2>
                <p className={styles.dashSub}>
                  File: <strong>{analysis.data.filename}</strong>
                </p>
              </div>
            </div>

            {/* KPI cards */}
            <div className="container">
              <SummaryCards summary={analysis.data.summary} />
            </div>

            {/* Charts row */}
            <div className="container">
              <div className={styles.chartsGrid}>
                <RiskDonut summary={analysis.data.summary} />
                <PredictionBreakdown results={analysis.data.results} />
              </div>
            </div>

            {/* Results table */}
            <div className="container">
              <ResultsTable
                results={analysis.data.results}
                filename={analysis.data.filename}
              />
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className={styles.footer}>
        <div className="container">
          <span>AI Revenue Leakage Detector</span>
          <span>XGBoost · Business Rules · Hybrid Decision Engine</span>
        </div>
      </footer>
    </>
  );
}
