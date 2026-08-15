import { useRef } from 'react';
import { formatFileSize } from '../../utils/formatters';
import styles from './UploadZone.module.css';

const ACCEPTED = '.csv,.xlsx,.xls';

/**
 * Drag-and-drop file upload area.
 */
export default function UploadZone({
  file,
  isDragging,
  error,
  onDragEnter,
  onDragLeave,
  onDragOver,
  onDrop,
  onInputChange,
  onClear,
}) {
  const inputRef = useRef(null);

  const triggerInput = () => inputRef.current?.click();

  return (
    <div className={styles.container}>
      {/* Drop zone */}
      <div
        className={`${styles.zone} ${isDragging ? styles.dragging : ''} ${file ? styles.hasFile : ''} ${error ? styles.hasError : ''}`}
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDragOver={onDragOver}
        onDrop={onDrop}
        onClick={!file ? triggerInput : undefined}
        role="button"
        tabIndex={0}
        aria-label="Drop zone — drag and drop a CSV or Excel file here"
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') triggerInput(); }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          onChange={onInputChange}
          className={styles.hiddenInput}
          aria-hidden="true"
          tabIndex={-1}
          id="file-input"
        />

        {/* Idle state */}
        {!file && (
          <div className={styles.idle}>
            <div className={styles.iconWrap}>
              {isDragging ? (
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path d="M12 3v12M7 8l5-5 5 5" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M20 16v3a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-3" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" />
                </svg>
              ) : (
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" />
                  <path d="M12 3v12M8 9l4-4 4 4" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </div>

            <div className={styles.idleText}>
              {isDragging ? (
                <p className={styles.dropNow}>Release to upload</p>
              ) : (
                <>
                  <p className={styles.primary}>
                    <span className={styles.clickLink}>Click to upload</span> or drag & drop
                  </p>
                  <p className={styles.secondary}>CSV, XLSX, XLS · Max 50 MB</p>
                </>
              )}
            </div>

            {/* Format hints */}
            <div className={styles.formatHints}>
              {['CSV', 'XLSX', 'XLS'].map(fmt => (
                <span key={fmt} className={styles.formatTag}>{fmt}</span>
              ))}
            </div>
          </div>
        )}

        {/* File selected state */}
        {file && (
          <div className={styles.fileInfo}>
            <div className={styles.fileIcon} aria-hidden="true">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z" stroke="var(--success)" strokeWidth="1.5" strokeLinejoin="round" />
                <path d="M14 2v6h6" stroke="var(--success)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M9 12h6M9 16h4" stroke="var(--success)" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </div>
            <div className={styles.fileMeta}>
              <span className={styles.fileName}>{file.name}</span>
              <span className={styles.fileSize}>{formatFileSize(file.size)}</span>
            </div>
            <button
              className={styles.clearBtn}
              onClick={(e) => { e.stopPropagation(); onClear(); }}
              type="button"
              aria-label="Remove selected file"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M18 6 6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </button>
          </div>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className={styles.errorMsg} role="alert">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="10" stroke="var(--danger)" strokeWidth="2" />
            <path d="M12 7v5M12 15v2" stroke="var(--danger)" strokeWidth="2" strokeLinecap="round" />
          </svg>
          {error}
        </div>
      )}
    </div>
  );
}
