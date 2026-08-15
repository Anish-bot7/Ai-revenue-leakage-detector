import { useState, useCallback } from 'react';

const ACCEPTED_EXTENSIONS = ['.csv', '.xlsx', '.xls'];
const MAX_SIZE_MB = 50;

function validateFile(file) {
  if (!file) return 'No file selected.';

  const name = file.name.toLowerCase();
  const hasValidExt = ACCEPTED_EXTENSIONS.some(ext => name.endsWith(ext));
  if (!hasValidExt) {
    return `Unsupported format. Please upload CSV or Excel (.csv, .xlsx, .xls).`;
  }

  const sizeMB = file.size / (1024 * 1024);
  if (sizeMB > MAX_SIZE_MB) {
    return `File too large (${sizeMB.toFixed(1)} MB). Max allowed: ${MAX_SIZE_MB} MB.`;
  }

  return null; // valid
}

/**
 * Manages file selection, drag-and-drop, and validation.
 */
export function useFileUpload() {
  const [file, setFile]           = useState(null);
  const [error, setError]         = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const selectFile = useCallback((selectedFile) => {
    const err = validateFile(selectedFile);
    if (err) {
      setError(err);
      setFile(null);
    } else {
      setError(null);
      setFile(selectedFile);
    }
  }, []);

  const clearFile = useCallback(() => {
    setFile(null);
    setError(null);
  }, []);

  // Drag handlers
  const onDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    // Only clear if leaving the drop zone itself (not a child element)
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsDragging(false);
    }
  }, []);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) selectFile(droppedFile);
  }, [selectFile]);

  const onInputChange = useCallback((e) => {
    const picked = e.target.files[0];
    if (picked) selectFile(picked);
    // Reset input so same file can be re-selected
    e.target.value = '';
  }, [selectFile]);

  return {
    file,
    error,
    isDragging,
    selectFile,
    clearFile,
    onDragEnter,
    onDragLeave,
    onDragOver,
    onDrop,
    onInputChange,
  };
}
