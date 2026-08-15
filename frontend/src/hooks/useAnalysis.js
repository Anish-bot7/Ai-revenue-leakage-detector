import { useState, useCallback } from 'react';
import { analyzeFile } from '../services/api';

/**
 * Manages the API call lifecycle: idle → loading → success / error.
 */
export function useAnalysis() {
  const [status, setStatus]     = useState('idle');   // 'idle' | 'loading' | 'success' | 'error'
  const [data, setData]         = useState(null);      // full API response
  const [error, setError]       = useState(null);      // error message string
  const [progress, setProgress] = useState(0);         // upload progress 0–100

  const analyze = useCallback(async (file) => {
    if (!file) return;

    setStatus('loading');
    setError(null);
    setData(null);
    setProgress(0);

    try {
      const result = await analyzeFile(file, (pct) => setProgress(pct));
      setData(result);
      setStatus('success');
    } catch (err) {
      setError(err.message || 'An unexpected error occurred.');
      setStatus('error');
    }
  }, []);

  const reset = useCallback(() => {
    setStatus('idle');
    setData(null);
    setError(null);
    setProgress(0);
  }, []);

  return {
    status,
    data,
    error,
    progress,
    analyze,
    reset,
    isLoading: status === 'loading',
    isSuccess: status === 'success',
    isError:   status === 'error',
  };
}
