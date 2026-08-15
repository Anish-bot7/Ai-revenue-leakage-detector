// API service layer — wraps all backend calls

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Upload a CSV/Excel file and run revenue leakage prediction.
 * @param {File} file
 * @param {function} onProgress - optional progress callback (0–100)
 * @returns {Promise<Object>} { success, filename, summary, results }
 */
export async function analyzeFile(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        const pct = Math.round((e.loaded / e.total) * 100);
        onProgress(pct);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error('Invalid response from server.'));
        }
      } else {
        let detail = `Server error (${xhr.status})`;
        try {
          const body = JSON.parse(xhr.responseText);
          if (body.detail) detail = body.detail;
        } catch { /* ignore */ }
        reject(new Error(detail));
      }
    });

    xhr.addEventListener('error', () => {
      reject(new Error('Network error. Is the backend running?'));
    });

    xhr.addEventListener('timeout', () => {
      reject(new Error('Request timed out. Please try again.'));
    });

    xhr.timeout = 120_000; // 2 minutes
    xhr.open('POST', `${BASE_URL}/api/predict`);
    xhr.send(formData);
  });
}

/**
 * Health check — confirm backend is reachable.
 * @returns {Promise<boolean>}
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${BASE_URL}/health`, { method: 'GET' });
    return res.ok;
  } catch {
    return false;
  }
}
