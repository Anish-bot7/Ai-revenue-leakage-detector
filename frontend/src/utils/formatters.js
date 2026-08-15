// Number and display formatters

/**
 * Format a number as Indian Rupee currency.
 * e.g. 123456.78 → "₹1,23,456.78"
 */
export function formatCurrency(value, decimals = 2) {
  if (value == null || isNaN(value)) return '—';
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

/**
 * Format a plain number with Indian locale separators.
 * e.g. 123456 → "1,23,456"
 */
export function formatNumber(value) {
  if (value == null || isNaN(value)) return '—';
  return new Intl.NumberFormat('en-IN').format(value);
}

/**
 * Round a float to N decimal places and return a string.
 */
export function formatDecimal(value, places = 2) {
  if (value == null || isNaN(value)) return '—';
  return Number(value).toFixed(places);
}

/**
 * Format bytes as human-readable file size.
 */
export function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / 1024 ** i).toFixed(1)} ${units[i]}`;
}

/**
 * Return a friendly display label for a Prediction string.
 */
export function labelPrediction(prediction) {
  const map = {
    'Revenue Leakage Detected':    'Leakage Detected',
    'Potential Revenue Leakage':   'Potential Leakage',
    'Potential Anomaly - Review':  'Needs Review',
    'No Revenue Leakage':          'Clean',
  };
  return map[prediction] ?? prediction;
}

/**
 * Return a CSS variable name / color key for risk level.
 */
export function riskColor(level) {
  const map = {
    HIGH:   'danger',
    MEDIUM: 'warning',
    LOW:    'success',
  };
  return map[level] ?? 'info';
}

/**
 * Return a hex color for Recharts per risk level.
 */
export function riskHex(level) {
  const map = {
    HIGH:   '#ef4444',
    MEDIUM: '#f59e0b',
    LOW:    '#10b981',
  };
  return map[level] ?? '#6366f1';
}

/**
 * Return a hex color for Recharts per prediction type.
 */
export function predictionHex(prediction) {
  const map = {
    'Revenue Leakage Detected':   '#ef4444',
    'Potential Revenue Leakage':  '#f59e0b',
    'Potential Anomaly - Review': '#6366f1',
    'No Revenue Leakage':         '#10b981',
  };
  return map[prediction] ?? '#94a3b8';
}
