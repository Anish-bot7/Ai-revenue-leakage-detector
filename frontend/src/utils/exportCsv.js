/**
 * Export an array of objects to a downloadable CSV file.
 * @param {Object[]} data   - Array of row objects
 * @param {string} filename - Download filename (without .csv)
 */
export function exportToCsv(data, filename = 'results') {
  if (!data || data.length === 0) return;

  const headers = Object.keys(data[0]);

  const csvLines = [
    headers.join(','),
    ...data.map(row =>
      headers.map(key => {
        const val = row[key] ?? '';
        // Wrap in quotes if the value contains commas, quotes, or newlines
        const str = String(val);
        if (str.includes(',') || str.includes('"') || str.includes('\n')) {
          return `"${str.replace(/"/g, '""')}"`;
        }
        return str;
      }).join(',')
    ),
  ];

  const blob = new Blob([csvLines.join('\n')], {
    type: 'text/csv;charset=utf-8;',
  });

  const url  = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href     = url;
  link.download = `${filename}.csv`;
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
