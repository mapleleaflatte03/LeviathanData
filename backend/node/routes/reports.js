import { Router } from 'express';
import fs from 'node:fs/promises';
import { createReadStream, existsSync } from 'node:fs';
import path from 'node:path';
import { getDb } from '../db.js';
import { config } from '../config.js';

const router = Router();
let puppeteerModule;

const safeJsonParse = (value, fallback = {}) => {
  if (!value) return fallback;
  try {
    return JSON.parse(value);
  } catch (err) {
    return fallback;
  }
};

const escapeHtml = (value = '') =>
  String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');

const loadPuppeteer = async () => {
  if (puppeteerModule) return puppeteerModule;
  try {
    const mod = await import('puppeteer');
    puppeteerModule = mod.default || mod;
    return puppeteerModule;
  } catch (err) {
    return null;
  }
};

const buildHtml = ({ run, user, runMeta, svgMarkup }) => {
  const stageMeta = runMeta?.stages || {};
  const analyze = stageMeta?.analyze?.meta || {};
  const ml = analyze?.ml || {};
  const reflectMeta = stageMeta?.reflect?.meta || {};
  const insights = reflectMeta?.insights || [];
  const summaryText = stageMeta?.reflect?.message || reflectMeta?.summary || 'Summary unavailable';
  const metric = ml?.primaryMetric ? `${ml.primaryMetric}: ${ml.primaryMetricValue ?? 'n/a'}` : 'Metric unavailable';
  const task = ml?.task || 'unknown';

  return `
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Leviathan Report</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 32px; color: #0b1a22; }
    h1 { color: #002233; }
    .card { border: 1px solid #dde6ee; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    .meta { font-size: 12px; color: #556; }
    .insight { margin: 6px 0; }
    .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #e6f5ff; font-size: 12px; margin-right: 6px; }
    .chart-wrap { border: 1px solid #e3eef5; border-radius: 10px; padding: 10px; background: #f9fdff; }
  </style>
</head>
<body>
  <h1>Leviathan Report</h1>
  <div class="card">
    <div class="meta">Generated: ${new Date().toISOString()}</div>
    <p><strong>User:</strong> ${escapeHtml(user.email)}</p>
    <p><strong>Run:</strong> ${escapeHtml(run.id)}</p>
    <p><strong>Status:</strong> ${escapeHtml(run.status)}</p>
    <p><strong>Stage:</strong> ${escapeHtml(run.stage)}</p>
    <p><strong>Task:</strong> ${escapeHtml(task)}</p>
    <p><strong>Primary Metric:</strong> ${escapeHtml(metric)}</p>
  </div>
  <div class="card">
    <h3>Pipeline Summary</h3>
    <p>${escapeHtml(summaryText)}</p>
    <div class="metric-grid">
      <div><span class="badge">Model</span> ${escapeHtml(ml?.selectedModel || 'n/a')}</div>
      <div><span class="badge">Refined</span> ${ml?.refined ? 'yes' : 'no'}</div>
      <div><span class="badge">Baseline</span> ${escapeHtml(String(ml?.baselineMetric ?? 'n/a'))}</div>
      <div><span class="badge">Needs Refinement</span> ${ml?.needsRefinement ? 'yes' : 'no'}</div>
    </div>
  </div>
  <div class="card">
    <h3>Insights</h3>
    ${
      insights.length
        ? insights.map((insight) => `<p class="insight">• ${escapeHtml(insight)}</p>`).join('')
        : '<p>No insights were generated for this run.</p>'
    }
  </div>
  <div class="card">
    <h3>Visualization</h3>
    <div class="chart-wrap">
      ${svgMarkup || '<p>No SVG artifact available.</p>'}
    </div>
  </div>
</body>
</html>
`;
};

router.post('/pdf', async (req, res) => {
  const runId = req.body?.runId;
  if (!runId) return res.status(400).json({ error: 'runId required' });
  const db = await getDb();
  const run = await db.get(`SELECT * FROM runs WHERE id = ? AND user_id = ?`, runId, req.user.sub);
  if (!run) return res.status(404).json({ error: 'Run not found' });
  const user = await db.get(`SELECT email FROM users WHERE id = ?`, req.user.sub);
  const runMeta = safeJsonParse(run.meta_json, {});

  let svgMarkup = '';
  const svgPath = runMeta?.stages?.visualize?.meta?.svgPath;
  if (svgPath) {
    try {
      svgMarkup = await fs.readFile(svgPath, 'utf-8');
    } catch (err) {
      svgMarkup = '';
    }
  }

  const puppeteer = await loadPuppeteer();
  if (!puppeteer) {
    return res.status(500).json({ error: 'Puppeteer not installed. Run npm install puppeteer.' });
  }

  let browser;
  try {
    browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
    const page = await browser.newPage();
    await page.setContent(buildHtml({ run, user, runMeta, svgMarkup }), { waitUntil: 'networkidle0' });

    const filename = `report-${run.id}.pdf`;
    const filePath = path.join(config.reportDir, filename);
    await fs.mkdir(config.reportDir, { recursive: true });
    await page.pdf({ path: filePath, format: 'A4' });
    await browser.close();
    return res.json({ ok: true, path: filePath });
  } catch (err) {
    if (browser) {
      try {
        await browser.close();
      } catch (_) {
        // no-op
      }
    }
    return res.status(500).json({ error: `PDF generation failed: ${err.message}` });
  }
});

// Serve report files (PDF/HTML) from data/reports
router.get('/:filename', async (req, res) => {
  const { filename } = req.params;
  
  // Security: only allow alphanumeric, dash, underscore, dot
  if (!/^[a-zA-Z0-9_\-\.]+\.(pdf|html)$/.test(filename)) {
    return res.status(400).json({ error: 'Invalid filename' });
  }
  
  const filePath = path.join(config.reportDir, filename);
  
  // Check file exists
  if (!existsSync(filePath)) {
    return res.status(404).json({ error: 'Report not found' });
  }
  
  // Set content type
  const ext = path.extname(filename).toLowerCase();
  const contentType = ext === '.pdf' ? 'application/pdf' : 'text/html';
  
  res.setHeader('Content-Type', contentType);
  res.setHeader('Content-Disposition', `inline; filename="${filename}"`);
  
  // Stream file
  const stream = createReadStream(filePath);
  stream.pipe(res);
  stream.on('error', (err) => {
    res.status(500).json({ error: `Error reading file: ${err.message}` });
  });
});

// List available reports
router.get('/', async (req, res) => {
  try {
    const files = await fs.readdir(config.reportDir);
    const reports = files
      .filter(f => f.endsWith('.pdf') || f.endsWith('.html'))
      .map(f => ({
        filename: f,
        url: `/api/reports/${f}`,
        type: f.endsWith('.pdf') ? 'pdf' : 'html'
      }));
    return res.json({ reports, count: reports.length });
  } catch (err) {
    return res.json({ reports: [], count: 0 });
  }
});

export default router;
