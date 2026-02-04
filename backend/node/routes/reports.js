import { Router } from 'express';
import fs from 'node:fs/promises';
import path from 'node:path';
import { getDb } from '../db.js';
import { config } from '../config.js';

const router = Router();
let puppeteerModule;

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

const buildHtml = ({ run, user }) => `
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
  </style>
</head>
<body>
  <h1>Leviathan Report</h1>
  <div class="card">
    <div class="meta">Generated: ${new Date().toISOString()}</div>
    <p><strong>User:</strong> ${user.email}</p>
    <p><strong>Run:</strong> ${run.id}</p>
    <p><strong>Status:</strong> ${run.status}</p>
    <p><strong>Stage:</strong> ${run.stage}</p>
  </div>
  <div class="card">
    <h3>Summary</h3>
    <p>This report includes a condensed view of the run metadata and a placeholder for charts and insights.</p>
  </div>
</body>
</html>
`;

router.post('/pdf', async (req, res) => {
  const runId = req.body?.runId;
  if (!runId) return res.status(400).json({ error: 'runId required' });
  const db = await getDb();
  const run = await db.get(`SELECT * FROM runs WHERE id = ? AND user_id = ?`, runId, req.user.sub);
  if (!run) return res.status(404).json({ error: 'Run not found' });
  const user = await db.get(`SELECT email FROM users WHERE id = ?`, req.user.sub);

  const puppeteer = await loadPuppeteer();
  if (!puppeteer) {
    return res.status(500).json({ error: 'Puppeteer not installed. Run npm install puppeteer.' });
  }

  const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  await page.setContent(buildHtml({ run, user }), { waitUntil: 'networkidle0' });

  const filename = `report-${run.id}.pdf`;
  const filePath = path.join(config.reportDir, filename);
  await fs.mkdir(config.reportDir, { recursive: true });
  await page.pdf({ path: filePath, format: 'A4' });
  await browser.close();

  return res.json({ ok: true, path: filePath });
});

export default router;
