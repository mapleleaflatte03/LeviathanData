import { Router } from 'express';
import { config } from '../config.js';
import { getLlmStats } from '../llmClient.js';

const router = Router();

router.get('/', (req, res) => {
  res.json({ ok: true, env: config.nodeEnv, ts: new Date().toISOString() });
});

router.get('/llm', (req, res) => {
  res.json({ ok: true, stats: getLlmStats(), ts: new Date().toISOString() });
});

export default router;
