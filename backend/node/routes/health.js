import { Router } from 'express';
import { config } from '../config.js';

const router = Router();

router.get('/', (req, res) => {
  res.json({ ok: true, env: config.nodeEnv, ts: new Date().toISOString() });
});

export default router;
