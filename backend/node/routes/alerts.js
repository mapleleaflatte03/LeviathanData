import { Router } from 'express';
import { getDb } from '../db.js';

const router = Router();

router.get('/', async (req, res) => {
  const db = await getDb();
  const alerts = await db.all(
    `SELECT * FROM alerts WHERE user_id = ? ORDER BY created_at DESC LIMIT 50`,
    req.user.sub
  );
  return res.json({ alerts });
});

export default router;
