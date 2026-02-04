import { Router } from 'express';
import { getDb } from '../db.js';

const router = Router();

router.get('/:id', async (req, res) => {
  const db = await getDb();
  const run = await db.get(`SELECT * FROM runs WHERE id = ? AND user_id = ?`, req.params.id, req.user.sub);
  if (!run) return res.status(404).json({ error: 'Run not found' });
  return res.json({ run });
});

export default router;
