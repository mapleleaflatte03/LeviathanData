import { Router } from 'express';
import { nanoid } from 'nanoid';
import { getDb } from '../db.js';
import { hashPassword, verifyPassword, issueTokens, rotateRefreshToken, revokeRefreshToken, authMiddleware } from '../auth.js';
import { requireFields } from '../middleware/validate.js';

const router = Router();

router.post('/register', requireFields(['email', 'password']), async (req, res) => {
  const { email, password } = req.body;
  const db = await getDb();
  const existing = await db.get('SELECT id FROM users WHERE email = ?', email);
  if (existing) return res.status(409).json({ error: 'Email already registered' });
  const id = nanoid(16);
  const passwordHash = await hashPassword(password);
  await db.run(
    `INSERT INTO users (id, email, password_hash, role, created_at)
     VALUES (?, ?, ?, ?, ?)`
    , id, email, passwordHash, 'user', new Date().toISOString()
  );
  const user = await db.get('SELECT id, email, role FROM users WHERE id = ?', id);
  const tokens = await issueTokens(user);
  return res.json({ user, ...tokens });
});

router.post('/login', requireFields(['email', 'password']), async (req, res) => {
  const { email, password } = req.body;
  const db = await getDb();
  const user = await db.get('SELECT * FROM users WHERE email = ?', email);
  if (!user) return res.status(401).json({ error: 'Invalid credentials' });
  const ok = await verifyPassword(password, user.password_hash);
  if (!ok) return res.status(401).json({ error: 'Invalid credentials' });
  const tokens = await issueTokens(user);
  return res.json({ user: { id: user.id, email: user.email, role: user.role }, ...tokens });
});

router.post('/refresh', requireFields(['userId', 'refreshToken']), async (req, res) => {
  const { userId, refreshToken } = req.body;
  const tokens = await rotateRefreshToken(userId, refreshToken);
  if (!tokens) return res.status(401).json({ error: 'Invalid refresh token' });
  return res.json(tokens);
});

router.post('/logout', requireFields(['userId', 'refreshToken']), async (req, res) => {
  const { userId, refreshToken } = req.body;
  await revokeRefreshToken(userId, refreshToken);
  return res.json({ ok: true });
});

router.get('/me', authMiddleware, async (req, res) => {
  if (!req.user?.sub) return res.status(401).json({ error: 'Unauthorized' });
  const db = await getDb();
  const user = await db.get('SELECT id, email, role FROM users WHERE id = ?', req.user.sub);
  return res.json({ user });
});

export default router;
