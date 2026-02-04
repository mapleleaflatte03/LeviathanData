import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import crypto from 'node:crypto';
import { nanoid } from 'nanoid';
import { config } from './config.js';
import { getDb } from './db.js';

export const hashPassword = async (password) => {
  const salt = await bcrypt.genSalt(10);
  return bcrypt.hash(password, salt);
};

export const verifyPassword = async (password, hash) => bcrypt.compare(password, hash);

const hashToken = (token) => crypto.createHash('sha256').update(token).digest('hex');

export const issueTokens = async (user) => {
  const accessToken = jwt.sign(
    { sub: user.id, role: user.role, email: user.email },
    config.jwtSecret,
    { expiresIn: config.jwtAccessTtl }
  );
  const refreshToken = nanoid(64);
  const db = await getDb();
  const now = new Date().toISOString();
  const expiresAt = new Date(Date.now() + parseTtlToMs(config.jwtRefreshTtl)).toISOString();
  await db.run(
    `INSERT INTO refresh_tokens (id, user_id, token_hash, expires_at, created_at)
     VALUES (?, ?, ?, ?, ?)`
    , nanoid(16), user.id, hashToken(refreshToken), expiresAt, now
  );
  return { accessToken, refreshToken, expiresAt };
};

export const rotateRefreshToken = async (userId, refreshToken) => {
  const db = await getDb();
  const tokenHash = hashToken(refreshToken);
  const row = await db.get(
    `SELECT * FROM refresh_tokens WHERE user_id = ? AND token_hash = ? AND revoked_at IS NULL`,
    userId, tokenHash
  );
  if (!row) return null;
  if (new Date(row.expires_at) < new Date()) return null;
  const revokedAt = new Date().toISOString();
  await db.run(`UPDATE refresh_tokens SET revoked_at = ? WHERE id = ?`, revokedAt, row.id);
  const user = await db.get(`SELECT * FROM users WHERE id = ?`, userId);
  if (!user) return null;
  return issueTokens(user);
};

export const revokeRefreshToken = async (userId, refreshToken) => {
  const db = await getDb();
  const tokenHash = hashToken(refreshToken);
  await db.run(
    `UPDATE refresh_tokens SET revoked_at = ? WHERE user_id = ? AND token_hash = ?`,
    new Date().toISOString(), userId, tokenHash
  );
};

export const parseTtlToMs = (ttl) => {
  if (!ttl) return 0;
  const match = String(ttl).trim().match(/^(\d+)([smhd])?$/i);
  if (!match) return 0;
  const value = Number(match[1]);
  const unit = (match[2] || 's').toLowerCase();
  switch (unit) {
    case 'm': return value * 60 * 1000;
    case 'h': return value * 60 * 60 * 1000;
    case 'd': return value * 24 * 60 * 60 * 1000;
    case 's':
    default: return value * 1000;
  }
};

export const authMiddleware = (req, res, next) => {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : null;
  if (!token) return res.status(401).json({ error: 'Missing token' });
  try {
    const payload = jwt.verify(token, config.jwtSecret);
    req.user = payload;
    return next();
  } catch (err) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};
