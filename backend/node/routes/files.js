import { Router } from 'express';
import multer from 'multer';
import path from 'node:path';
import { nanoid } from 'nanoid';
import fs from 'node:fs';
import { config } from '../config.js';
import { getDb } from '../db.js';
import { runPipeline } from '../orchestrator.js';

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, config.uploadDir),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const base = path.basename(file.originalname, ext).replace(/[^a-zA-Z0-9._-]+/g, '_').slice(0, 80) || 'upload';
    cb(null, `${Date.now()}-${nanoid(8)}-${base}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: config.maxUploadMb * 1024 * 1024 }
});

export const createFilesRouter = ({ emitToUser }) => {
  const router = Router();

  router.post('/upload', upload.single('file'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    const db = await getDb();
    const fileId = nanoid(16);
    await db.run(
      `INSERT INTO files (id, user_id, path, mime, size, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`
      , fileId, req.user.sub, req.file.path, req.file.mimetype, req.file.size, new Date().toISOString()
    );

    try {
      const runId = await runPipeline({
        userId: req.user.sub,
        fileId,
        filePath: req.file.path
      }, (type, payload) => emitToUser(req.user.sub, type, payload));

      return res.json({ fileId, runId, path: req.file.path });
    } catch (err) {
      return res.status(500).json({ error: err.message });
    }
  });

  router.get('/:id', async (req, res) => {
    const db = await getDb();
    const file = await db.get(`SELECT * FROM files WHERE id = ? AND user_id = ?`, req.params.id, req.user.sub);
    if (!file) return res.status(404).json({ error: 'Not found' });
    if (!fs.existsSync(file.path)) return res.status(404).json({ error: 'File missing on disk' });
    return res.download(file.path);
  });

  return router;
};
