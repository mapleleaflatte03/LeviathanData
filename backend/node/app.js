import 'dotenv/config';
import http from 'node:http';
import path from 'node:path';
import fs from 'node:fs';
import crypto from 'node:crypto';
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import pino from 'pino';
import pinoHttp from 'pino-http';
import client from 'prom-client';
import { config, paths } from './config.js';
import { initDb, getDb } from './db.js';
import { authMiddleware } from './auth.js';
import { initWebsocket } from './ws.js';
import { ensurePythonService } from './pythonBridge.js';
import authRoutes from './routes/auth.js';
import { createFilesRouter } from './routes/files.js';
import runRoutes from './routes/runs.js';
import alertRoutes from './routes/alerts.js';
import chatRoutes from './routes/chat.js';
import healthRoutes from './routes/health.js';
import reportRoutes from './routes/reports.js';
import { errorHandler } from './middleware/error.js';

const startBackgroundScanner = (emitToUser, log) => {
  setInterval(async () => {
    const db = await getDb();
    const cutoff = new Date(Date.now() - 10 * 60 * 1000).toISOString();
    const staleRuns = await db.all(
      `SELECT * FROM runs WHERE status = 'running' AND started_at < ?`,
      cutoff
    );
    for (const run of staleRuns) {
      const existing = await db.get(`SELECT id FROM alerts WHERE run_id = ?`, run.id);
      if (existing) continue;
      await db.run(
        `INSERT INTO alerts (id, user_id, run_id, level, message, created_at)
         VALUES (?, ?, ?, ?, ?, ?)`
        , crypto.randomUUID(), run.user_id, run.id, 'warning', 'Run appears stalled', new Date().toISOString()
      );
      emitToUser(run.user_id, 'alert:new', {
        runId: run.id,
        level: 'warning',
        message: 'Run appears stalled'
      });
      log?.info({ runId: run.id }, 'stale run alert emitted');
    }
  }, 60 * 1000);
};

export const createApp = async ({ startPython = true, startBackground = true } = {}) => {
  const log = pino({ level: config.logLevel });
  const app = express();
  const server = http.createServer(app);

  // Trust proxy - needed when behind Caddy/nginx reverse proxy
  app.set('trust proxy', 1);

  const registry = new client.Registry();
  client.collectDefaultMetrics({ register: registry });

  app.use(pinoHttp({ logger: log }));
  app.use(helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "https://cdn.plot.ly"],
        styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
        fontSrc: ["'self'", "https://fonts.gstatic.com"],
        imgSrc: ["'self'", "data:"],
        connectSrc: ["'self'", "wss:", "ws:"],
      },
    },
  }));
  app.use(cors({ origin: config.corsOrigin, credentials: true }));
  app.use(rateLimit({ windowMs: 60 * 1000, limit: 120, validate: { xForwardedForHeader: false } }));
  app.use(express.json({ limit: '2mb' }));
  app.use(express.urlencoded({ extended: true }));

  await initDb();
  fs.mkdirSync(paths.uploads, { recursive: true });
  fs.mkdirSync(paths.reports, { recursive: true });

  const ws = initWebsocket(server, log);

  app.use('/api/health', healthRoutes);
  app.get('/api/metrics', async (req, res) => {
    res.set('Content-Type', registry.contentType);
    res.end(await registry.metrics());
  });

  app.use('/api/auth', authRoutes);
  app.use('/api/chat', authMiddleware, chatRoutes);
  app.use('/api/files', authMiddleware, createFilesRouter({ emitToUser: ws.emitToUser }));
  app.use('/api/runs', authMiddleware, runRoutes);
  app.use('/api/alerts', authMiddleware, alertRoutes);
  app.use('/api/reports', authMiddleware, reportRoutes);

  const frontendRoot = path.join(paths.root, 'frontend');
  
  // Disable caching for development - force fresh assets
  app.use((req, res, next) => {
    if (req.path.endsWith('.js') || req.path.endsWith('.css') || req.path.endsWith('.html')) {
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
      res.setHeader('Pragma', 'no-cache');
      res.setHeader('Expires', '0');
    }
    next();
  });
  
  app.use(express.static(frontendRoot, { etag: false, lastModified: false }));
  app.get('*', (req, res) => {
    if (req.path.startsWith('/api')) return res.status(404).json({ error: 'Not found' });
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.sendFile(path.join(frontendRoot, 'index.html'));
  });

  app.use(errorHandler);

  if (startPython) {
    await ensurePythonService(log);
  }
  if (startBackground) {
    startBackgroundScanner(ws.emitToUser, log);
  }

  return { app, server, log, registry, ws };
};

export const startServer = async () => {
  const { server, log } = await createApp({ startPython: true, startBackground: true });
  server.listen(config.port, () => {
    log.info(`Leviathan gateway listening on port ${config.port}`);
  });
  return server;
};
