import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const env = process.env;

const parseBool = (value, fallback = false) => {
  if (value === undefined) return fallback;
  return ['1', 'true', 'yes', 'on'].includes(String(value).toLowerCase());
};

const resolvePath = (value, fallback) => {
  if (!value) return path.resolve(process.cwd(), fallback);
  if (value.startsWith('.')) return path.resolve(process.cwd(), value);
  return value;
};

export const config = {
  nodeEnv: env.NODE_ENV || 'development',
  port: Number(env.PORT || 3000),
  pythonUrl: env.PYTHON_URL || 'http://127.0.0.1:8000',
  pythonAutostart: parseBool(env.PYTHON_AUTOSTART, true),
  jwtSecret: env.JWT_SECRET || 'change-me',
  jwtAccessTtl: env.JWT_ACCESS_TTL || '15m',
  jwtRefreshTtl: env.JWT_REFRESH_TTL || '7d',
  dbPath: resolvePath(env.DB_PATH, './data/leviathan.db'),
  uploadDir: resolvePath(env.UPLOAD_DIR, './data/uploads'),
  reportDir: resolvePath(env.REPORT_DIR, './data/reports'),
  maxUploadMb: Number(env.MAX_UPLOAD_MB || 50),
  corsOrigin: env.CORS_ORIGIN || 'http://localhost:3000',
  logLevel: env.LOG_LEVEL || 'info',
  promPort: Number(env.PROM_PORT || 9090),
  llm: {
    baseUrl: env.LLM_BASE_URL || '',
    apiKey: env.LLM_API_KEY || '',
    model: env.LLM_MODEL || 'qwen3-32b'
  },
  llmFallback: {
    baseUrl: env.LLM_FALLBACK_BASE_URL || '',
    apiKey: env.LLM_FALLBACK_API_KEY || '',
    model: env.LLM_FALLBACK_MODEL || ''
  }
};

export const paths = {
  root: path.resolve(__dirname, '../..'),
  uploads: config.uploadDir,
  reports: config.reportDir
};
