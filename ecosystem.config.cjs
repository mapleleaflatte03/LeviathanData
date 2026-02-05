require('dotenv').config();

module.exports = {
  apps: [
    {
      name: 'leviathan-python',
      cwd: '/root/leviathan',
      script: '.venv/bin/python',
      args: '-m uvicorn backend.python.app:app --host 127.0.0.1 --port 8000',
      interpreter: 'none',
      env: {
        DB_PATH: process.env.DB_PATH || './data/leviathan.db',
        UPLOAD_DIR: process.env.UPLOAD_DIR || './data/uploads',
        REPORT_DIR: process.env.REPORT_DIR || './data/reports',
        LLM_BASE_URL: process.env.LLM_BASE_URL || '',
        LLM_API_KEY: process.env.LLM_API_KEY || '',
        LLM_MODEL: process.env.LLM_MODEL || 'qwen3-32b',
        LLM_FALLBACK_BASE_URL: process.env.LLM_FALLBACK_BASE_URL || '',
        LLM_FALLBACK_API_KEY: process.env.LLM_FALLBACK_API_KEY || '',
        LLM_FALLBACK_MODEL: process.env.LLM_FALLBACK_MODEL || '',
        LOG_LEVEL: process.env.LOG_LEVEL || 'info'
      }
    },
    {
      name: 'leviathan-node',
      cwd: '/root/leviathan',
      script: 'backend/node/server.js',
      node_args: '--experimental-specifier-resolution=node',
      env: {
        NODE_ENV: process.env.NODE_ENV || 'production',
        PORT: process.env.PORT || 3000,
        PYTHON_URL: process.env.PYTHON_URL || 'http://127.0.0.1:8000',
        JWT_SECRET: process.env.JWT_SECRET,
        JWT_ACCESS_TTL: process.env.JWT_ACCESS_TTL || '15m',
        JWT_REFRESH_TTL: process.env.JWT_REFRESH_TTL || '7d',
        DB_PATH: process.env.DB_PATH || './data/leviathan.db',
        UPLOAD_DIR: process.env.UPLOAD_DIR || './data/uploads',
        REPORT_DIR: process.env.REPORT_DIR || './data/reports',
        MAX_UPLOAD_MB: process.env.MAX_UPLOAD_MB || 250,
        CORS_ORIGIN: process.env.CORS_ORIGIN || 'https://app.welliam.codes',
        LLM_BASE_URL: process.env.LLM_BASE_URL || '',
        LLM_API_KEY: process.env.LLM_API_KEY || '',
        LLM_MODEL: process.env.LLM_MODEL || 'qwen3-32b',
        LLM_FALLBACK_BASE_URL: process.env.LLM_FALLBACK_BASE_URL || '',
        LLM_FALLBACK_API_KEY: process.env.LLM_FALLBACK_API_KEY || '',
        LLM_FALLBACK_MODEL: process.env.LLM_FALLBACK_MODEL || '',
        LOG_LEVEL: process.env.LOG_LEVEL || 'info'
      }
    }
  ]
};
