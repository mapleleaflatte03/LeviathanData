import { spawn } from 'node:child_process';
import { config } from './config.js';

let pythonProcess;

const healthUrl = () => `${config.pythonUrl.replace(/\/$/, '')}/health`;

export const ensurePythonService = async (log) => {
  const ok = await checkHealth();
  if (ok) return true;
  if (!config.pythonAutostart) return false;
  if (pythonProcess) return false;

  pythonProcess = spawn('python3', ['-m', 'uvicorn', 'backend.python.app:app', '--host', '0.0.0.0', '--port', '8000'], {
    cwd: process.cwd(),
    stdio: ['ignore', 'pipe', 'pipe']
  });
  pythonProcess.stdout.on('data', (data) => log?.info({ msg: data.toString() }, 'python stdout'));
  pythonProcess.stderr.on('data', (data) => log?.error({ msg: data.toString() }, 'python stderr'));
  pythonProcess.on('exit', (code) => {
    log?.warn({ code }, 'python process exited');
    pythonProcess = undefined;
  });

  for (let i = 0; i < 10; i += 1) {
    await new Promise((r) => setTimeout(r, 500));
    if (await checkHealth()) return true;
  }
  return false;
};

export const checkHealth = async () => {
  try {
    const res = await fetch(healthUrl());
    return res.ok;
  } catch (err) {
    return false;
  }
};

export const callPython = async (path, body) => {
  const url = `${config.pythonUrl.replace(/\/$/, '')}${path}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {})
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Python service error ${res.status}: ${text}`);
  }
  return res.json();
};
