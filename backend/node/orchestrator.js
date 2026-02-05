import { nanoid } from 'nanoid';
import { getDb } from './db.js';
import { callPython } from './pythonBridge.js';

const stages = ['ingest', 'analyze', 'visualize', 'reflect'];

const emitStatus = (emit, runId, stage, status, message) => {
  emit?.('pipeline:status', {
    runId,
    stage,
    status,
    message
  });
};

export const runPipeline = async ({ userId, fileId, filePath }, emit) => {
  const db = await getDb();
  const runId = nanoid(16);
  const now = new Date().toISOString();
  const stageResults = {};
  await db.run(
    `INSERT INTO runs (id, user_id, file_id, status, stage, started_at)
     VALUES (?, ?, ?, ?, ?, ?)`
    , runId, userId, fileId, 'running', 'ingest', now
  );

  try {
    for (const stage of stages) {
      await db.run(`UPDATE runs SET stage = ? WHERE id = ?`, stage, runId);
      emitStatus(emit, runId, stage, 'running', `Stage ${stage} started`);
      const result = await callPython(`/pipeline/${stage}`, { runId, fileId, filePath });
      stageResults[stage] = {
        message: result?.message || `Stage ${stage} complete`,
        meta: result?.meta || null,
        alert: result?.alert || null,
        completedAt: new Date().toISOString()
      };
      await db.run(
        `UPDATE runs SET meta_json = ? WHERE id = ?`,
        JSON.stringify({
          runId,
          fileId,
          filePath,
          currentStage: stage,
          stages: stageResults
        }),
        runId
      );
      emitStatus(emit, runId, stage, 'complete', result?.message || `Stage ${stage} complete`);
      if (result?.alert) {
        await createAlert(db, userId, runId, result.alert.level || 'info', result.alert.message);
        emit?.('alert:new', result.alert);
      }
    }
    await db.run(
      `UPDATE runs SET status = ?, completed_at = ?, meta_json = ? WHERE id = ?`,
      'complete',
      new Date().toISOString(),
      JSON.stringify({
        runId,
        fileId,
        filePath,
        status: 'complete',
        stages: stageResults
      }),
      runId
    );
    emitStatus(emit, runId, 'reflect', 'complete', 'Pipeline complete');
  } catch (err) {
    await db.run(
      `UPDATE runs SET status = ?, completed_at = ?, meta_json = ? WHERE id = ?`,
      'failed',
      new Date().toISOString(),
      JSON.stringify({ error: err.message, stages: stageResults }),
      runId
    );
    emitStatus(emit, runId, 'error', 'failed', err.message);
    throw err;
  }

  return runId;
};

export const createAlert = async (db, userId, runId, level, message) => {
  await db.run(
    `INSERT INTO alerts (id, user_id, run_id, level, message, created_at)
     VALUES (?, ?, ?, ?, ?, ?)`
    , nanoid(16), userId, runId, level, message, new Date().toISOString()
  );
};
