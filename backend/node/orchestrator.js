import { nanoid } from 'nanoid';
import { getDb } from './db.js';
import { callPython } from './pythonBridge.js';
import fs from 'node:fs';

const stages = ['ingest', 'analyze', 'visualize', 'reflect'];

const emitStatus = (emit, runId, stage, status, message) => {
  emit?.('pipeline:status', {
    runId,
    stage,
    status,
    message
  });
};

const emitLog = (emit, level, stage, message) => {
  emit?.('swarm:log', { level, stage, message });
};

const emitChart = (emit, type, data) => {
  emit?.('pipeline:chart', { type, data });
};

const emitInsight = (emit, insight) => {
  emit?.('pipeline:insight', { insight });
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
      emitLog(emit, 'info', stage.toUpperCase(), `Starting ${stage}...`);
      
      const result = await callPython(`/pipeline/${stage}`, { runId, fileId, filePath });
      
      stageResults[stage] = {
        message: result?.message || `Stage ${stage} complete`,
        meta: result?.meta || null,
        alert: result?.alert || null,
        completedAt: new Date().toISOString()
      };

      // Emit stage-specific data to frontend
      if (stage === 'ingest' && result?.meta) {
        emitLog(emit, 'success', 'INGEST', `File: ${result.meta.fileName || 'unknown'} (${formatBytes(result.meta.size)})`);
        emitLog(emit, 'info', 'INGEST', `Dataset: ${result.meta.dataset || 'unknown'}, Type: ${result.meta.inputKind || 'unknown'}`);
      }

      if (stage === 'analyze' && result?.meta) {
        const ml = result.meta.ml || {};
        const eda = result.meta.eda || {};
        
        // Emit ML metrics
        if (ml.primaryMetric) {
          emitLog(emit, 'success', 'ANALYZE', `Model: ${ml.selectedModel || 'auto'} | ${ml.primaryMetric}: ${(ml.primaryMetricValue || 0).toFixed(4)}`);
        }
        
        // Emit chart data for visualization
        if (ml.vizPayload) {
          const vizData = ml.vizPayload;
          if (vizData.type === 'bar' && vizData.labels?.length) {
            emitChart(emit, 'main', {
              x: vizData.labels,
              actual: vizData.actual,
              predicted: vizData.predicted,
              chartType: 'bar'
            });
          } else if (vizData.type === 'line' && vizData.actual?.length) {
            emitChart(emit, 'main', {
              actual: vizData.actual,
              predicted: vizData.predicted,
              chartType: 'line'
            });
          }
        }
        
        // Emit distribution data
        if (eda.numericStats) {
          const stats = Object.entries(eda.numericStats).slice(0, 3);
          stats.forEach(([col, s]) => {
            emitLog(emit, 'info', 'EDA', `${col}: mean=${(s.mean || 0).toFixed(2)}, std=${(s.std || 0).toFixed(2)}`);
          });
        }
        
        // Emit sample predictions
        if (ml.samplePredictions?.length) {
          emitLog(emit, 'info', 'ML', `Sample predictions: ${ml.samplePredictions.slice(0, 5).map(p => `${p.actual}→${p.predicted}`).join(', ')}`);
        }
      }

      if (stage === 'visualize' && result?.meta) {
        emitLog(emit, 'success', 'VIZ', `Chart generated: ${result.meta.kind || 'bar'} (${formatBytes(result.meta.svgSize)})`);
        
        // Try to read and send SVG if exists
        if (result.meta.svgPath && fs.existsSync(result.meta.svgPath)) {
          try {
            const svgContent = fs.readFileSync(result.meta.svgPath, 'utf-8');
            emitChart(emit, 'svg', { svg: svgContent, kind: result.meta.kind });
          } catch {}
        }
      }

      if (stage === 'reflect' && result?.meta) {
        const reflect = result.meta;
        
        // Emit insights
        if (reflect.insights?.length) {
          reflect.insights.forEach(insight => {
            emitInsight(emit, insight);
            emitLog(emit, 'info', 'REFLECT', insight);
          });
        }
        
        // Emit summary to chat
        if (reflect.summary) {
          emit?.('chat:token', { token: reflect.summary });
          emit?.('chat:end', { ok: true });
        }
        
        // Emit quality status
        emitLog(emit, reflect.quality === 'excellent' || reflect.quality === 'good' ? 'success' : 'warning', 
                'REFLECT', `Quality: ${reflect.quality || 'unknown'} | Gate: ${reflect.proactiveScan?.qualityGate || 'n/a'}`);
      }

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

const formatBytes = (bytes) => {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const createAlert = async (db, userId, runId, level, message) => {
  await db.run(
    `INSERT INTO alerts (id, user_id, run_id, level, message, created_at)
     VALUES (?, ?, ?, ?, ?, ?)`
    , nanoid(16), userId, runId, level, message, new Date().toISOString()
  );
};
