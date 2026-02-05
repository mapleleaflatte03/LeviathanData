import { WebSocketServer, WebSocket } from 'ws';
import jwt from 'jsonwebtoken';
import { config } from './config.js';
import { streamChatTokens, chatCompletionText, llmEvents, getLlmStats } from './llmClient.js';
import { callPython } from './pythonBridge.js';

const parseMessage = (data) => {
  try {
    return JSON.parse(data.toString());
  } catch (err) {
    return null;
  }
};

const envelope = (type, requestId, payload) => ({
  type,
  requestId: requestId || 'system',
  ts: new Date().toISOString(),
  payload
});

export const initWebsocket = (server, log) => {
  const wss = new WebSocketServer({ server, path: '/ws' });
  const clients = new Map();

  log?.info('WebSocket server initialized on /ws');

  const addClient = (ws) => {
    clients.set(ws, { userId: null, authed: false });
    log?.info({ clientCount: clients.size }, 'WebSocket client connected');
  };

  const removeClient = (ws) => {
    clients.delete(ws);
    log?.info({ clientCount: clients.size }, 'WebSocket client disconnected');
  };

  const send = (ws, type, requestId, payload) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(envelope(type, requestId, payload)));
  };

  const broadcastToUser = (userId, type, payload) => {
    for (const [ws, meta] of clients.entries()) {
      if (meta.userId === userId && meta.authed) {
        send(ws, type, 'system', payload);
      }
    }
  };

  const broadcastToAll = (type, payload) => {
    for (const [ws, meta] of clients.entries()) {
      if (meta.authed) {
        send(ws, type, 'system', payload);
      }
    }
  };

  // Listen to LLM events and broadcast to all authenticated clients
  llmEvents.on('llm:log', (logEntry) => {
    broadcastToAll('llm:log', logEntry);
  });

  wss.on('connection', (ws) => {
    addClient(ws);

    ws.on('message', async (data) => {
      const msg = parseMessage(data);
      if (!msg?.type) return;
      const meta = clients.get(ws);

      if (msg.type === 'auth') {
        try {
          const token = msg.payload?.token || '';
          log?.info({ hasToken: !!token, tokenLength: token.length }, 'WS auth attempt');
          const payload = jwt.verify(token, config.jwtSecret);
          meta.userId = payload.sub;
          meta.authed = true;
          send(ws, 'auth:ok', msg.requestId, { userId: meta.userId });
          log?.info({ userId: meta.userId }, 'WS auth success');
        } catch (err) {
          log?.error({ err: err.message }, 'WS auth failed');
          send(ws, 'error', msg.requestId, { message: 'auth failed' });
        }
        return;
      }

      if (!meta?.authed) {
        send(ws, 'error', msg.requestId, { message: 'unauthorized' });
        return;
      }

      if (msg.type === 'chat:request') {
        const messages = msg.payload?.messages || [];
        try {
          const stream = msg.payload?.stream !== false;
          if (stream) {
            for await (const token of streamChatTokens(messages)) {
              send(ws, 'chat:token', msg.requestId, { token });
            }
            send(ws, 'chat:end', msg.requestId, { ok: true });
          } else {
            const text = await chatCompletionText(messages);
            send(ws, 'chat:token', msg.requestId, { token: text });
            send(ws, 'chat:end', msg.requestId, { ok: true });
          }
        } catch (err) {
          log?.error({ err }, 'chat stream failed');
          // No dummy fallback - show real error to user
          send(ws, 'chat:end', msg.requestId, { ok: false, error: err.message });
          send(ws, 'error', msg.requestId, { message: `LLM Error: ${err.message}` });
        }
      }

      if (msg.type === 'llm:status') {
        send(ws, 'llm:stats', msg.requestId, getLlmStats());
      }

      // Autonomous hunt handler
      if (msg.type === 'hunt:request') {
        const prompt = msg.payload?.prompt || '';
        const messages = msg.payload?.messages || [];
        
        try {
          // Emit progress start
          send(ws, 'hunt:progress', msg.requestId, { 
            stage: 'INIT', 
            message: 'Starting autonomous hunt...' 
          });
          
          // Call Python hunt endpoint
          const result = await callPython('/hunt', { 
            prompt,
            messages: messages.slice(-5)  // Send last 5 messages for context
          });
          
          // Stream response if available
          if (result.summary) {
            for (const token of result.summary.split(' ')) {
              send(ws, 'hunt:token', msg.requestId, { token: token + ' ' });
              await new Promise(r => setTimeout(r, 30));  // Simulate streaming
            }
          }
          
          // Send completion
          send(ws, 'hunt:complete', msg.requestId, {
            itemsCollected: result.items_collected || 0,
            sources: result.sources || [],
            chartData: result.chart_data || null
          });
          send(ws, 'chat:end', msg.requestId, { ok: true });
          
        } catch (err) {
          log?.error({ err }, 'hunt failed');
          send(ws, 'hunt:error', msg.requestId, { 
            error: err.message,
            partial: false
          });
          send(ws, 'chat:end', msg.requestId, { ok: false, error: err.message });
        }
      }

      // OpenClaw Full-Stack OSINT Company Analysis handler
      if (msg.type === 'openclaw:analyze') {
        const companyName = msg.payload?.company_name || msg.payload?.companyName || '';
        const prompt = msg.payload?.prompt || '';
        const language = msg.payload?.language || 'vi';
        
        if (!companyName) {
          send(ws, 'openclaw:error', msg.requestId, { 
            error: 'Company name is required for analysis'
          });
          return;
        }
        
        try {
          // Stage 1: OSINT
          send(ws, 'openclaw:progress', msg.requestId, { 
            stage: 'OSINT', 
            message: `Đang thu thập dữ liệu OSINT cho ${companyName}...`,
            percent: 10
          });
          
          // Call Python OpenClaw analyze endpoint
          const result = await callPython('/openclaw/analyze', { 
            company_name: companyName,
            prompt,
            language,
            export_pdf: true,
            export_html: true
          });
          
          // Stage 2: Progress updates from execution log
          const stages = ['OSINT', 'PIPELINE', 'LLM', 'DASHBOARD', 'REPORT'];
          let stageIndex = 0;
          
          if (result.execution_log) {
            for (const logEntry of result.execution_log) {
              for (const stage of stages) {
                if (logEntry.includes(`[${stage}]`)) {
                  stageIndex = stages.indexOf(stage);
                  break;
                }
              }
              send(ws, 'openclaw:progress', msg.requestId, { 
                stage: stages[stageIndex] || 'PROCESSING', 
                message: logEntry,
                percent: Math.min(20 + (stageIndex * 16), 95)
              });
            }
          }
          
          // Stream analysis text if available
          if (result.analysis) {
            send(ws, 'openclaw:progress', msg.requestId, { 
              stage: 'OUTPUT', 
              message: 'Đang xuất phân tích...',
              percent: 95
            });
            
            const words = result.analysis.split(' ');
            for (const word of words) {
              send(ws, 'openclaw:token', msg.requestId, { token: word + ' ' });
              await new Promise(r => setTimeout(r, 25));  // Streaming effect
            }
          }
          
          // Send completion with all data
          send(ws, 'openclaw:complete', msg.requestId, {
            success: result.success || true,
            company: companyName,
            osintToolsUsed: result.osint_tools_used || [],
            osintItemsCollected: result.osint_items_collected || 0,
            osintData: result.osint_data || {},
            kpis: result.kpis || {},
            analysis: result.analysis || '',
            dashboardData: result.dashboard_data || {},
            reportPdfPath: result.report_pdf_path || null,
            reportHtmlPath: result.report_html_path || null,
            timestamp: result.timestamp || new Date().toISOString()
          });
          send(ws, 'chat:end', msg.requestId, { ok: true });
          
        } catch (err) {
          log?.error({ err }, 'OpenClaw analysis failed');
          send(ws, 'openclaw:error', msg.requestId, { 
            error: err.message,
            company: companyName
          });
          send(ws, 'chat:end', msg.requestId, { ok: false, error: err.message });
        }
      }

      if (msg.type === 'pipeline:subscribe') {
        send(ws, 'pipeline:status', msg.requestId, {
          runId: msg.payload?.runId || 'unknown',
          stage: 'subscribe',
          status: 'ok',
          message: 'Subscribed to pipeline updates'
        });
      }
    });

    ws.on('close', (code, reason) => {
      log?.info({ code, reason: reason?.toString() }, 'WS close');
      removeClient(ws);
    });
    
    ws.on('error', (err) => {
      log?.error({ err: err.message }, 'WS error');
    });
  });

  return {
    broadcastToUser,
    emitToUser: (userId, type, payload) => broadcastToUser(userId, type, payload)
  };
};
