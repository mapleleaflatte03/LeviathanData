import { WebSocketServer, WebSocket } from 'ws';
import jwt from 'jsonwebtoken';
import { config } from './config.js';
import { streamChatTokens, chatCompletionText, llmEvents, getLlmStats } from './llmClient.js';

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
