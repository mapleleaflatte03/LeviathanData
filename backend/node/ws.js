import { WebSocketServer, WebSocket } from 'ws';
import jwt from 'jsonwebtoken';
import { config } from './config.js';
import { streamChatTokens, chatCompletionText } from './llmClient.js';

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
  const wss = new WebSocketServer({ server });
  const clients = new Map();

  const addClient = (ws) => {
    clients.set(ws, { userId: null, authed: false });
  };

  const removeClient = (ws) => clients.delete(ws);

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

  wss.on('connection', (ws) => {
    addClient(ws);

    ws.on('message', async (data) => {
      const msg = parseMessage(data);
      if (!msg?.type) return;
      const meta = clients.get(ws);

      if (msg.type === 'auth') {
        try {
          const payload = jwt.verify(msg.payload?.token || '', config.jwtSecret);
          meta.userId = payload.sub;
          meta.authed = true;
          send(ws, 'auth:ok', msg.requestId, { userId: meta.userId });
        } catch (err) {
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
          send(ws, 'error', msg.requestId, { message: err.message });
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

    ws.on('close', () => removeClient(ws));
  });

  return {
    broadcastToUser,
    emitToUser: (userId, type, payload) => broadcastToUser(userId, type, payload)
  };
};
