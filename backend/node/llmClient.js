import { config } from './config.js';

const MAX_RETRIES = 5;
const BASE_DELAY_MS = 500;

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

const buildHeaders = (apiKey) => ({
  'Content-Type': 'application/json',
  ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {})
});

const fetchChatOnce = async (payload, useFallback = false) => {
  const baseUrl = useFallback && config.llmFallback.baseUrl ? config.llmFallback.baseUrl : config.llm.baseUrl;
  const apiKey = useFallback && config.llmFallback.apiKey ? config.llmFallback.apiKey : config.llm.apiKey;
  if (!baseUrl) throw new Error('LLM base URL not configured');
  const url = `${baseUrl.replace(/\/$/, '')}/v1/chat/completions`;
  const res = await fetch(url, {
    method: 'POST',
    headers: buildHeaders(apiKey),
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`LLM error ${res.status}: ${text}`);
  }
  return res;
};

const fetchChatWithRetry = async (payload, useFallback = false) => {
  let lastErr;
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await fetchChatOnce(payload, useFallback);
    } catch (err) {
      lastErr = err;
      const waitMs = BASE_DELAY_MS * Math.pow(2, attempt - 1); // 500, 1000, 2000, 4000, 8000
      console.warn(`[LLM] Attempt ${attempt}/${MAX_RETRIES} failed${useFallback ? ' (fallback)' : ''}: ${err.message}. Retrying in ${waitMs}ms...`);
      if (attempt < MAX_RETRIES) await delay(waitMs);
    }
  }
  throw lastErr;
};

export const chatCompletion = async ({ messages, stream = false }) => {
  const model = config.llm.model || 'qwen3-32b';
  const payload = { model, messages, stream };
  try {
    return await fetchChatWithRetry(payload, false);
  } catch (err) {
    if (!config.llmFallback.baseUrl) throw err;
    console.warn(`[LLM] Primary exhausted after ${MAX_RETRIES} attempts, trying fallback...`);
    const fallbackModel = config.llmFallback.model || model;
    const fallbackPayload = { model: fallbackModel, messages, stream };
    return fetchChatWithRetry(fallbackPayload, true);
  }
};

export const streamChatTokens = async function* (messages) {
  const res = await chatCompletion({ messages, stream: true });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const data = trimmed.replace(/^data:\s*/, '');
      if (data === '[DONE]') return;
      try {
        const json = JSON.parse(data);
        const token = json.choices?.[0]?.delta?.content;
        if (token) yield token;
      } catch (err) {
        continue;
      }
    }
  }
};

export const chatCompletionText = async (messages) => {
  const res = await chatCompletion({ messages, stream: false });
  const json = await res.json();
  return json?.choices?.[0]?.message?.content || '';
};
