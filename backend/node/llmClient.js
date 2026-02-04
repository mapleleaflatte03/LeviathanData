import { config } from './config.js';

const buildHeaders = (apiKey) => ({
  'Content-Type': 'application/json',
  ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {})
});

const fetchChat = async (payload, useFallback = false) => {
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

export const chatCompletion = async ({ messages, stream = false }) => {
  const model = config.llm.model || 'qwen3-32b';
  const payload = { model, messages, stream };
  try {
    const res = await fetchChat(payload, false);
    return res;
  } catch (err) {
    if (!config.llmFallback.baseUrl) throw err;
    const fallbackModel = config.llmFallback.model || model;
    const fallbackPayload = { model: fallbackModel, messages, stream };
    return fetchChat(fallbackPayload, true);
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
