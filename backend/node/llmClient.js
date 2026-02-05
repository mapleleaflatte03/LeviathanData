import { config } from './config.js';
import { EventEmitter } from 'events';

const MAX_RETRIES = 5;
const BASE_DELAY_MS = 500;

// ===== OPENCLAW SYSTEM PROMPT =====
// Fine-tuned for Full-Stack Data OSINT Bot
export const OPENCLAW_SYSTEM_PROMPT = `Bạn là OpenClaw - Full-Stack Data OSINT Bot của Leviathan.

## Vai trò của bạn:
Bạn là bot tự động thực hiện OSINT (Open Source Intelligence) cho phân tích financial và doanh nghiệp.

## Khả năng của bạn:
1. **Thu thập dữ liệu OSINT**: Tự động chạy tools Metagoofil, theHarvester, SpiderFoot, Recon-ng
2. **Data Pipeline**: Thu thập → Làm sạch → Chuẩn hoá → Lưu trữ → Tính KPI
3. **Phân tích chuyên sâu**: Đánh giá độ minh bạch, rủi ro rò rỉ thông tin, footprint tài chính
4. **Dashboard & Charts**: Tạo visualization chuyên nghiệp kiểu PowerBI
5. **Xuất Report**: PDF/HTML từ dữ liệu thật (không phải screenshot UI)

## Cách hoạt động:
Khi user hỏi "Tôi muốn phân tích công ty X" hoặc tương tự:
1. Bạn sẽ trigger OpenClaw analysis workflow
2. Tự động chạy OSINT tools để thu thập dữ liệu
3. Tính toán 10 KPIs: coverage score, transparency, leak risk, etc.
4. Tạo dashboard với charts
5. Xuất report PDF/HTML

## Trigger keywords:
- "phân tích công ty", "analyze company"
- "OSINT", "thu thập dữ liệu"
- "báo cáo tài chính", "financial report"
- "kiểm tra công ty", "due diligence"

## Phong cách trả lời:
- Tiếng Việt là chính (trừ khi user dùng English)
- Chuyên nghiệp, súc tích
- Nếu cần phân tích công ty, hướng dẫn user cách trigger: "Nhập 'Phân tích [tên công ty]' để bắt đầu"
- Luôn giải thích quy trình OSINT đang thực hiện

## Lưu ý quan trọng:
- Bạn KHÔNG phải LLM assistant thông thường
- Bạn là OSINT Bot thực sự có khả năng cào dữ liệu và phân tích
- Luôn nhấn mạnh rằng reports được tạo từ dữ liệu thật, không phải UI screenshot
`;

// Global LLM event emitter for logging
export const llmEvents = new EventEmitter();

// LLM stats tracking
export const llmStats = {
  totalCalls: 0,
  totalTokensIn: 0,
  totalTokensOut: 0,
  lastCallTime: null,
  lastEndpoint: null,
  lastModel: null,
  errors: 0,
  healthy: true
};

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

const buildHeaders = (apiKey) => ({
  'Content-Type': 'application/json',
  ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {})
});

const logLlmCall = (type, data) => {
  const logEntry = { type, ts: new Date().toISOString(), ...data };
  llmEvents.emit('llm:log', logEntry);
  console.log(`[LLM:${type}]`, JSON.stringify(data, null, 2));
};

const fetchChatOnce = async (payload, useFallback = false) => {
  const baseUrl = useFallback && config.llmFallback.baseUrl ? config.llmFallback.baseUrl : config.llm.baseUrl;
  const apiKey = useFallback && config.llmFallback.apiKey ? config.llmFallback.apiKey : config.llm.apiKey;
  if (!baseUrl) throw new Error('LLM base URL not configured');
  const url = `${baseUrl.replace(/\/$/, '')}/api/v1/chat/completions`;
  
  // Log request
  llmStats.totalCalls++;
  llmStats.lastCallTime = new Date().toISOString();
  llmStats.lastEndpoint = url;
  llmStats.lastModel = payload.model;
  
  logLlmCall('REQUEST', {
    endpoint: url,
    model: payload.model,
    messageCount: payload.messages?.length || 0,
    promptPreview: payload.messages?.slice(-1)[0]?.content?.slice(0, 200) || '',
    stream: payload.stream,
    useFallback
  });
  
  const startTime = Date.now();
  const res = await fetch(url, {
    method: 'POST',
    headers: buildHeaders(apiKey),
    body: JSON.stringify(payload)
  });
  
  if (!res.ok) {
    const text = await res.text();
    llmStats.errors++;
    llmStats.healthy = false;
    logLlmCall('ERROR', { status: res.status, error: text.slice(0, 500), latencyMs: Date.now() - startTime });
    throw new Error(`LLM error ${res.status}: ${text}`);
  }
  
  llmStats.healthy = true;
  return { res, startTime, url };
};

const fetchChatWithRetry = async (payload, useFallback = false) => {
  let lastErr;
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await fetchChatOnce(payload, useFallback);
    } catch (err) {
      lastErr = err;
      const waitMs = BASE_DELAY_MS * Math.pow(2, attempt - 1);
      logLlmCall('RETRY', { attempt, maxRetries: MAX_RETRIES, waitMs, error: err.message, useFallback });
      if (attempt < MAX_RETRIES) await delay(waitMs);
    }
  }
  throw lastErr;
};

export const chatCompletion = async ({ messages, stream = false }) => {
  const model = config.llm.model || 'qwen3-32b';
  const payload = { model, messages, stream };
  try {
    const { res, startTime, url } = await fetchChatWithRetry(payload, false);
    return { res, startTime, url, model };
  } catch (err) {
    if (!config.llmFallback.baseUrl) throw err;
    logLlmCall('FALLBACK', { reason: `Primary exhausted after ${MAX_RETRIES} attempts` });
    const fallbackModel = config.llmFallback.model || model;
    const fallbackPayload = { model: fallbackModel, messages, stream };
    const { res, startTime, url } = await fetchChatWithRetry(fallbackPayload, true);
    return { res, startTime, url, model: fallbackModel };
  }
};

// Inject OpenClaw system prompt into messages if not present
export const injectSystemPrompt = (messages) => {
  if (!messages || messages.length === 0) {
    return [{ role: 'system', content: OPENCLAW_SYSTEM_PROMPT }];
  }
  
  // Check if system prompt already exists
  if (messages[0]?.role === 'system') {
    return messages;
  }
  
  // Inject system prompt at the beginning
  return [{ role: 'system', content: OPENCLAW_SYSTEM_PROMPT }, ...messages];
};

export const streamChatTokens = async function* (messages) {
  const messagesWithPrompt = injectSystemPrompt(messages);
  const { res, startTime, url, model } = await chatCompletion({ messages: messagesWithPrompt, stream: true });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let tokenCount = 0;
  let fullText = '';
  
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
      if (data === '[DONE]') {
        llmStats.totalTokensOut += tokenCount;
        logLlmCall('STREAM_END', {
          endpoint: url,
          model,
          tokenCount,
          latencyMs: Date.now() - startTime,
          responsePreview: fullText.slice(0, 300)
        });
        return;
      }
      try {
        const json = JSON.parse(data);
        const token = json.choices?.[0]?.delta?.content;
        if (token) {
          tokenCount++;
          fullText += token;
          yield token;
        }
      } catch (err) {
        continue;
      }
    }
  }
  
  llmStats.totalTokensOut += tokenCount;
  logLlmCall('STREAM_END', {
    endpoint: url,
    model,
    tokenCount,
    latencyMs: Date.now() - startTime,
    responsePreview: fullText.slice(0, 300)
  });
};

export const chatCompletionText = async (messages) => {
  const messagesWithPrompt = injectSystemPrompt(messages);
  const { res, startTime, url, model } = await chatCompletion({ messages: messagesWithPrompt, stream: false });
  const json = await res.json();
  const content = json?.choices?.[0]?.message?.content || '';
  const usage = json?.usage || {};
  
  llmStats.totalTokensIn += usage.prompt_tokens || 0;
  llmStats.totalTokensOut += usage.completion_tokens || 0;
  
  logLlmCall('RESPONSE', {
    endpoint: url,
    model,
    tokensIn: usage.prompt_tokens || 0,
    tokensOut: usage.completion_tokens || 0,
    totalTokens: usage.total_tokens || 0,
    latencyMs: Date.now() - startTime,
    responsePreview: content.slice(0, 300)
  });
  
  return content;
};

export const getLlmStats = () => ({ ...llmStats });
