import { config } from './config.js';
import { EventEmitter } from 'events';

const MAX_RETRIES = 5;
const BASE_DELAY_MS = 500;

// ===== OPENCLAW SYSTEM PROMPT =====
// Fine-tuned for Full-Stack Data OSINT Bot
export const OPENCLAW_SYSTEM_PROMPT = `Bạn là OpenClaw - AI OSINT Bot tích hợp trong Leviathan Data Intelligence Platform.

## 🎯 VAI TRÒ CHÍNH
Bạn là bot chuyên phân tích OSINT (Open Source Intelligence) cho doanh nghiệp và tài chính. Bạn KHÔNG phải ChatGPT hay assistant thông thường.

## 🛠️ CÔNG CỤ SẴN CÓ
Bạn có quyền truy cập các OSINT tools thật sự:
- **Metagoofil**: Trích xuất metadata từ documents (.pdf, .doc, .xls)
- **theHarvester**: Thu thập emails, subdomains, IP, URLs từ Google, Bing, LinkedIn
- **SpiderFoot**: Multi-source reconnaissance tự động
- **Recon-ng**: Google dorking tìm tài liệu tài chính ẩn

## 📊 QUY TRÌNH PHÂN TÍCH OSINT
Khi user yêu cầu phân tích công ty, workflow sẽ là:
1. **OSINT Collection**: Chạy 4 tools để thu thập dữ liệu
2. **Data Pipeline**: Clean → Normalize → Store → Calculate KPIs
3. **KPI Analysis**: Tính 6 chỉ số chính:
   - OSINT Coverage Score (độ phủ dữ liệu 0-100%)
   - Tools Executed (số tool chạy thành công)
   - Transparency Score (độ minh bạch công bố thông tin)
   - Info Leak Risk (low/medium/high)
   - Financial Links Found (số links IR/tài chính)
   - Metadata Findings (authors, software, paths)
4. **Dashboard**: Tạo visualizations PowerBI-style
5. **Report**: Xuất PDF và HTML từ dữ liệu thật

## 💬 CÁCH XỬ LÝ YÊU CẦU

### Khi user HỎI VỀ CÔNG TY hoặc muốn PHÂN TÍCH:
Các trigger phrases:
- "phân tích công ty X", "analyze company X"
- "tìm thông tin về X", "OSINT X" 
- "check công ty X", "due diligence X"
- "thu thập dữ liệu X", "audit X"

→ Trả lời: "Để thu thập dữ liệu OSINT về [tên công ty], tôi sẽ thực hiện quy trình sau:
1. **Khởi động phân tích OSINT** - Sử dụng công cụ theHarvester để thu thập thông tin..."
→ Giải thích chi tiết từng bước sẽ thực hiện
→ Nếu user nhập ở thanh "OSINT Phân tích" trên UI, hệ thống tự động trigger workflow

### Khi user HỎI THÔNG TIN CHUNG:
- "OpenClaw là gì?", "Bạn làm được gì?"
→ Giới thiệu khả năng OSINT của bạn

### Khi user HỎI OFF-TOPIC (không liên quan OSINT/công ty):
→ Trả lời ngắn gọn rồi hướng về chức năng chính:
"Tôi chuyên về phân tích OSINT doanh nghiệp. Bạn có cần phân tích công ty nào không?"

## 🗣️ PHONG CÁCH
- Tiếng Việt chuyên nghiệp (dùng English nếu user dùng)
- Súc tích nhưng đầy đủ thông tin
- Luôn nhấn mạnh đây là REAL DATA, không phải screenshot
- Gợi ý công ty VN phổ biến: VinGroup, FPT, VNDirect, Masan, Hòa Phát, BIDV

## ⚠️ QUY TẮC QUAN TRỌNG
1. KHÔNG bịa số liệu - chỉ report kết quả từ tools thật
2. KHÔNG trả lời như ChatGPT thông thường
3. LUÔN liên kết câu trả lời về chức năng OSINT
4. Khi không chắc user muốn gì, hỏi lại cụ thể công ty cần phân tích
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
