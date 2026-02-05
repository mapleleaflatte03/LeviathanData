/**
 * Leviathan Frontend - Ultimate UX
 * Real-time streaming, monster immersion, full interactivity
 */

// ===== DOM ELEMENTS =====
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const authPanel = $('#authPanel');
const dashboardPanel = $('#dashboardPanel');
const authMessage = $('#authMessage');
const statusEl = $('#status');
const chatLog = $('#chatLog');
const chatInput = $('#chatInput');
const chatSend = $('#chatSend');
const typingIndicator = $('#typingIndicator');
const uploadZone = $('#uploadZone');
const fileInput = $('#fileInput');
const pipelineProgress = $('#pipelineProgress');
const swarmConsole = $('#swarmConsole');
const alertsList = $('#alertsList');
const alertBadge = $('#alertBadge');
const alertModal = $('#alertModal');
const alertModalBody = $('#alertModalBody');
const modalClose = $('#modalClose');
const dismissAlert = $('#dismissAlert');
const autoDismissCheck = $('#autoDismiss');
const toastContainer = $('#toastContainer');
const plotlyChart = $('#plotlyChart');
const chartTabs = $('#chartTabs');
const monsterEye = $('#monsterEye');
const tentacles = $('#tentacles');
const brandTitle = $('#brandTitle');
const alertSound = $('#alertSound');
const logoutBtn = $('#logout');

// LLM Status Panel elements
const llmStatus = $('#llmStatus');
const llmTokens = $('#llmTokens');
const llmToggle = $('#llmToggle');
const llmLogPanel = $('#llmLogPanel');
const llmLogClose = $('#llmLogClose');
const llmLogEntries = $('#llmLogEntries');
const llmCallCount = $('#llmCallCount');
const llmTokensIn = $('#llmTokensIn');
const llmTokensOut = $('#llmTokensOut');
const llmEndpoint = $('#llmEndpoint');

// ===== STATE =====
const state = {
  accessToken: localStorage.getItem('accessToken') || '',
  refreshToken: localStorage.getItem('refreshToken') || '',
  userId: localStorage.getItem('userId') || '',
  ws: null,
  chatMessages: [],
  currentBotEl: null,
  streamingText: '',
  alertCount: 0,
  currentChart: 'main',
  chartData: {},
  pipelineRunId: null,
  autoDismissTimer: null,
  llmStats: {
    totalCalls: 0,
    totalTokensIn: 0,
    totalTokensOut: 0,
    lastEndpoint: null,
    healthy: true
  },
  llmLogVisible: false
};

// ===== UTILITIES =====
const getTimestamp = () => {
  const now = new Date();
  return `[${now.toTimeString().slice(0, 8)}]`;
};

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ===== TOAST NOTIFICATIONS =====
const showToast = (message, type = 'info', duration = 4000) => {
  const icons = { error: '❌', success: '✅', warning: '⚠️', info: 'ℹ️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type]}</span>
    <span class="toast-message">${message}</span>
    <button class="toast-close">×</button>
  `;
  toastContainer.appendChild(toast);
  
  toast.querySelector('.toast-close').onclick = () => toast.remove();
  setTimeout(() => toast.remove(), duration);
};

// ===== STATUS =====
const setStatus = (text, connected = false) => {
  statusEl.querySelector('.status-text').textContent = text;
  statusEl.classList.toggle('connected', connected);
};

// ===== AUTH FLOW =====
const showDashboard = () => {
  authPanel.style.display = 'none';
  dashboardPanel.classList.add('active');
  initCharts();
};

const showAuth = () => {
  authPanel.style.display = 'block';
  dashboardPanel.classList.remove('active');
};

// ===== API FETCH =====
const apiFetch = async (url, options = {}) => {
  const headers = { ...options.headers };
  if (state.accessToken) {
    headers.Authorization = `Bearer ${state.accessToken}`;
  }
  try {
    const res = await fetch(url, { ...options, headers });
    if (res.status === 401 && state.refreshToken && state.userId) {
      const refreshed = await refreshTokens();
      if (refreshed) {
        headers.Authorization = `Bearer ${state.accessToken}`;
        return fetch(url, { ...options, headers });
      }
    }
    return res;
  } catch (err) {
    showToast('Network error: ' + err.message, 'error');
    throw err;
  }
};

const setTokens = (data) => {
  state.accessToken = data.accessToken;
  state.refreshToken = data.refreshToken;
  state.userId = data.user?.id || data.userId || state.userId;
  localStorage.setItem('accessToken', state.accessToken);
  localStorage.setItem('refreshToken', state.refreshToken);
  localStorage.setItem('userId', state.userId);
};

const refreshTokens = async () => {
  try {
    const res = await fetch('/api/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userId: state.userId, refreshToken: state.refreshToken })
    });
    if (!res.ok) return false;
    const data = await res.json();
    setTokens(data);
    return true;
  } catch {
    return false;
  }
};

// ===== WEBSOCKET =====
const connectWs = () => {
  if (state.ws) state.ws.close();
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${protocol}://${location.host}/ws`;
  console.log('[WS] Connecting to:', wsUrl);
  const ws = new WebSocket(wsUrl);
  state.ws = ws;

  ws.onopen = () => {
    setStatus('Connected', true);
    ws.send(JSON.stringify({ type: 'auth', requestId: 'auth', payload: { token: state.accessToken } }));
    logToConsole('info', 'SYSTEM', 'WebSocket connected');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleWsMessage(msg);
    } catch (err) {
      console.error('WS parse error:', err);
    }
  };

  ws.onclose = () => {
    setStatus('Disconnected', false);
    logToConsole('error', 'SYSTEM', 'Connection lost. Reconnecting...');
    setTimeout(connectWs, 3000);
  };

  ws.onerror = () => {
    showToast('WebSocket error', 'error');
  };
};

const handleWsMessage = (msg) => {
  switch (msg.type) {
    case 'chat:token':
      handleChatToken(msg.payload.token);
      break;
    case 'chat:end':
      finishStreaming();
      break;
    case 'hunt:token':
      handleChatToken(msg.payload.token);
      break;
    case 'hunt:progress':
      handleHuntProgress(msg.payload);
      break;
    case 'hunt:complete':
      handleHuntComplete(msg.payload);
      break;
    case 'hunt:error':
      handleHuntError(msg.payload);
      break;
    case 'pipeline:status':
      updatePipeline(msg.payload);
      break;
    case 'pipeline:chart':
      updateChartData(msg.payload);
      break;
    case 'pipeline:insight':
      handleInsight(msg.payload);
      break;
    case 'alert:new':
      pushAlert(msg.payload);
      break;
    case 'swarm:log':
      logToConsole(msg.payload.level, msg.payload.stage, msg.payload.message);
      break;
    case 'llm:log':
      handleLlmLog(msg.payload);
      break;
    case 'llm:stats':
      updateLlmStats(msg.payload);
      break;
  }
};

// ===== HUNT HANDLERS =====
const handleHuntProgress = (payload) => {
  const { stage, message, source, itemsFound } = payload;
  logToConsole('hunt', stage?.toUpperCase() || 'HUNT', message || 'Hunting...');
  
  if (source) {
    logToConsole('info', 'SOURCE', `Crawling: ${source}`);
  }
  if (itemsFound) {
    logToConsole('success', 'DATA', `Found ${itemsFound} items`);
  }
};

const handleHuntComplete = (payload) => {
  endHunt(true);
  finishStreaming();
  
  const { itemsCollected, sources, chartData } = payload;
  logToConsole('success', 'HUNT', `Hunt complete! Collected ${itemsCollected || 0} items from ${sources?.length || 0} sources`);
  
  if (chartData) {
    updateChartData(chartData);
  }
  
  showToast(`Hunt complete: ${itemsCollected || 0} items collected`, 'success');
};

const handleHuntError = (payload) => {
  endHunt(false);
  finishStreaming();
  
  const { error, partial } = payload;
  logToConsole('error', 'HUNT', error || 'Hunt failed');
  
  if (partial) {
    showToast(`Partial hunt: ${error}`, 'warning');
  } else {
    showToast(`Hunt failed: ${error}`, 'error');
  }
};

// ===== LLM LOG HANDLING =====
const handleLlmLog = (logEntry) => {
  // Update stats
  if (logEntry.type === 'RESPONSE' || logEntry.type === 'STREAM_END') {
    state.llmStats.totalCalls++;
    state.llmStats.totalTokensIn += logEntry.tokensIn || 0;
    state.llmStats.totalTokensOut += logEntry.tokensOut || logEntry.tokenCount || 0;
    state.llmStats.lastEndpoint = logEntry.endpoint || state.llmStats.lastEndpoint;
    state.llmStats.healthy = true;
    updateLlmUI();
  } else if (logEntry.type === 'ERROR') {
    state.llmStats.healthy = false;
    updateLlmUI();
  } else if (logEntry.type === 'REQUEST') {
    state.llmStats.lastEndpoint = logEntry.endpoint;
    updateLlmUI();
  }
  
  // Add log entry to panel
  addLlmLogEntry(logEntry);
  
  // Also log to swarm console
  const level = logEntry.type === 'ERROR' ? 'error' : (logEntry.type === 'RESPONSE' ? 'success' : 'info');
  const preview = logEntry.responsePreview || logEntry.promptPreview || logEntry.error || '';
  logToConsole(level, `LLM:${logEntry.type}`, preview.slice(0, 100) + (preview.length > 100 ? '...' : ''));
};

const addLlmLogEntry = (logEntry) => {
  if (!llmLogEntries) return;
  
  const entry = document.createElement('div');
  entry.className = `llm-log-entry ${logEntry.type}`;
  
  const time = logEntry.ts ? new Date(logEntry.ts).toLocaleTimeString() : new Date().toLocaleTimeString();
  const details = [];
  
  if (logEntry.model) details.push(`model=${logEntry.model}`);
  if (logEntry.tokensIn) details.push(`in=${logEntry.tokensIn}`);
  if (logEntry.tokensOut) details.push(`out=${logEntry.tokensOut}`);
  if (logEntry.tokenCount) details.push(`tokens=${logEntry.tokenCount}`);
  if (logEntry.latencyMs) details.push(`${logEntry.latencyMs}ms`);
  if (logEntry.error) details.push(`err: ${logEntry.error.slice(0, 50)}`);
  
  const preview = logEntry.responsePreview || logEntry.promptPreview || '';
  
  entry.innerHTML = `
    <span class="log-time">${time}</span>
    <span class="log-type">${logEntry.type}</span>
    ${details.length ? `<span>${details.join(' | ')}</span>` : ''}
    ${preview ? `<div style="margin-top:4px;opacity:0.8">${escapeHtml(preview.slice(0, 150))}...</div>` : ''}
  `;
  
  llmLogEntries.prepend(entry);
  
  // Limit entries
  while (llmLogEntries.children.length > 50) {
    llmLogEntries.removeChild(llmLogEntries.lastChild);
  }
};

const updateLlmStats = (stats) => {
  state.llmStats = { ...state.llmStats, ...stats };
  updateLlmUI();
};

const updateLlmUI = () => {
  if (llmTokens) {
    const total = state.llmStats.totalTokensIn + state.llmStats.totalTokensOut;
    llmTokens.textContent = `${total.toLocaleString()} tok`;
  }
  
  if (llmStatus) {
    llmStatus.classList.toggle('healthy', state.llmStats.healthy);
    llmStatus.classList.toggle('error', !state.llmStats.healthy);
  }
  
  if (llmCallCount) llmCallCount.textContent = state.llmStats.totalCalls;
  if (llmTokensIn) llmTokensIn.textContent = state.llmStats.totalTokensIn.toLocaleString();
  if (llmTokensOut) llmTokensOut.textContent = state.llmStats.totalTokensOut.toLocaleString();
  if (llmEndpoint && state.llmStats.lastEndpoint) {
    const url = state.llmStats.lastEndpoint;
    llmEndpoint.textContent = url.length > 30 ? '...' + url.slice(-30) : url;
  }
};

const toggleLlmLogPanel = () => {
  state.llmLogVisible = !state.llmLogVisible;
  if (llmLogPanel) {
    llmLogPanel.classList.toggle('active', state.llmLogVisible);
  }
};

const handleInsight = (payload) => {
  if (payload.insight) {
    logToConsole('info', 'INSIGHT', payload.insight);
  }
};

// ===== CHAT =====
const appendChat = (role, content) => {
  const entry = document.createElement('div');
  entry.className = `chat-entry ${role}`;
  
  if (role === 'user') {
    entry.innerHTML = `
      <div class="avatar user-avatar">U</div>
      <div class="message">
        <span class="sender">You</span>
        <p>${escapeHtml(content)}</p>
      </div>
    `;
  } else {
    entry.innerHTML = `
      <div class="avatar agent-avatar">
        <div class="mini-eye"></div>
      </div>
      <div class="message">
        <span class="sender">Leviathan</span>
        <p></p>
      </div>
    `;
    state.currentBotEl = entry.querySelector('p');
  }
  
  chatLog.appendChild(entry);
  chatLog.scrollTop = chatLog.scrollHeight;
  triggerRipple();
  return entry;
};

const handleChatToken = (token) => {
  if (!state.currentBotEl) {
    appendChat('bot', '');
  }
  state.streamingText += token;
  state.currentBotEl.textContent = state.streamingText;
  chatLog.scrollTop = chatLog.scrollHeight;
  typingIndicator.classList.remove('active');
};

const finishStreaming = () => {
  state.currentBotEl = null;
  state.streamingText = '';
  typingIndicator.classList.remove('active');
};

const sendChat = () => {
  const text = chatInput.value.trim();
  if (!text || !state.ws) return;
  
  appendChat('user', text);
  state.chatMessages.push({ role: 'user', content: text });
  
  typingIndicator.classList.add('active');
  activateTentacles();
  intensifyEye();
  
  state.ws.send(JSON.stringify({
    type: 'chat:request',
    requestId: crypto.randomUUID(),
    payload: { messages: state.chatMessages, stream: true }
  }));
  
  chatInput.value = '';
};

const escapeHtml = (text) => {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
};

// ===== UPLOAD & PIPELINE =====
const setupUpload = () => {
  uploadZone.onclick = () => fileInput.click();
  fileInput.onchange = handleFileSelect;
  
  uploadZone.ondragover = (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  };
  
  uploadZone.ondragleave = () => {
    uploadZone.classList.remove('dragover');
  };
  
  uploadZone.ondrop = (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length) uploadFiles(files);
  };
};

const handleFileSelect = () => {
  if (fileInput.files.length) {
    uploadFiles(fileInput.files);
  }
};

const uploadFiles = async (files) => {
  activateTentacles();
  intensifyEye();
  logToConsole('ingest', 'INGEST', `Uploading ${files.length} file(s)...`);
  
  for (const file of files) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      resetPipelineUI();
      setPipelineStage('ingest', 'active');
      
      const res = await apiFetch('/api/files/upload', { method: 'POST', body: formData });
      const data = await res.json();
      
      if (!res.ok) {
        showToast(data.error || 'Upload failed', 'error');
        setPipelineStage('ingest', 'error');
        logToConsole('error', 'INGEST', `Failed: ${data.error}`);
        continue;
      }
      
      state.pipelineRunId = data.runId;
      logToConsole('success', 'INGEST', `File "${file.name}" queued → Run ${data.runId}`);
      showToast(`Uploaded: ${file.name}`, 'success');
      
    } catch (err) {
      showToast('Upload error: ' + err.message, 'error');
      setPipelineStage('ingest', 'error');
    }
  }
};

const resetPipelineUI = () => {
  $$('.pipeline-stage').forEach(s => {
    s.classList.remove('active', 'complete', 'error');
    s.querySelector('.stage-fill').style.width = '0%';
  });
};

const setPipelineStage = (stage, status) => {
  const el = $(`.pipeline-stage[data-stage="${stage}"]`);
  if (!el) return;
  
  el.classList.remove('active', 'complete', 'error');
  if (status) el.classList.add(status);
  
  if (status === 'complete') {
    el.querySelector('.stage-fill').style.width = '100%';
  }
};

const updatePipeline = (payload) => {
  const { stage, status, message, runId } = payload;
  
  if (status === 'running') {
    setPipelineStage(stage, 'active');
    logToConsole(stage, stage.toUpperCase(), message || `${stage} in progress...`);
  } else if (status === 'complete') {
    setPipelineStage(stage, 'complete');
    logToConsole('success', stage.toUpperCase(), message || `${stage} complete`);
  } else if (status === 'error') {
    setPipelineStage(stage, 'error');
    logToConsole('error', stage.toUpperCase(), message || `${stage} failed`);
    showToast(`Pipeline error: ${message}`, 'error');
  }
  
  // Map stage names
  const stages = ['ingest', 'analyze', 'visualize', 'reflect'];
  const stageNames = { ingest: 'ingest', analyze: 'analyze', viz: 'visualize', visualize: 'visualize', reflect: 'reflect' };
  const normalized = stageNames[stage] || stage;
  
  if (normalized === 'reflect' && status === 'complete') {
    deactivateTentacles();
    calmEye();
    showToast('Pipeline complete!', 'success');
  }
};

// ===== SWARM CONSOLE =====
const logToConsole = (level, stage, message) => {
  const entry = document.createElement('div');
  entry.className = `console-entry ${level}`;
  entry.innerHTML = `
    <span class="timestamp">${getTimestamp()}</span>
    <span class="stage-tag">${stage}</span>
    <span class="log-message">${escapeHtml(message)}</span>
  `;
  swarmConsole.appendChild(entry);
  swarmConsole.scrollTop = swarmConsole.scrollHeight;
  
  // Keep max 100 entries
  while (swarmConsole.children.length > 100) {
    swarmConsole.removeChild(swarmConsole.firstChild);
  }
};

// ===== ALERTS =====
const pushAlert = (alert) => {
  state.alertCount++;
  alertBadge.textContent = state.alertCount;
  
  const item = document.createElement('div');
  item.className = `alert-item ${alert.level || 'error'}`;
  item.innerHTML = `
    <div class="alert-level">${alert.level || 'ALERT'}</div>
    <div class="alert-message">${escapeHtml(alert.message)}</div>
    <div class="alert-time">${new Date().toLocaleTimeString()}</div>
  `;
  item.onclick = () => showAlertModal(alert);
  alertsList.prepend(item);
  
  // Play sound
  try {
    alertSound.currentTime = 0;
    alertSound.volume = 0.3;
    alertSound.play().catch(() => {});
  } catch {}
  
  // Show modal for critical
  if (alert.level === 'critical' || alert.level === 'error') {
    showAlertModal(alert);
  }
  
  intensifyEye();
  logToConsole('error', 'ALERT', alert.message);
};

const showAlertModal = (alert) => {
  alertModalBody.innerHTML = `
    <p><strong>Level:</strong> ${alert.level || 'Unknown'}</p>
    <p><strong>Message:</strong> ${escapeHtml(alert.message)}</p>
    <p><strong>Time:</strong> ${new Date().toLocaleString()}</p>
    ${alert.details ? `<p><strong>Details:</strong> ${escapeHtml(alert.details)}</p>` : ''}
  `;
  alertModal.classList.add('active');
  
  if (autoDismissCheck.checked) {
    state.autoDismissTimer = setTimeout(hideAlertModal, 10000);
  }
};

const hideAlertModal = () => {
  alertModal.classList.remove('active');
  if (state.autoDismissTimer) {
    clearTimeout(state.autoDismissTimer);
    state.autoDismissTimer = null;
  }
};

const loadAlerts = async () => {
  try {
    const res = await apiFetch('/api/alerts');
    if (!res.ok) return;
    const data = await res.json();
    alertsList.innerHTML = '';
    state.alertCount = 0;
    (data.alerts || []).forEach(a => pushAlert(a));
  } catch {}
};

// ===== CHARTS =====
const initCharts = () => {
  // Show placeholder until real data arrives
  renderChart('main');
};

const renderChart = (type) => {
  state.currentChart = type;
  
  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,15,30,0.5)',
    font: { color: '#d8f3ff', family: 'Space Grotesk' },
    margin: { t: 30, r: 20, b: 40, l: 50 },
    xaxis: { gridcolor: 'rgba(0,136,170,0.2)', zerolinecolor: 'rgba(0,136,170,0.3)' },
    yaxis: { gridcolor: 'rgba(0,136,170,0.2)', zerolinecolor: 'rgba(0,136,170,0.3)' },
    dragmode: 'pan'
  };
  
  const config = {
    responsive: true,
    displayModeBar: false,
    scrollZoom: true
  };
  
  let data;
  
  // Handle distribution and correlation views by deriving from main data
  if (type === 'distribution' || type === 'correlation' || type === 'forecast' || type === 'custom') {
    try {
      const mainData = state.chartData['main'];
      if (!mainData || (!mainData.actual && !mainData.y && !mainData.values)) {
        // No data available
        data = [{
          x: ['Start a Hunt'],
          y: [0],
          type: 'bar',
          marker: { color: 'rgba(0,136,170,0.3)' }
        }];
        layout.annotations = [{
          text: type === 'forecast' ? 
            'Hunt for time-series data to see forecasts' :
            type === 'custom' ? 
            'Hunt for data, then customize your visualization' :
            'Hunt for data to see ' + type + ' analysis',
          showarrow: false,
          x: 0.5,
          y: 0.5,
          xref: 'paper',
          yref: 'paper',
          font: { size: 14, color: 'rgba(216,243,255,0.5)' }
        }];
        Plotly.newPlot('plotlyChart', data, layout, config);
        return;
      }
      
      if (type === 'forecast') {
        // Generate simple forecast from available data
        const actual = mainData.actual || mainData.y || mainData.values || [];
        const labels = mainData.x || mainData.labels || actual.map((_, i) => i + 1);
        
        if (actual.length < 3) {
          throw new Error('Need at least 3 data points for forecast');
        }
        
        // Simple moving average forecast
        const windowSize = Math.min(5, Math.floor(actual.length / 2));
        const lastValues = actual.slice(-windowSize);
        const avgGrowth = (lastValues[lastValues.length - 1] - lastValues[0]) / (windowSize - 1);
        const lastValue = actual[actual.length - 1];
        
        // Generate 5 forecast points
        const forecastPoints = [];
        const forecastLabels = [];
        for (let i = 1; i <= 5; i++) {
          forecastPoints.push(lastValue + avgGrowth * i);
          const label = typeof labels[0] === 'string' ? 
            `Forecast ${i}` : 
            (labels[labels.length - 1] || actual.length) + i;
          forecastLabels.push(label);
        }
        
        // Confidence bands (±10%)
        const upperBand = forecastPoints.map(v => v * 1.1);
        const lowerBand = forecastPoints.map(v => v * 0.9);
        
        data = [
          {
            x: labels,
            y: actual,
            name: 'Historical',
            type: 'scatter',
            mode: 'lines+markers',
            marker: { color: '#0088aa' },
            line: { color: '#0088aa', width: 2 }
          },
          {
            x: forecastLabels,
            y: forecastPoints,
            name: 'Forecast',
            type: 'scatter',
            mode: 'lines+markers',
            marker: { color: '#ff9900', symbol: 'diamond' },
            line: { color: '#ff9900', width: 2, dash: 'dot' }
          },
          {
            x: forecastLabels,
            y: upperBand,
            name: 'Upper 90%',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#ff990055', width: 0 },
            showlegend: false
          },
          {
            x: forecastLabels,
            y: lowerBand,
            name: 'Lower 90%',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#ff990055', width: 0 },
            fill: 'tonexty',
            fillcolor: 'rgba(255,153,0,0.2)',
            showlegend: false
          }
        ];
        
        layout.title = { text: 'Trend Forecast', font: { color: '#d8f3ff', size: 16 } };
        layout.xaxis.title = 'Period';
        layout.yaxis.title = 'Value';
        
        Plotly.newPlot('plotlyChart', data, layout, config);
        return;
      }
      
      if (type === 'custom') {
        // Custom multi-chart view
        const actual = mainData.actual || mainData.y || mainData.values || [];
        const labels = mainData.x || mainData.labels || actual.map((_, i) => i + 1);
        
        // Calculate statistics
        const sum = actual.reduce((a, b) => a + (parseFloat(b) || 0), 0);
        const mean = sum / actual.length;
        const sorted = [...actual].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        const min = sorted[0];
        const max = sorted[sorted.length - 1];
        
        // Box plot with overlaid scatter
        data = [
          {
            y: actual,
            type: 'box',
            name: 'Distribution',
            marker: { color: '#0088aa' },
            boxpoints: 'all',
            jitter: 0.3,
            pointpos: -1.8
          },
          {
            x: labels.slice(0, 20),
            y: actual.slice(0, 20),
            type: 'scatter',
            mode: 'markers',
            name: 'Sample Points',
            xaxis: 'x2',
            yaxis: 'y2',
            marker: { color: '#ff3333', size: 8 }
          }
        ];
        
        layout.title = { 
          text: `Custom View | Mean: ${mean.toFixed(2)} | Median: ${median} | Range: [${min}, ${max}]`, 
          font: { color: '#d8f3ff', size: 14 } 
        };
        layout.xaxis = { ...layout.xaxis, domain: [0, 0.4] };
        layout.xaxis2 = { ...layout.xaxis, domain: [0.5, 1], anchor: 'y2' };
        layout.yaxis2 = { ...layout.yaxis, anchor: 'x2' };
        
        Plotly.newPlot('plotlyChart', data, layout, config);
        return;
      }
      
      if (type === 'distribution') {
        // Create histogram of actual vs predicted values
        const actual = mainData.actual || mainData.y || [];
        const predicted = mainData.predicted || [];
        
        if (actual.length === 0) {
          throw new Error('No data for distribution');
        }
        
        data = [{
          x: actual,
          type: 'histogram',
          name: 'Actual',
          marker: { color: '#0088aa', line: { color: '#00aacc', width: 1 } },
          opacity: 0.7,
          nbinsx: 20
        }];
        
        if (predicted.length > 0) {
          data.push({
            x: predicted,
            type: 'histogram',
            name: 'Predicted',
            marker: { color: '#00cc66', line: { color: '#00ff88', width: 1 } },
            opacity: 0.7,
            nbinsx: 20
          });
        }
        
        layout.barmode = 'overlay';
        layout.title = { text: 'Value Distribution', font: { color: '#d8f3ff', size: 16 } };
        layout.xaxis.title = 'Value';
        layout.yaxis.title = 'Frequency';
        
        Plotly.newPlot('plotlyChart', data, layout, config);
        return;
      }
      
      if (type === 'correlation') {
        // Create correlation heatmap
        const actual = mainData.actual || mainData.y || [];
        const predicted = mainData.predicted || [];
        
        if (predicted.length === 0 || actual.length === 0 || actual.length !== predicted.length) {
          data = [{
            x: ['Actual', 'Predicted'],
            y: ['Predicted', 'Actual'],
            z: [[1, 0], [0, 1]],
            type: 'heatmap',
            colorscale: [[0, '#000a1a'], [0.5, '#0088aa'], [1, '#00ffcc']],
            showscale: true,
            text: [['1.000', '0.000'], ['0.000', '1.000']],
            texttemplate: '%{text}',
            textfont: { color: '#ffffff', size: 14 }
          }];
          layout.title = { text: 'Correlation Matrix (No prediction data)', font: { color: '#d8f3ff', size: 16 } };
        } else {
          // Calculate correlation coefficient
          const n = actual.length;
          const meanActual = actual.reduce((a, b) => a + b, 0) / n;
          const meanPredicted = predicted.reduce((a, b) => a + b, 0) / n;
          
          let numerator = 0;
          let denomActual = 0;
          let denomPredicted = 0;
          
          for (let i = 0; i < n; i++) {
            const diffActual = actual[i] - meanActual;
            const diffPredicted = predicted[i] - meanPredicted;
            numerator += diffActual * diffPredicted;
            denomActual += diffActual * diffActual;
            denomPredicted += diffPredicted * diffPredicted;
          }
          
          const correlation = denomActual === 0 || denomPredicted === 0 ? 0 : 
            numerator / Math.sqrt(denomActual * denomPredicted);
          
          data = [{
            x: ['Actual', 'Predicted'],
            y: ['Predicted', 'Actual'],
            z: [[1, correlation], [correlation, 1]],
            type: 'heatmap',
            colorscale: [[0, '#000a1a'], [0.5, '#0088aa'], [1, '#00ffcc']],
            showscale: true,
            text: [[1, correlation.toFixed(3)], [correlation.toFixed(3), 1]],
            texttemplate: '%{text}',
            textfont: { color: '#ffffff', size: 14 }
          }];
          
          layout.title = { 
            text: `Correlation Matrix (r=${correlation.toFixed(3)})`, 
            font: { color: '#d8f3ff', size: 16 } 
          };
        }
        
        layout.xaxis.title = '';
        layout.yaxis.title = '';
        
        Plotly.newPlot('plotlyChart', data, layout, config);
        return;
      }
    } catch (err) {
      console.error('Chart error:', err);
      // Fallback to placeholder
      data = [{
        x: ['Error'],
        y: [0],
        type: 'bar',
        marker: { color: 'rgba(255,51,51,0.3)' }
      }];
      layout.annotations = [{
        text: 'Error loading ' + type + ' view. Try Overview tab.',
        showarrow: false,
        x: 0.5,
        y: 0.5,
        xref: 'paper',
        yref: 'paper',
        font: { size: 14, color: 'rgba(255,100,100,0.8)' }
      }];
      Plotly.newPlot('plotlyChart', data, layout, config);
      return;
    }
  }
  
  const chartData = state.chartData[type];
  
  if (chartData) {
    // Use real data from pipeline
    if (chartData.chartType === 'bar' && chartData.x) {
      // Bar chart with actual vs predicted
      const traces = [{
        x: chartData.x,
        y: chartData.actual || chartData.x.map(() => 0),
        name: 'Actual',
        type: 'bar',
        marker: { color: '#0088aa', line: { color: '#00aacc', width: 1 } }
      }];
      if (chartData.predicted?.length) {
        traces.push({
          x: chartData.x,
          y: chartData.predicted,
          name: 'Predicted',
          type: 'bar',
          marker: { color: '#00cc66', line: { color: '#00ff88', width: 1 } }
        });
      }
      data = traces;
    } else if (chartData.chartType === 'line') {
      // Line chart for regression/time-series
      const x = Array.from({ length: chartData.actual?.length || 0 }, (_, i) => i + 1);
      data = [{
        x: x,
        y: chartData.actual,
        name: 'Actual',
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: '#0088aa' },
        line: { color: '#0088aa' }
      }];
      if (chartData.predicted?.length) {
        data.push({
          x: x,
          y: chartData.predicted,
          name: 'Predicted',
          type: 'scatter',
          mode: 'lines+markers',
          marker: { color: '#00cc66' },
          line: { color: '#00cc66', dash: 'dash' }
        });
      }
    } else if (chartData.svg) {
      // SVG from pipeline visualization
      plotlyChart.innerHTML = chartData.svg;
      return;
    } else {
      // Fallback - show whatever data we have
      data = [{
        x: chartData.x || [],
        y: chartData.y || chartData.actual || [],
        type: chartData.chartType || 'bar',
        marker: { color: '#0088aa' }
      }];
    }
  } else {
    // No data yet - show "awaiting hunt" placeholder
    data = [{
      x: ['Start a Hunt'],
      y: [0],
      type: 'bar',
      marker: { color: 'rgba(0,136,170,0.3)' },
      text: ['Awaiting data...'],
      textposition: 'inside',
      hoverinfo: 'none'
    }];
    layout.annotations = [{
      text: 'Ask Leviathan to hunt data for visualization',
      showarrow: false,
      x: 0.5,
      y: 0.5,
      xref: 'paper',
      yref: 'paper',
      font: { size: 14, color: 'rgba(216,243,255,0.5)' }
    }];
  }
  
  Plotly.newPlot('plotlyChart', data, layout, config);
};

const updateChartData = (payload) => {
  if (payload.type && payload.data) {
    state.chartData[payload.type] = payload.data;
  } else if (payload.data) {
    state.chartData['main'] = payload.data;
  } else {
    // Direct payload is the data
    state.chartData['main'] = payload;
  }
  renderChart(state.currentChart);
  logToConsole('viz', 'VIZ', `Chart updated with real data`);
};

// ===== MONSTER EFFECTS =====
const activateTentacles = () => {
  tentacles.classList.add('active');
};

const deactivateTentacles = () => {
  setTimeout(() => tentacles.classList.remove('active'), 2000);
};

const intensifyEye = () => {
  monsterEye.classList.add('intense');
  setTimeout(() => monsterEye.classList.remove('intense'), 3000);
};

const calmEye = () => {
  monsterEye.classList.remove('intense');
};

const triggerRipple = () => {
  brandTitle.classList.remove('ripple');
  void brandTitle.offsetWidth; // Reflow
  brandTitle.classList.add('ripple');
  setTimeout(() => brandTitle.classList.remove('ripple'), 1000);
};

// ===== QUICK ACTIONS =====
const setupQuickActions = () => {
  $$('.action-btn').forEach(btn => {
    btn.onclick = () => handleQuickAction(btn.dataset.action);
  });
};

const handleQuickAction = (action) => {
  const prompts = {
    'auto-eda': 'Run automatic exploratory data analysis on my latest uploaded dataset.',
    'classify-images': 'Classify all images in the uploads folder using computer vision.',
    'spam-detect': 'Analyze the uploaded text data for spam detection using BERT.',
    'time-series': 'Perform time-series forecasting on my temporal data.',
    'anomaly-hunt': 'Hunt for anomalies and outliers in the current dataset.'
  };
  
  const prompt = prompts[action];
  if (prompt) {
    chatInput.value = prompt;
    sendChat();
  }
};

// ===== CHART CONTROLS =====
const setupChartControls = () => {
  $('#zoomIn').onclick = () => {
    Plotly.relayout('plotlyChart', {
      'xaxis.range': [
        (plotlyChart._fullLayout.xaxis.range[0] + plotlyChart._fullLayout.xaxis.range[1]) / 2 - 
        (plotlyChart._fullLayout.xaxis.range[1] - plotlyChart._fullLayout.xaxis.range[0]) / 4,
        (plotlyChart._fullLayout.xaxis.range[0] + plotlyChart._fullLayout.xaxis.range[1]) / 2 + 
        (plotlyChart._fullLayout.xaxis.range[1] - plotlyChart._fullLayout.xaxis.range[0]) / 4
      ]
    });
  };
  
  $('#zoomOut').onclick = () => {
    Plotly.relayout('plotlyChart', { 'xaxis.autorange': true, 'yaxis.autorange': true });
  };
  
  $('#resetView').onclick = () => {
    Plotly.relayout('plotlyChart', { 'xaxis.autorange': true, 'yaxis.autorange': true });
  };
  
  chartTabs.onclick = (e) => {
    if (e.target.classList.contains('chart-tab')) {
      $$('.chart-tab').forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      renderChart(e.target.dataset.chart);
    }
  };
};

// ===== AUTH =====
const setupTabs = () => {
  $$('.tab').forEach(btn => {
    btn.onclick = () => {
      $$('.tab').forEach(t => t.classList.remove('active'));
      $$('.form').forEach(f => f.classList.remove('active'));
      btn.classList.add('active');
      $(`#${btn.dataset.tab}Form`).classList.add('active');
    };
  });
};

const handleAuth = async (endpoint, form) => {
  const formData = new FormData(form);
  const body = {
    email: formData.get('email'),
    password: formData.get('password')
  };
  
  try {
    const res = await fetch(`/api/auth/${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    
    if (!res.ok) {
      authMessage.textContent = data.error || 'Authentication failed';
      showToast(data.error || 'Auth failed', 'error');
      return;
    }
    
    authMessage.textContent = 'Success! Entering the abyss...';
    setTokens(data);
    showDashboard();
    connectWs();
    loadAlerts();
    showToast('Welcome to Leviathan', 'success');
    
  } catch (err) {
    authMessage.textContent = 'Network error';
    showToast('Network error: ' + err.message, 'error');
  }
};

// ===== LOGOUT =====
const logout = async () => {
  try {
    if (state.refreshToken && state.userId) {
      await fetch('/api/auth/logout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: state.userId, refreshToken: state.refreshToken })
      });
    }
  } catch {}
  
  localStorage.clear();
  state.accessToken = '';
  state.refreshToken = '';
  state.userId = '';
  state.ws?.close();
  showAuth();
  showToast('Logged out', 'info');
};

// ===== AUTONOMOUS HUNT =====
const setupHuntButtons = () => {
  const huntSuggestions = $('#huntSuggestions');
  if (!huntSuggestions) return;
  
  huntSuggestions.querySelectorAll('.hunt-btn').forEach(btn => {
    btn.onclick = () => {
      const huntType = btn.dataset.hunt;
      const huntPrompts = {
        'vn-stock': 'Hunt for Vietnam stock market trends. Crawl Yahoo Finance and Alpha Vantage for VNI index data, analyze patterns, and show me price trends with forecasts.',
        'vn-news': 'Hunt for Vietnam news sentiment. Crawl VNExpress and Cafef RSS feeds, analyze sentiment of recent articles, and show me the sentiment distribution.',
        'kaggle-titanic': 'Hunt for the Kaggle Titanic dataset. Download from Kaggle API, analyze survival factors, train a classifier, and visualize the results.',
        'world-bank': 'Hunt for World Bank Vietnam GDP data. Fetch from the World Bank API, analyze economic trends, and show me GDP growth over time with forecasts.'
      };
      
      const prompt = huntPrompts[huntType] || `Hunt for ${huntType} data`;
      chatInput.value = prompt;
      startHunt();
    };
  });
};

const startHunt = () => {
  const text = chatInput.value.trim();
  if (!text) return;
  
  const huntStatus = $('#huntStatus');
  const huntBtn = $('#huntBtn');
  
  // Update UI to show hunting state
  if (huntStatus) {
    huntStatus.textContent = '🔴 Hunting...';
    huntStatus.classList.add('hunting');
  }
  if (huntBtn) {
    huntBtn.classList.add('hunting');
  }
  
  appendChat('user', text);
  state.chatMessages.push({ role: 'user', content: text });
  
  typingIndicator.classList.add('active');
  activateTentacles();
  intensifyEye();
  
  // Send hunt request via WebSocket
  state.ws.send(JSON.stringify({
    type: 'hunt:request',
    requestId: crypto.randomUUID(),
    payload: { 
      prompt: text, 
      messages: state.chatMessages, 
      stream: true,
      hunt: true  // Flag for autonomous hunting
    }
  }));
  
  logToConsole('hunt', 'HUNT', `Autonomous hunt started: "${text.slice(0, 50)}..."`);
  chatInput.value = '';
};

const endHunt = (success = true) => {
  const huntStatus = $('#huntStatus');
  const huntBtn = $('#huntBtn');
  
  if (huntStatus) {
    huntStatus.textContent = success ? '✅ Hunt Complete' : '⚠️ Hunt Partial';
    huntStatus.classList.remove('hunting');
    setTimeout(() => {
      huntStatus.textContent = '🔍 Ready';
    }, 3000);
  }
  if (huntBtn) {
    huntBtn.classList.remove('hunting');
  }
};

// ===== INIT =====
const boot = () => {
  setupTabs();
  setupHuntButtons();
  setupQuickActions();
  setupChartControls();
  
  // Event listeners
  chatSend.onclick = sendChat;
  chatInput.onkeydown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  };
  
  // Hunt button listener
  const huntBtn = $('#huntBtn');
  if (huntBtn) {
    huntBtn.onclick = startHunt;
  }
  
  $('#loginForm').onsubmit = (e) => {
    e.preventDefault();
    handleAuth('login', e.target);
  };
  
  $('#registerForm').onsubmit = (e) => {
    e.preventDefault();
    handleAuth('register', e.target);
  };
  
  logoutBtn.onclick = logout;
  modalClose.onclick = hideAlertModal;
  dismissAlert.onclick = hideAlertModal;
  
  // LLM log panel controls
  if (llmToggle) llmToggle.onclick = toggleLlmLogPanel;
  if (llmLogClose) llmLogClose.onclick = toggleLlmLogPanel;
  
  // Check auth
  if (state.accessToken) {
    showDashboard();
    connectWs();
    loadAlerts();
  } else {
    showAuth();
  }
  
  logToConsole('info', 'SYSTEM', 'Leviathan autonomous hunter initialized');
};

// Start
boot();
