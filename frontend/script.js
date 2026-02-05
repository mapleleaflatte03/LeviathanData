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
  autoDismissTimer: null
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
    case 'pipeline:status':
      updatePipeline(msg.payload);
      break;
    case 'pipeline:chart':
      updateChartData(msg.payload);
      break;
    case 'alert:new':
      pushAlert(msg.payload);
      break;
    case 'swarm:log':
      logToConsole(msg.payload.level, msg.payload.stage, msg.payload.message);
      break;
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
  
  if (type === 'main' || !state.chartData[type]) {
    // Demo data
    const x = Array.from({ length: 20 }, (_, i) => `Day ${i + 1}`);
    const y = Array.from({ length: 20 }, () => Math.floor(Math.random() * 100) + 20);
    data = [{
      x, y,
      type: 'bar',
      marker: {
        color: y.map(v => v > 70 ? '#00cc66' : v > 40 ? '#0088aa' : '#ff4444'),
        line: { color: '#00aacc', width: 1 }
      }
    }];
  } else if (type === 'distribution') {
    const values = Array.from({ length: 100 }, () => Math.random() * 100);
    data = [{
      x: values,
      type: 'histogram',
      marker: { color: '#0088aa' },
      nbinsx: 20
    }];
  } else if (type === 'correlation') {
    const x = Array.from({ length: 50 }, () => Math.random() * 100);
    const y = x.map(v => v * 0.8 + Math.random() * 30);
    data = [{
      x, y,
      mode: 'markers',
      type: 'scatter',
      marker: {
        color: '#00aacc',
        size: 8,
        opacity: 0.7
      }
    }];
  }
  
  Plotly.newPlot('plotlyChart', data, layout, config);
};

const updateChartData = (payload) => {
  state.chartData[payload.type] = payload.data;
  if (state.currentChart === payload.type) {
    renderChart(payload.type);
  }
  logToConsole('viz', 'VIZ', `Chart "${payload.type}" updated`);
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

// ===== INIT =====
const boot = () => {
  setupTabs();
  setupUpload();
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
  
  // Check auth
  if (state.accessToken) {
    showDashboard();
    connectWs();
    loadAlerts();
  } else {
    showAuth();
  }
  
  logToConsole('info', 'SYSTEM', 'Leviathan frontend initialized');
};

// Start
boot();
