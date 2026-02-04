const authPanel = document.getElementById('authPanel');
const dashboardPanel = document.getElementById('dashboardPanel');
const authMessage = document.getElementById('authMessage');
const statusEl = document.getElementById('status');
const chatLog = document.getElementById('chatLog');
const chatInput = document.getElementById('chatInput');
const chatSend = document.getElementById('chatSend');
const uploadForm = document.getElementById('uploadForm');
const pipelineStatus = document.getElementById('pipelineStatus');
const alertsList = document.getElementById('alertsList');
const chartEl = document.getElementById('chart');
const refreshChart = document.getElementById('refreshChart');
const logoutBtn = document.getElementById('logout');

const state = {
  accessToken: localStorage.getItem('accessToken') || '',
  refreshToken: localStorage.getItem('refreshToken') || '',
  userId: localStorage.getItem('userId') || '',
  ws: null,
  chatMessages: [],
  currentBotEl: null
};

const setStatus = (text) => {
  statusEl.textContent = text;
};

const showDashboard = () => {
  authPanel.style.display = 'none';
  dashboardPanel.classList.add('active');
};

const showAuth = () => {
  authPanel.style.display = 'block';
  dashboardPanel.classList.remove('active');
};

const apiFetch = async (url, options = {}) => {
  const headers = options.headers || {};
  if (state.accessToken) {
    headers.Authorization = `Bearer ${state.accessToken}`;
  }
  const res = await fetch(url, { ...options, headers });
  if (res.status === 401 && state.refreshToken && state.userId) {
    const refreshed = await refreshTokens();
    if (refreshed) {
      headers.Authorization = `Bearer ${state.accessToken}`;
      return fetch(url, { ...options, headers });
    }
  }
  return res;
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
  } catch (err) {
    return false;
  }
};

const connectWs = () => {
  if (state.ws) state.ws.close();
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${protocol}://${window.location.host}`);
  state.ws = ws;

  ws.onopen = () => {
    setStatus('Connected');
    ws.send(JSON.stringify({ type: 'auth', requestId: 'auth', payload: { token: state.accessToken } }));
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'chat:token') {
      appendChat('bot', msg.payload.token, true);
      return;
    }
    if (msg.type === 'chat:end') {
      state.currentBotEl = null;
      return;
    }
    if (msg.type === 'pipeline:status') {
      updatePipeline(msg.payload);
      return;
    }
    if (msg.type === 'alert:new') {
      pushAlert(msg.payload);
      return;
    }
  };

  ws.onclose = () => {
    setStatus('Disconnected');
  };
};

const appendChat = (role, content, stream = false) => {
  if (!content) return;
  let entry = null;
  if (stream && role === 'bot' && state.currentBotEl) {
    entry = state.currentBotEl;
    entry.textContent += content;
  } else {
    entry = document.createElement('div');
    entry.className = `chat-entry ${role}`;
    entry.textContent = content;
    chatLog.appendChild(entry);
    if (stream && role === 'bot') {
      state.currentBotEl = entry;
    }
  }
  chatLog.scrollTop = chatLog.scrollHeight;
};

const updatePipeline = (payload) => {
  const item = document.createElement('div');
  item.className = 'pipeline-item';
  item.textContent = `${payload.stage}: ${payload.status} — ${payload.message}`;
  pipelineStatus.prepend(item);
};

const pushAlert = (alert) => {
  const item = document.createElement('div');
  item.className = 'alert';
  item.textContent = `${alert.level || 'info'}: ${alert.message}`;
  alertsList.prepend(item);
};

const renderChart = () => {
  const values = Array.from({ length: 8 }, () => Math.floor(Math.random() * 100));
  const width = 260;
  const height = 180;
  const max = Math.max(...values, 1);
  const barWidth = width / values.length;
  const bars = values
    .map((v, i) => {
      const barHeight = (v / max) * (height - 20);
      const x = i * barWidth + 6;
      const y = height - barHeight;
      return `<rect x="${x}" y="${y}" width="${barWidth - 12}" height="${barHeight}" rx="6" fill="#0088aa" />`;
    })
    .join('');
  chartEl.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
      <rect width="${width}" height="${height}" fill="rgba(0,0,0,0.2)" rx="12" />
      ${bars}
    </svg>
  `;
};

const loadAlerts = async () => {
  const res = await apiFetch('/api/alerts');
  if (!res.ok) return;
  const data = await res.json();
  alertsList.innerHTML = '';
  (data.alerts || []).forEach(pushAlert);
};

const setupTabs = () => {
  document.querySelectorAll('.tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      document.querySelectorAll('.form').forEach((f) => f.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`${btn.dataset.tab}Form`).classList.add('active');
    });
  });
};

const boot = async () => {
  setupTabs();
  renderChart();
  refreshChart.addEventListener('click', renderChart);

  if (state.accessToken) {
    showDashboard();
    connectWs();
    loadAlerts();
  } else {
    showAuth();
  }
};

const handleAuth = async (endpoint, form) => {
  const formData = new FormData(form);
  const body = {
    email: formData.get('email'),
    password: formData.get('password')
  };
  const res = await fetch(`/api/auth/${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const data = await res.json();
  if (!res.ok) {
    authMessage.textContent = data.error || 'Auth failed';
    return;
  }
  authMessage.textContent = 'Success.';
  setTokens({ ...data, user: data.user });
  showDashboard();
  connectWs();
  loadAlerts();
};

document.getElementById('loginForm').addEventListener('submit', (e) => {
  e.preventDefault();
  handleAuth('login', e.target);
});

document.getElementById('registerForm').addEventListener('submit', (e) => {
  e.preventDefault();
  handleAuth('register', e.target);
});

chatSend.addEventListener('click', () => {
  const text = chatInput.value.trim();
  if (!text) return;
  appendChat('user', text);
  state.chatMessages.push({ role: 'user', content: text });
  state.ws?.send(JSON.stringify({
    type: 'chat:request',
    requestId: crypto.randomUUID(),
    payload: { messages: state.chatMessages, stream: true }
  }));
  chatInput.value = '';
});

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(uploadForm);
  const res = await apiFetch('/api/files/upload', { method: 'POST', body: formData });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || 'Upload failed');
    return;
  }
  updatePipeline({ stage: 'ingest', status: 'queued', message: `Run ${data.runId} started` });
});

logoutBtn.addEventListener('click', async () => {
  if (state.refreshToken && state.userId) {
    await fetch('/api/auth/logout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userId: state.userId, refreshToken: state.refreshToken })
    });
  }
  localStorage.clear();
  state.accessToken = '';
  state.refreshToken = '';
  state.userId = '';
  state.ws?.close();
  showAuth();
});

boot();
