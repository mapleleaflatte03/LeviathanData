# Leviathan

Leviathan is a production-ready, vanilla full-stack autonomous data intelligence platform. It ships with a Node.js gateway, a Python FastAPI tool service, a modular integration layer for open-source data tools, and a custom Leviathan UI theme.

## Features
- Vanilla frontend: HTML5/CSS3/JS with an oceanic Leviathan theme and subtle tentacle/wave animations.
- Node gateway: JWT auth (access + refresh), file upload, WebSocket streaming, orchestration, metrics.
- Python service: pipeline stages, tool registry, LLM client with fallback.
- Tool integrations: one Python file per major tool with safe optional imports and examples.
- Orchestrator: OpenClaw-like multi-stage swarm (Ingest → Analyze → Viz → Reflect).
- PDF report export (Puppeteer HTML → PDF).
- CI/CD GitHub Actions with SSH deploy.
- Docker Compose with optional profiles and minimal Kubernetes manifests.

## Architecture
```
frontend (vanilla)  <->  Node gateway (REST + WS)  <->  Python tools (FastAPI)
            \_____________________________/
                 orchestrator + LLM
```

## Quick Start (Local)
1. Install Node 20+ and Python 3.11+.
2. Copy `.env.example` to `.env` and update values.
3. Install dependencies:
   - `npm install`
   - `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
4. Start Python service:
   - `python -m uvicorn backend.python.app:app --host 0.0.0.0 --port 8000`
5. Start Node gateway:
   - `node backend/node/server.js`
6. Open `http://localhost:3000`.

## Docker Compose
Default profile runs the core stack only:
- `docker compose -f deploy/docker-compose.yml up`

Optional profiles:
- `vector`: Chroma/Qdrant/Weaviate
- `orchestration`: Airflow/Prefect/Mage/NiFi
- `observability`: Prometheus/Grafana

Example:
- `docker compose -f deploy/docker-compose.yml --profile vector --profile observability up`

## API Reference

### Authentication
All protected endpoints require `Authorization: Bearer <access_token>` header.

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/auth/register` | Register new user | No |
| POST | `/api/auth/login` | Login and get tokens | No |
| POST | `/api/auth/refresh` | Refresh access token | Yes (refresh) |

**Register/Login Request:**
```json
{ "email": "user@example.com", "password": "SecurePass123!" }
```

**Response:**
```json
{
  "accessToken": "eyJ...",
  "refreshToken": "eyJ...",
  "user": { "id": 1, "email": "user@example.com" }
}
```

### Chat (WebSocket)
Connect to `ws://localhost:3000?token=<accessToken>` for real-time chat.

**Send Message:**
```json
{ "type": "chat", "payload": { "message": "Analyze sales data" } }
```

**Receive Response (streamed):**
```json
{ "type": "chat:chunk", "content": "Analyzing..." }
{ "type": "chat:done", "runId": "run_123" }
```

### Runs
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/runs` | List all runs |
| GET | `/api/runs/:id` | Get run details |

### Alerts
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/alerts` | List alerts |
| POST | `/api/alerts` | Create alert |
| DELETE | `/api/alerts/:id` | Delete alert |

### Files
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/files/upload` | Upload file (multipart) |
| GET | `/api/files` | List uploaded files |
| DELETE | `/api/files/:filename` | Delete file |

### Reports
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/reports` | List PDF reports |
| POST | `/api/reports/generate` | Generate new PDF |
| GET | `/api/reports/:filename` | Download PDF |

### Health
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/health` | Health check | No |

**Response:**
```json
{ "status": "ok", "node": true, "python": true }
```

## Tool Integrations (41 Tools)

### Machine Learning (9)
| Tool | File | Description |
|------|------|-------------|
| NumPy | `ml_numpy.py` | Array operations |
| Pandas | `ml_pandas.py` | DataFrames |
| Scikit-learn | `ml_sklearn.py` | Classic ML |
| XGBoost | `ml_xgboost.py` | Gradient boosting |
| LightGBM | `ml_lightgbm.py` | Fast gradient boosting |
| PyTorch | `ml_pytorch.py` | Deep learning |
| TensorFlow | `ml_tensorflow.py` | Deep learning |
| Transformers | `ml_transformers.py` | HuggingFace models |
| FLAML | `ml_flaml.py` | AutoML |

### Orchestration (10)
| Tool | File | Description |
|------|------|-------------|
| Airflow | `orchestration_airflow.py` | DAG workflows |
| Prefect | `orchestration_prefect.py` | Modern workflows |
| Mage | `orchestration_mage.py` | Data pipelines |
| DBT | `orchestration_dbt.py` | SQL transforms |
| LangGraph | `orchestration_langgraph.py` | LLM graphs |
| CrewAI | `orchestration_crewai.py` | Multi-agent |
| AutoGen | `orchestration_autogen.py` | MS agents |
| SemanticKernel | `orchestration_semantickernel.py` | MS AI SDK |
| NiFi | `orchestration_nifi.py` | Data routing |
| OpenClaw | `orchestration_openclaw.py` | Leviathan swarm |

### Databases (6)
| Tool | File | Description |
|------|------|-------------|
| SQLite | `db_sqlite.py` | Local SQL |
| Chroma | `db_chroma.py` | Vector DB |
| Qdrant | `db_qdrant.py` | Vector DB |
| Weaviate | `db_weaviate.py` | Vector DB |
| LanceDB | `db_lancedb.py` | Vector DB |
| Neo4j | `db_neo4j.py` | Graph DB |

### Visualization (5)
| Tool | File | Description |
|------|------|-------------|
| Plotly/Dash | `viz_plotly_dash.py` | Interactive charts |
| Streamlit | `viz_streamlit.py` | Data apps |
| Superset | `viz_superset.py` | BI dashboards |
| Metabase | `viz_metabase.py` | Business analytics |
| Evidence | `viz_evidence.py` | SQL reports |

### Infrastructure (5)
| Tool | File | Description |
|------|------|-------------|
| Docker | `infra_docker.py` | Containers |
| Kubernetes | `infra_kubernetes.py` | Container orchestration |
| Caddy | `infra_caddy.py` | Reverse proxy |
| Traefik | `infra_traefik.py` | Edge router |
| Prometheus/Grafana | `infra_prometheus_grafana.py` | Monitoring |

### Multi-Modal (4)
| Tool | File | Description |
|------|------|-------------|
| FFmpeg | `mm_ffmpeg.py` | Video/audio |
| Whisper | `mm_whisper.py` | Speech-to-text |
| Tesseract | `mm_tesseract.py` | OCR |
| OpenCLIP | `mm_openclip.py` | Image embeddings |

### Browser (2)
| Tool | File | Description |
|------|------|-------------|
| Playwright | `browser_playwright.py` | Browser automation |
| Puppeteer | `browser_puppeteer.py` | Chrome automation |

## Environment Variables
See `.env.example` for full list. Key values:
- `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
- `LLM_FALLBACK_BASE_URL`, `LLM_FALLBACK_API_KEY`, `LLM_FALLBACK_MODEL`
- `JWT_SECRET`, `JWT_ACCESS_TTL`, `JWT_REFRESH_TTL`

## Tests
- Unit tests:
  - `npm run test:unit`
  - `pytest`
- E2E tests:
  - `npx playwright install`
  - `npm run test:e2e`

## Production Deployment (Ubuntu)
1. Install system packages:
   - `sudo apt update`
   - `sudo apt install -y git python3.11 python3.11-venv python3.11-dev build-essential`
2. Install Node.js:
   - `curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -`
   - `sudo apt install -y nodejs`
3. Install Caddy:
   - `sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https`
   - `curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo tee /usr/share/keyrings/caddy-stable-archive-keyring.gpg > /dev/null`
   - `curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list`
   - `sudo apt update && sudo apt install -y caddy`
   - Caddy handles auto-HTTPS (Certbot is not required).
4. Clone and install:
   - `git clone <your-repo> /opt/leviathan`
   - `cd /opt/leviathan`
   - `npm install`
   - `python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
5. Install PM2:
   - `sudo npm install -g pm2`
6. Configure env:
   - `cp .env.example .env` and edit values
7. Start services:
   - `pm2 start backend/node/server.js --name leviathan-node`
   - `pm2 startup && pm2 save`
   - `sudo cp deploy/caddyfile /etc/caddy/Caddyfile`
   - `sudo systemctl reload caddy`

## Server Setup (Exact Commands)
```bash
sudo apt update
sudo apt install -y git python3.11 python3.11-venv python3.11-dev build-essential
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo tee /usr/share/keyrings/caddy-stable-archive-keyring.gpg > /dev/null
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
git clone <your-repo> /opt/leviathan
cd /opt/leviathan
cp .env.example .env
npm install
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
sudo npm install -g pm2
pm2 start backend/node/server.js --name leviathan-node
pm2 startup && pm2 save
sudo cp deploy/caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

## CI/CD
The GitHub Actions workflow builds, tests, and deploys via SSH.
Set these repository secrets:
- `SSH_HOST`, `SSH_USER`, `SSH_KEY`, `SSH_PORT`

## Rust Acceleration (Optional)
A minimal Rust crate lives in `rust/leviathan_accel` with a tiny CLI and a Node/Python adapter example in the docs.

## License
MIT
