# Enterprise vLLM Production Inference Platform (GCP MLOps Blueprint)

A production-grade, highly reliable, cost-optimized LLM inference platform built to serve quantized open-source language models using **vLLM (`AsyncLLMEngine`)** on Google Cloud Platform (GCP) GPU VMs or local CPU/GPU environments.

Features an **OpenAI-compatible REST/SSE streaming API (FastAPI)** with API Key authentication, **Docker Profiles**, **Nginx reverse proxy**, **Prometheus & Grafana observability**, and **asynchronous load benchmarking**.

---

## 🏛️ System Architecture

```mermaid
graph TD
    Client[Client / SDK / Benchmark Suite] -->|HTTPS / SSE| Nginx[Nginx Reverse Proxy - SSE Enabled]
    Nginx -->|HTTP / Request Tracing| FastAPI[FastAPI Gateway Server - API Key Auth]
    Nginx -->|/prometheus| Prometheus[Prometheus Server - Monitoring Profile]
    Nginx -->|/grafana| Grafana[Grafana Dashboard - Monitoring Profile]
    FastAPI -->|Async Engine Call| vLLMEngine[vLLM AsyncLLMEngine Core]
    vLLMEngine -->|PagedAttention & CUDA| Hardware[NVIDIA GPU VRAM / Host RAM]
    Hardware -->|Persistent Weight Storage| ModelCache[Host HF Disk Cache - Survives Restarts]
    vLLMEngine -->|Metrics Collector| PrometheusExporter[Prometheus Metrics Exporter]
    PrometheusExporter -->|Scrape /metrics| Prometheus
    Prometheus -->|Live Visuals| Grafana
```

---

## 📁 Repository Structure

```
d:/Edutainer/vLLM/
├── .env.example             # Template for runtime environment variables & API keys
├── Makefile                 # Developer & DevOps CLI automation commands
├── README.md                # Primary project overview & operational documentation
├── requirements.txt         # Python runtime dependencies
├── app/
│   ├── main.py              # FastAPI entrypoint, lifespan manager, CORS, request tracing, health probes
│   ├── config.py            # Comprehensive runtime parameters & environment variables
│   ├── engine.py            # AsyncLLMEngine core wrapper with SSE streaming & TTFT/TPOT metrics
│   ├── router.py            # OpenAI API routes (/v1/chat/completions, /v1/models) with API key auth
│   ├── metrics.py           # Custom Prometheus metrics definitions (TTFT, TPOT, KV Cache %)
│   └── schemas.py           # Pydantic schemas matching OpenAI ChatCompletion format
├── docker/
│   ├── Dockerfile           # Multi-stage CUDA 12.1 GPU Dockerfile (vLLM 0.26.0)
│   ├── Dockerfile.cpu       # Multi-stage CPU Dockerfile for local development
│   └── docker-compose.yml   # Orchestrator with Docker Profiles (default, monitoring, exporters)
├── nginx/
│   └── nginx.conf           # Enterprise Nginx proxy (SSE proxy_buffering off, rate limiting, /grafana & /prometheus subpaths)
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml   # Prometheus scraper configuration (vLLM API, Node Exporter, DCGM Exporter)
│   └── grafana/
│       └── dashboards/
│           └── llm_performance.json # Production Grafana dashboard layout
├── benchmarks/
│   ├── benchmark_load.py    # Async multi-concurrency load generator measuring TTFT (P50/P90/P99) & TPOT
│   └── payload.json         # Benchmark prompt payload
├── gcp/
│   ├── setup_vm.sh          # Idempotent GCP VM setup (CUDA 535, Docker, Toolkit, Systemd Auto-Recovery)
│   ├── deploy_vm.sh         # Deployment script with Docker profiles & model flags
│   ├── deploy_cloud_run.sh  # Deployment script for GCP Cloud Run serverless GPU (L4)
│   ├── update.sh            # Safe update script preserving HF model cache
│   ├── backup.sh            # Backup configuration script
│   └── cleanup.sh           # Resource cleanup script preserving model weights
└── docs/
    ├── MASTER_GUIDE.md      # Master MLOps Infrastructure Handbook
    ├── GCP_VM_GUIDE.md      # Step-by-step GCP Compute Engine GPU VM handbook
    └── CLOUD_RUN_GUIDE.md   # Step-by-step GCP Cloud Run GPU handbook
```

---

## 💰 GCP Cost Matrix & Hardware Selection

| Deployment Option | GPU & Machine Specs | On-Demand Cost | Spot VM Cost | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **GCP T4 GPU VM** | 1x NVIDIA T4 (16GB) + `n1-standard-4` | ~$0.50 / hr | **~$0.18 / hr** | Budget development & 7B AWQ model testing |
| **GCP L4 GPU VM** | 1x NVIDIA L4 (24GB) + `g2-standard-4` | ~$0.70 / hr | **~$0.24 / hr** | High throughput production & 8B FP8 serving |
| **GCP Cloud Run GPU**| 1x NVIDIA L4 (Serverless) | ~$0.65 / hr | Scale to 0 ($0) | Low-concurrency APIs with scale-to-zero |

---

## 🛠️ Quickstart with Makefile

Use `make` commands for easy local execution, container lifecycle management, and benchmarking:

```bash
make help                 # Display all available operational targets
make env                  # Initialize .env from .env.example template
make run-dev              # Launch local FastAPI dev server with auto-reload
make docker-up            # Start API + Nginx containers (Default profile)
make docker-up-monitoring # Start API + Nginx + Prometheus + Grafana (Monitoring profile)
make benchmark            # Execute load test (5 concurrency, 20 requests)
make docker-down          # Stop all container profiles & networks
```

---

## 🚀 Deployment & Operations

### 1. Provision GPU VM on GCP
```bash
gcloud compute instances create vllm-gpu-server \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=count=1,type=nvidia-l4 \
    --maintenance-policy=TERMINATE \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-balanced \
    --tags=http-server,https-server
```

### 2. Environment Setup & Auto-Recovery Service
```bash
gcloud compute ssh vllm-gpu-server --zone=us-central1-a
cd vLLM
bash gcp/setup_vm.sh
```

### 3. Launch Platform Options

#### Option A: Deploy on Compute Engine GPU VM
```bash
# Minimal RAM/CPU footprint (API + Nginx only)
bash gcp/deploy_vm.sh "Qwen/Qwen2.5-0.5B-Instruct" "" ""

# Full Observability Stack (API + Nginx + Prometheus + Grafana)
bash gcp/deploy_vm.sh "Qwen/Qwen2.5-0.5B-Instruct" "" "monitoring"

# Full Stack + Hardware Exporters (API + Nginx + Prometheus + Grafana + Node + DCGM GPU Exporters)
bash gcp/deploy_vm.sh "Qwen/Qwen2.5-0.5B-Instruct" "" "monitoring,exporters"
```

#### Option B: Deploy on GCP Cloud Run (Serverless GPU)
```bash
bash gcp/deploy_cloud_run.sh "Qwen/Qwen2.5-0.5B-Instruct"
```

---

## 🔍 Verification & Health Endpoints

Once deployed, verify services through Nginx reverse proxy:

* **Readiness Check**:
  ```bash
  curl http://localhost/health/ready
  ```
* **OpenAI Chat Completion Stream**:
  ```bash
  curl -X POST http://localhost/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen2.5-0.5B-Instruct",
      "messages": [{"role": "user", "content": "Explain PagedAttention"}],
      "stream": true
    }'
  ```
* **Observability Dashboards**:
  * Prometheus: `http://localhost/prometheus`
  * Grafana: `http://localhost/grafana` (Login: `admin` / `admin`)

---

## 🧪 Benchmarking under Concurrency

Run the async load test generator to measure empirical **TTFT (P50/P90/P99)**, **TPOT**, and **Tokens/Second**:
```bash
python benchmarks/benchmark_load.py --url http://localhost/v1/chat/completions --concurrency 10 --requests 50
```

---

## 📚 Technical Handbooks

* 📖 [Master MLOps Infrastructure Handbook](file:///d:/Edutainer/vLLM/docs/MASTER_GUIDE.md)
* 🖥️ [GCP Compute Engine GPU VM Setup Handbook](file:///d:/Edutainer/vLLM/docs/GCP_VM_GUIDE.md)
* ☁️ [GCP Cloud Run GPU Setup Handbook](file:///d:/Edutainer/vLLM/docs/CLOUD_RUN_GUIDE.md)

