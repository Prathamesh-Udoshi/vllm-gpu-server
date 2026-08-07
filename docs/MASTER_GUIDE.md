# Enterprise MLOps Master Handbook: Self-Hosted vLLM Platform on GCP

This handbook provides an architectural and operational blueprint for serving quantized open-source language models using **vLLM**, **FastAPI**, **Docker**, **Nginx**, **Prometheus**, and **Grafana** on Google Cloud Platform (GCP).

---

## 1. Executive Summary & Production Philosophy

Building a self-hosted LLM inference platform requires balancing **throughput, latency, memory utilization, and cloud infrastructure cost**.

### High-Level Architecture
```
                                +-----------------------------------+
                                | Client SDK / Web App / Load Test  |
                                +-----------------------------------+
                                                  |
                                       (HTTPS / SSE Streaming)
                                                  v
                                +-----------------------------------+
                                |      Nginx Reverse Proxy          |
                                |  (Rate Limit, SSE Buffering OFF)  |
                                +-----------------------------------+
                                  |               |               |
                    +-------------+               |               +-------------+
                    | /v1/ & /health/             | /prometheus/                | /grafana/
                    v                             v                             v
  +-----------------------------------+ +-------------------+         +-------------------+
  |        FastAPI Gateway            | | Prometheus Server |         | Grafana Dashboard |
  | (API Key, Request ID, Tracing)    | | (Metrics Scraper)|         | (Live Analytics)  |
  +-----------------------------------+ +-------------------+         +-------------------+
                    |                             ^                             ^
                    v                             |                             |
  +-----------------------------------+           |                             |
  |       vLLM AsyncLLMEngine         |-----------+-----------------------------+
  | (PagedAttention, Prefix Cache)    |
  +-----------------------------------+
                    |
         +----------+----------+
         |                     |
         v                     v
+------------------+  +-----------------------------------+
| Persistent Disk  |  | NVIDIA GPU (T4 / L4 / A10G)       |
| (HF Model Cache) |  | (Paged VRAM Blocks & Tensor Cores)|
+------------------+  +-----------------------------------+
```

---

## 2. Hardware Profile Matrix: T4 vs. L4 vs. A10G

| Hardware Metric | NVIDIA T4 GPU | NVIDIA L4 GPU | NVIDIA A10G GPU |
| :--- | :--- | :--- | :--- |
| **VRAM Capacity** | 16 GB GDDR6 | 24 GB GDDR6 | 24 GB GDDR6 |
| **Memory Bandwidth** | 320 GB/s | **300 GB/s (Ada Lovelace)** | **600 GB/s (Ampere)** |
| **GCP Machine Family** | `n1-standard-4` | `g2-standard-4` | `g2-standard-8` |
| **On-Demand Hourly Cost**| ~$0.50 / hr | ~$0.70 / hr | ~$1.20 / hr |
| **Spot VM Hourly Cost**  | **~$0.18 / hr** | **~$0.24 / hr** | **~$0.40 / hr** |
| **Recommended Model Size**| 0.5B to 7B AWQ (4-bit) | 7B AWQ / 8B FP8 | 7B / 13B AWQ / 8B FP16 |
| **vLLM `gpu_memory_utilization`**| `0.90` | `0.90` | `0.92` |

---

## 3. Cost Optimization Strategies on Google Cloud

1. **Spot VMs (`--provisioning-model=SPOT`)**:
   - Reduces hourly GPU VM billing by **60% to 70%**.
   - Model weights are cached on persistent disk, so if a Spot VM is preempted, a new instance attaches the same disk and boots up in seconds without re-downloading.

2. **Persistent Model Weight Cache**:
   - Hugging Face weights are mapped to host directory `~/.cache/huggingface` via Docker volume `hf-cache`.
   - Deleting or rebuilding Docker containers does **NOT** re-download model weights.

3. **Docker Profiles (`monitoring` & `exporters`)**:
   - Default deployment runs **API + Nginx only**, consuming minimal CPU/RAM.
   - Prometheus and Grafana are enabled when `PROFILES=monitoring` is specified.
   - Node Exporter and DCGM Exporter are enabled when `PROFILES=monitoring,exporters` is specified.

---

## 4. Security & Authentication Architecture

* **API Key Protection**: Configured via `API_KEY` in `.env`.
* **Header Support**: Accepts authentication through either:
  * `X-API-Key: <key>`
  * `Authorization: Bearer <key>`
* **CORS Policy**: Configured via `CORS_ORIGINS` setting (`*` or comma-separated list of domains).

---

## 5. Observability & Telemetry Metrics

Custom Prometheus metrics exported from [app/metrics.py](file:///d:/Edutainer/vLLM/app/metrics.py):

| Metric Name | Type | Description |
| :--- | :--- | :--- |
| `llm_time_to_first_token_seconds` | Histogram | Time To First Token (TTFT - prefill latency) |
| `llm_time_per_output_token_seconds` | Histogram | Time Per Output Token (TPOT - decode latency) |
| `llm_request_duration_seconds` | Histogram | Total end-to-end API request duration |
| `llm_prompt_tokens_total` | Counter | Total input (prompt) tokens processed |
| `llm_completion_tokens_total` | Counter | Total output (completion) tokens generated |
| `llm_requests_total` | Counter | Total API request counter (labels: `status="success"`, `status="error"`) |
| `llm_num_requests_running` | Gauge | Active requests in execution batch |

Nginx exposes Prometheus at `http://<HOST>/prometheus` and Grafana at `http://<HOST>/grafana`.

---

## 6. Operational Best Practices & Recovery

### Automated Recovery After VM Reboot
The VM setup script registers a systemd unit (`vllm-platform.service`). When a VM starts or reboots after being stopped, systemd automatically executes `docker compose up -d`, bringing the API server back online.

### Operational CLI Shortcuts (Makefile)
* **Dev Server**: `make run-dev`
* **Default Stack**: `make docker-up`
* **Monitoring Stack**: `make docker-up-monitoring`
* **Load Test**: `make benchmark`

### Clean Operations
* **Safe Update**: `bash gcp/update.sh` (Pulls code updates, rebuilds containers, preserves model cache).
* **Backup**: `bash gcp/backup.sh` (Backs up `.env` and configuration files).
* **Cleanup**: `bash gcp/cleanup.sh` (Prunes unused Docker resources without wiping models).

