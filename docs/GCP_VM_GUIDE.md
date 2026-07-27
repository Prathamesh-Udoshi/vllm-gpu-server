# Step-by-Step Deployment Guide: GCP Compute Engine (GPU VM)

This handbook provides exact, step-by-step instructions to create, configure, deploy, and operate your enterprise vLLM inference platform on a Google Cloud Platform (GCP) Compute Engine GPU VM.

---

## Step 1: Request GPU Quota on GCP

1. Go to the **Google Cloud Console** -> **IAM & Admin** -> **Quotas**.
2. Filter by `NVIDIA T4 GPUs` or `NVIDIA L4 GPUs`.
3. Select your desired region (e.g. `us-central1`).
4. Click **Edit Quotas**, request a limit of at least **1 GPU**, and submit justification ("vLLM Inference Testing"). Approval usually takes 5 to 30 minutes.

---

## Step 2: Create a GPU Virtual Machine Instance

### Option A: Using GCP Console UI
1. Navigate to **Compute Engine** -> **VM Instances** -> **Create Instance**.
2. **Name**: `vllm-gpu-server`
3. **Region**: `us-central1` (or your quota region)
4. **Machine Family**:
   * For **NVIDIA T4 GPU**: Select **N1 standard** (`n1-standard-4` - 4 vCPUs, 15GB RAM). Click **CPU Platform and GPU**, add **1 NVIDIA T4 GPU**.
   * For **NVIDIA L4 GPU**: Select **G2 standard** (`g2-standard-4` - 4 vCPUs, 16GB RAM, 1 NVIDIA L4 GPU).
5. **Boot Disk**:
   * OS: **Ubuntu**
   * Version: **Ubuntu 22.04 LTS**
   * Disk Size: **100 GB** (Standard Persistent Disk or Balanced Persistent Disk to store HuggingFace models).
6. **Firewall**: Check **Allow HTTP traffic** and **Allow HTTPS traffic**.
7. Click **Create**.

### Option B: Using `gcloud` CLI (Command Line)
```bash
gcloud compute instances create vllm-gpu-server \
    --project=$(gcloud config get-value project) \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=count=1,type=nvidia-l4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE \
    --tags=http-server,https-server
```

---

## Step 3: Connect and Configure GPU Environment

1. Connect to your VM via SSH:
   ```bash
   gcloud compute ssh vllm-gpu-server --zone=us-central1-a
   ```
2. Clone your codebase repository onto the VM:
   ```bash
   git clone https://github.com/your-repo/vllm-platform.git
   cd vllm-platform
   ```
3. Run the automated GCP GPU setup script:
   ```bash
   bash gcp/setup_vm.sh
   ```
   *This script automatically installs NVIDIA CUDA Driver 535, Docker Engine, and NVIDIA Container Toolkit.*

4. Verify GPU status:
   ```bash
   nvidia-smi
   ```

---

## Step 4: Launch the Production vLLM Stack

Run the one-click deployment script:
```bash
bash gcp/deploy_vm.sh "Qwen/Qwen2.5-0.5B-Instruct" ""
```

To run a 4-bit AWQ quantized model (e.g. Qwen2.5 7B AWQ):
```bash
bash gcp/deploy_vm.sh "Qwen/Qwen2.5-7B-Instruct-AWQ" "awq"
```

---

## Step 5: Test and Benchmark Your Infrastructure

1. Test Liveness & Readiness:
   ```bash
   curl http://localhost/health/ready
   ```

2. Test Chat Completion Token Stream:
   ```bash
   curl -X POST http://localhost/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen2.5-0.5B-Instruct",
       "messages": [{"role": "user", "content": "Explain how PagedAttention works in vLLM."}],
       "stream": true
     }'
   ```

3. Run Concurrency Benchmark:
   ```bash
   python3 benchmarks/benchmark_load.py --url http://localhost/v1/chat/completions --concurrency 10 --requests 50
   ```

4. View Live Performance Metrics in Grafana:
   * Open SSH tunnel for port 3000:
     ```bash
     gcloud compute ssh vllm-gpu-server --zone=us-central1-a -- -L 3000:localhost:3000 -L 9090:localhost:9090
     ```
   * Open your browser to `http://localhost:3000` (Login: `admin` / `admin`).
   * Import dashboard from `monitoring/grafana/dashboards/llm_performance.json`.

---

## Step 6: Stop VM when Done (Cost Management)

To avoid incurring hourly GPU charges when not testing:
```bash
gcloud compute instances stop vllm-gpu-server --zone=us-central1-a
```
When ready to resume:
```bash
gcloud compute instances start vllm-gpu-server --zone=us-central1-a
```
