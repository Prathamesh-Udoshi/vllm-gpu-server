# Step-by-Step Deployment Guide: GCP Cloud Run (Serverless GPU)

Google Cloud Run allows deploying containerized workloads to a managed serverless platform with NVIDIA L4 GPU acceleration (`--gpu=1`, `--gpu-type=nvidia-l4`).

---

## Step 1: Prerequisites & GCP Setup

1. Ensure `gcloud` CLI is installed and authenticated:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```
2. Enable required Google Cloud APIs:
   ```bash
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
   ```

---

## Step 2: One-Click Deploy to Cloud Run GPU

Run the deployment script from the root directory:
```bash
bash gcp/deploy_cloud_run.sh "Qwen/Qwen2.5-0.5B-Instruct"
```

What this script executes under the hood:
1. Creates an Artifact Registry Docker repository (`vllm-repo`).
2. Submits the code to Google Cloud Build using `docker/Dockerfile`.
3. Deploys the built image to Cloud Run with 1x NVIDIA L4 GPU, `--min-instances=1`, and CPU unthrottled.

---

## Step 3: Test Cloud Run API Endpoint

Get your deployed service URL:
```bash
SERVICE_URL=$(gcloud run services describe vllm-cloud-run-service --region=us-central1 --format='value(status.url)')
echo "Service Endpoint: ${SERVICE_URL}"
```

Test streaming endpoint:
```bash
# Standard request without API key
curl -X POST "${SERVICE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What are the benefits of serverless GPUs?"}],
    "stream": true
  }'

# Request with API Key (if configured via env var API_KEY)
curl -X POST "${SERVICE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_SECRET_API_KEY" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What are the benefits of serverless GPUs?"}],
    "stream": true
  }'
```

---

## Step 4: Key Cloud Run Production Settings

* **`--min-instances=1`**: CRITICAL to prevent cold starts where model weight downloading causes 60-second initial request delays.
* **`--no-cpu-throttling`**: Ensures continuous CPU background processing for vLLM event loops.
* **`--memory=16Gi`**: Provides sufficient RAM for PyTorch context and model initialization.
* **`--gpu=1 --gpu-type=nvidia-l4`**: Allocates 1x NVIDIA L4 (24GB VRAM) for accelerated inference.

