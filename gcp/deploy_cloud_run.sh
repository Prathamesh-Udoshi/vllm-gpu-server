#!/usr/bin/env bash
# ==============================================================================
# One-Click Deployment Script for Google Cloud Run (Serverless GPU)
# ==============================================================================

set -euo pipefail

PROJECT_ID=$(gcloud config get-value project)
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="vllm-cloud-run-service"
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/vllm-repo/${SERVICE_NAME}:latest"
MODEL_NAME="${1:-Qwen/Qwen2.5-0.5B-Instruct}"

echo "======================================================================"
echo " Deploying vLLM to GCP Cloud Run (Serverless GPU)"
echo " GCP Project : ${PROJECT_ID}"
echo " GCP Region  : ${REGION}"
echo " Model Target: ${MODEL_NAME}"
echo "======================================================================"

# 1. Enable Cloud Run & Artifact Registry APIs
echo "[1/4] Enabling required GCP APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# 2. Create Artifact Registry repository if missing
echo "[2/4] Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories create vllm-repo \
    --repository-format=docker \
    --location=${REGION} \
    --description="Docker repository for vLLM Inference Platform" || true

# 3. Build & Push GPU Container Image using Google Cloud Build
echo "[3/4] Building container image using Cloud Build..."
gcloud builds submit --tag ${IMAGE_TAG} -f docker/Dockerfile .

# 4. Deploy to Cloud Run with NVIDIA L4 GPU Allocation
echo "[4/4] Deploying to Cloud Run with NVIDIA L4 GPU..."
gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_TAG} \
    --platform=managed \
    --region=${REGION} \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --no-cpu-throttling \
    --min-instances=1 \
    --max-instances=5 \
    --memory=16Gi \
    --cpu=4 \
    --timeout=600 \
    --set-env-vars="MODEL_NAME=${MODEL_NAME},GPU_MEMORY_UTILIZATION=0.90" \
    --allow-unauthenticated

SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')

echo "======================================================================"
echo " 🎉 Successfully deployed to Google Cloud Run GPU!"
echo " Service Endpoint: ${SERVICE_URL}/v1/chat/completions"
echo " Health Endpoint : ${SERVICE_URL}/health/live"
echo "======================================================================"
