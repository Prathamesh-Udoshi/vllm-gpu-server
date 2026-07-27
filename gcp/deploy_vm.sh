#!/usr/bin/env bash
# ==============================================================================
# One-Click Deployment Script for GCP Compute Engine GPU VM
# ==============================================================================

set -euo pipefail

MODEL_NAME="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
QUANTIZATION="${2:-}"

echo "======================================================================"
echo " Deploying Enterprise vLLM Stack on GCP VM"
echo " Model: ${MODEL_NAME}"
echo " Quantization: ${QUANTIZATION:-None}"
echo "======================================================================"

export MODEL_NAME="${MODEL_NAME}"
export QUANTIZATION="${QUANTIZATION}"

# Bring down existing containers if any
docker compose -f docker/docker-compose.yml down --remove-orphans || true

# Build and launch multi-service stack in detached mode
docker compose -f docker/docker-compose.yml up -d --build

echo "Waiting for vLLM API service to become healthy..."
until [ "$(docker inspect --format='{{.State.Health.Status}}' vllm-api-service 2>/dev/null)" == "healthy" ]; do
    echo -n "."
    sleep 3
done

echo ""
echo "======================================================================"
echo " 🎉 vLLM Platform successfully deployed!"
echo "----------------------------------------------------------------------"
echo " API Endpoint       : http://localhost/v1/chat/completions"
echo " Prometheus Metrics : http://localhost/metrics (or via Prometheus :9090)"
echo " Grafana Dashboard  : http://localhost:3000 (User: admin / Pass: admin)"
echo " Health Endpoint    : http://localhost/health/ready"
echo "======================================================================"
