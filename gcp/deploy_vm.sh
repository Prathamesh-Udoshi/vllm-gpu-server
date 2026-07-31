#!/usr/bin/env bash
# ==============================================================================
# Production Deployment Script for GCP VM
# Supports Docker Profiles (API-Only vs Full Monitoring), Model selection & API Key
# ==============================================================================

set -euo pipefail

MODEL_NAME="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
QUANTIZATION="${2:-}"
PROFILES="${3:-}"  # Options: "" (API-Only), "monitoring", or "monitoring,exporters"

echo "======================================================================"
echo " Deploying Enterprise vLLM Stack on GCP VM"
echo " Model Target : ${MODEL_NAME}"
echo " Quantization : ${QUANTIZATION:-None}"
echo " Profiles     : ${PROFILES:-Default (API + Nginx only)}"
echo "======================================================================"

export MODEL_NAME="${MODEL_NAME}"
export QUANTIZATION="${QUANTIZATION}"

# Auto-detect docker command permission
DOCKER_CMD="docker"
if ! docker info &>/dev/null; then
    DOCKER_CMD="sudo docker"
fi

# Determine Docker Compose profile flags
PROFILE_FLAGS=""
if [ -n "${PROFILES}" ]; then
    IFS=',' read -ra ADDR <<< "${PROFILES}"
    for i in "${ADDR[@]}"; do
        PROFILE_FLAGS="${PROFILE_FLAGS} --profile ${i}"
    done
fi

# Stop existing containers if running
${DOCKER_CMD} compose -f docker/docker-compose.yml ${PROFILE_FLAGS} down --remove-orphans || true

# Build & launch stack
${DOCKER_CMD} compose -f docker/docker-compose.yml ${PROFILE_FLAGS} up -d --build

echo "Waiting for vLLM API service healthcheck..."
until [ "$(${DOCKER_CMD} inspect --format='{{.State.Health.Status}}' vllm-api-service 2>/dev/null)" == "healthy" ]; do
    echo -n "."
    sleep 3
done

echo ""
echo "======================================================================"
echo " 🎉 vLLM Platform successfully deployed!"
echo "----------------------------------------------------------------------"
echo " API Endpoint       : http://localhost/v1/chat/completions"
echo " Health Endpoint    : http://localhost/health/ready"
if [[ "${PROFILES}" == *"monitoring"* ]]; then
    echo " Prometheus Metrics : http://localhost:9090"
    echo " Grafana Dashboard  : http://localhost:3000 (User: admin / Pass: admin)"
fi
echo "======================================================================"
