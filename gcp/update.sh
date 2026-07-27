#!/usr/bin/env bash
# ==============================================================================
# Safe Update Script for Production vLLM Platform
# Pulls latest git code and restarts containers while preserving Hugging Face model cache
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo " Updating vLLM Platform Stack"
echo "======================================================================"

# 1. Pull latest code from git repository
if git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "[1/3] Pulling latest git repository updates..."
    git pull origin main || git pull
else
    echo "[1/3] Not a git repository. Skipping git pull."
fi

# 2. Rebuild and restart containers
echo "[2/3] Rebuilding updated Docker containers..."
docker compose -f docker/docker-compose.yml build

echo "[3/3] Restarting services..."
docker compose -f docker/docker-compose.yml up -d

echo "======================================================================"
echo " 🎉 Update complete! Hugging Face model weight cache was preserved."
echo "======================================================================"
