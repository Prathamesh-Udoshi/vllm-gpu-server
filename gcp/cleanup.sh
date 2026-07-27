#!/usr/bin/env bash
# ==============================================================================
# Safe Resource Cleanup Script
# Stops containers and frees unused Docker resources WITHOUT deleting downloaded model caches
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo " Cleaning Up vLLM Containers & Docker Cache"
echo " (Hugging Face model weights will NOT be deleted)"
echo "======================================================================"

# Stop and remove containers
docker compose -f docker/docker-compose.yml down --remove-orphans

# Clean up dangling images and stopped containers
docker system prune -f --filter "until=24h"

echo "✓ Cleanup complete! Model weight cache preserved."
