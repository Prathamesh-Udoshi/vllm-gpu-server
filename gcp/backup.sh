#!/usr/bin/env bash
# ==============================================================================
# Configuration & Metrics Backup Script for vLLM Platform
# ==============================================================================

set -euo pipefail

BACKUP_DIR="backups/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BACKUP_DIR}"

echo "======================================================================"
echo " Backing up vLLM Platform Configuration to ${BACKUP_DIR}"
echo "======================================================================"

# Backup environment variables if present
if [ -f ".env" ]; then
    cp .env "${BACKUP_DIR}/.env"
fi

# Backup custom configs
cp app/config.py "${BACKUP_DIR}/config.py"
cp nginx/nginx.conf "${BACKUP_DIR}/nginx.conf"
cp monitoring/prometheus/prometheus.yml "${BACKUP_DIR}/prometheus.yml"

echo "✓ Configuration files backed up successfully in ${BACKUP_DIR}."
