#!/usr/bin/env bash
# ==============================================================================
# Idempotent GCP Compute Engine GPU VM Environment Setup Script
# Configures CUDA drivers, Docker, NVIDIA Container Toolkit, and Systemd Service
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo " Starting GCP GPU VM Production Environment Setup"
echo "======================================================================"

# 1. Update system packages
echo "[1/6] Updating system package index..."
sudo apt-get update -y
sudo apt-get install -y build-essential curl wget git jq ca-certificates gnupg lsb-release

# 2. Install NVIDIA CUDA Driver (535 headless driver)
echo "[2/6] Verifying NVIDIA GPU Driver..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA CUDA Driver 535..."
    sudo apt-get install -y linux-headers-$(uname -r)
    sudo apt-get install -y nvidia-driver-535-server nvidia-utils-535-server
    echo "✓ NVIDIA CUDA Driver installed successfully."
else
    echo "✓ NVIDIA CUDA Driver already installed."
    nvidia-smi
fi

# 3. Install Docker Engine
echo "[3/6] Verifying Docker Engine..."
if ! command -v docker &> /dev/null; then
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker $USER
    echo "✓ Docker installed successfully."
else
    echo "✓ Docker already installed."
    sudo usermod -aG docker $USER || true
fi

# 4. Install NVIDIA Container Toolkit
echo "[4/6] Verifying NVIDIA Container Toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "✓ NVIDIA Container Toolkit configured."
else
    echo "✓ NVIDIA Container Toolkit already configured."
fi

# 5. Configure Systemd Auto-Recovery Service for VM Reboots
echo "[5/6] Registering Systemd auto-recovery service..."
SERVICE_PATH="/etc/systemd/system/vllm-platform.service"
CURRENT_DIR=$(pwd)

sudo bash -c "cat <<EOF > ${SERVICE_PATH}
[Unit]
Description=vLLM Production Inference Platform Service
After=docker.service nvidia-persistenced.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${CURRENT_DIR}
ExecStart=/usr/bin/docker compose -f docker/docker-compose.yml up -d
ExecStop=/usr/bin/docker compose -f docker/docker-compose.yml stop
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF"

sudo systemctl daemon-reload
sudo systemctl enable vllm-platform.service
echo "✓ Systemd service 'vllm-platform.service' registered and enabled for auto-boot recovery."

# 6. Verify GPU access from Docker
echo "[6/6] Testing GPU passthrough inside Docker..."
sudo docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi || {
    echo "⚠️ Warning: System reboot recommended to initialize CUDA driver modules."
    echo "Run 'sudo reboot' then run deploy_vm.sh."
    exit 0
}

echo "======================================================================"
echo " 🎉 Environment Setup Complete! System is ready for deployment."
echo "======================================================================"
