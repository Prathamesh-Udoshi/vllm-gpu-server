#!/usr/bin/env bash
# ==============================================================================
# Automated GCP Compute Engine GPU VM Setup Script
# Target OS: Ubuntu 22.04 LTS (x86_64) with NVIDIA GPU (T4 / L4 / A10G / A100)
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo " Starting GCP GPU VM Environment Setup for vLLM Platform"
echo "======================================================================"

# 1. Update system packages
echo "[1/5] Updating system packages..."
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y build-essential curl wget git jq ca-certificates gnupg lsb-release

# 2. Install NVIDIA CUDA Driver (535 headless driver)
echo "[2/5] Installing NVIDIA GPU CUDA Driver..."
if ! command -v nvidia-smi &> /dev/null; then
    sudo apt-get install -y linux-headers-$(uname -r)
    sudo apt-get install -y nvidia-driver-535-server nvidia-utils-535-server
    echo "✓ NVIDIA CUDA Driver installed successfully."
else
    echo "✓ NVIDIA CUDA Driver already installed."
    nvidia-smi
fi

# 3. Install Docker Engine
echo "[3/5] Installing Docker Engine..."
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
fi

# 4. Install NVIDIA Container Toolkit (GPU passthrough for Docker)
echo "[4/5] Installing NVIDIA Container Toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "✓ NVIDIA Container Toolkit installed & Docker configured."
else
    echo "✓ NVIDIA Container Toolkit already configured."
fi

# 5. Verify Setup
echo "[5/5] Verifying GPU access from inside Docker container..."
sudo docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi || {
    echo "⚠️ Warning: System reboot may be required to activate NVIDIA driver modules."
    echo "Run 'sudo reboot' and then run this script once more."
    exit 0
}

echo "======================================================================"
echo " 🎉 GCP GPU VM Setup Complete! You are ready to deploy vLLM."
echo "======================================================================"
