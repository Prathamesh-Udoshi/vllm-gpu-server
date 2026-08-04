# Production CUDA 12.1 GPU Dockerfile for vLLM Platform
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace

# Install vLLM and API server requirements
RUN pip install --no-cache-dir \
    vllm==0.26.0 \
    fastapi \
    uvicorn \
    pydantic \
    pydantic-settings \
    prometheus_client \
    requests \
    aiohttp

# Copy application source code
COPY app/ /workspace/app/

# Expose port
EXPOSE 8000

# Set Hugging Face cache directory
ENV HF_HOME=/workspace/.cache/huggingface

# Healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Launch uvicorn server
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
