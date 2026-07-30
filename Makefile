# ==============================================================================
# Enterprise vLLM Production Inference Platform - Makefile
# ==============================================================================

.PHONY: help env venv install run-dev docker-build docker-build-cpu docker-up docker-up-monitoring docker-down docker-logs benchmark lint clean gcp-setup gcp-deploy gcp-update gcp-cleanup

.DEFAULT_GOAL := help

# Colors for terminal output
CYAN  := \033[36m
GREEN := \033[32m
RESET := \033[0m

help: ## Show available Makefile targets with descriptions
	@echo ""
	@echo "Enterprise vLLM Platform - Operational Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ------------------------------------------------------------------------------
# 1. Environment & Setup
# ------------------------------------------------------------------------------
env: ## Copy .env.example to .env if .env does not exist
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN).env created successfully from .env.example$(RESET)"; \
	else \
		echo ".env already exists"; \
	fi

venv: ## Create virtual environment (.venv)
	python -m venv .venv
	@echo "$(GREEN)Virtual environment created in .venv$(RESET)"

install: ## Install Python dependencies from requirements.txt
	pip install -r requirements.txt

# ------------------------------------------------------------------------------
# 2. Local Execution
# ------------------------------------------------------------------------------
run-dev: ## Run local FastAPI development server with uvicorn reloader
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# ------------------------------------------------------------------------------
# 3. Docker Management
# ------------------------------------------------------------------------------
docker-build: ## Build CUDA 12.1 GPU production Docker image
	docker build -f docker/Dockerfile -t vllm-platform:latest .

docker-build-cpu: ## Build CPU multi-stage Docker image for local testing
	docker build -f docker/Dockerfile.cpu -t vllm-platform:cpu .

docker-up: ## Run API & Nginx containers via Docker Compose (Default profile)
	docker compose -f docker/docker-compose.yml up -d --build

docker-up-monitoring: ## Run API, Nginx, Prometheus & Grafana via Docker Compose
	docker compose -f docker/docker-compose.yml --profile monitoring up -d --build

docker-down: ## Stop and remove all running Docker containers & networks
	docker compose -f docker/docker-compose.yml --profile monitoring down

docker-logs: ## Tail live logs from Docker Compose services
	docker compose -f docker/docker-compose.yml logs -f

# ------------------------------------------------------------------------------
# 4. Benchmarking & Quality
# ------------------------------------------------------------------------------
benchmark: ## Run asynchronous load benchmark (5 concurrency, 20 requests)
	python benchmarks/benchmark_load.py --url http://localhost:8000/v1/chat/completions --concurrency 5 --requests 20

lint: ## Run syntax and bytecode check across Python codebase
	python -m compileall app benchmarks

clean: ## Clean Python cache files, pyc files, and temporary artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Cleanup completed.$(RESET)"

# ------------------------------------------------------------------------------
# 5. GCP Operations
# ------------------------------------------------------------------------------
gcp-setup: ## Run GCP GPU VM setup script (CUDA, Docker, Toolkit setup)
	bash gcp/setup_vm.sh

gcp-deploy: ## Deploy platform on GCP VM using gcp/deploy_vm.sh
	bash gcp/deploy_vm.sh

gcp-update: ## Safely update containers while preserving Hugging Face cache
	bash gcp/update.sh

gcp-cleanup: ## Cleanup GCP resources while preserving persistent model disk
	bash gcp/cleanup.sh
