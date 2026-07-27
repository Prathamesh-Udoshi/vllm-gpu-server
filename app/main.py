import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app.config import settings
from app.engine import llm_engine
from app.router import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle context manager for FastAPI application.
    Initializes vLLM AsyncLLMEngine on startup and shuts down gracefully.
    """
    print(f"[vLLM Platform] Starting server on {settings.HOST}:{settings.PORT}...")
    try:
        await llm_engine.initialize()
    except Exception as e:
        print(f"[vLLM Platform] Warning: Failed to initialize vLLM engine: {e}")
        print("[vLLM Platform] Ensure CUDA GPU or PyTorch CPU environment is configured.")
    yield
    print("[vLLM Platform] Shutting down inference platform...")
    await llm_engine.shutdown()

app = FastAPI(
    title="Enterprise vLLM Production Inference Platform",
    description="High-throughput OpenAI-compatible REST API powered by vLLM, Docker, Nginx, and Prometheus",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request processing time middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response

# Mount Prometheus metrics endpoint at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include OpenAI API Router
app.include_router(api_router)

# Health Checks
@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Liveness probe: verifies API server container is responsive."""
    return {"status": "live", "timestamp": int(time.time())}

@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Readiness probe: verifies vLLM engine is initialized and ready for traffic."""
    is_ready = llm_engine.engine is not None
    if is_ready:
        return {"status": "ready", "model": settings.MODEL_NAME}
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "detail": "vLLM engine initializing"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
