import time
import uuid
import traceback
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app.config import settings
from app.engine import llm_engine
from app.router import router as api_router

# Configure Structured Logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [request_id=%(name)s] %(message)s"
)
logger = logging.getLogger("vllm-platform")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle context manager for FastAPI application.
    Initializes vLLM AsyncLLMEngine on startup and shuts down gracefully.
    """
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}...")

    try:
        await llm_engine.initialize()
    except Exception:
        logger.exception("Failed to initialize vLLM engine")
        traceback.print_exc()
        raise

    yield

    logger.info("Shutting down inference platform...")
    await llm_engine.shutdown()

app = FastAPI(
    title="Enterprise vLLM Production Inference Platform",
    description="High-throughput OpenAI-compatible REST API powered by vLLM, Docker, Nginx, and Prometheus",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware dynamically from settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID Tracing and Timing Middleware
@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", f"req-{uuid.uuid4().hex[:12]}")
    request.state.request_id = request_id

    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response

# Custom Exception Handler for OpenAI API Error Format
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail), "type": "invalid_request_error", "code": exc.status_code}
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": detail}
    )

# Mount Prometheus metrics endpoint if enabled
if settings.ENABLE_METRICS:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

# Include OpenAI API Router
app.include_router(api_router)

# Health Checks (Kubernetes / GCP Probes compliant)
@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Liveness probe: verifies API server container is alive."""
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

@app.get("/health/startup", tags=["Health"])
async def startup_check():
    """Startup probe: verifies application has finished initial boot."""
    return {"status": "started", "model": settings.MODEL_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
