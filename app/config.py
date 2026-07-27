import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Platform and Engine Configuration.
    Uses environment variables with fallback defaults.
    """
    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    QUANTIZATION: str | None = os.getenv("QUANTIZATION", None)  # Options: 'awq', 'gptq', 'fp8', or None
    TENSOR_PARALLEL_SIZE: int = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    GPU_MEMORY_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
    MAX_MODEL_LEN: int = int(os.getenv("MAX_MODEL_LEN", "4096"))
    MAX_NUM_SEQS: int = int(os.getenv("MAX_NUM_SEQS", "256"))
    DTYPE: str = os.getenv("DTYPE", "auto")
    ENFORCE_EAGER: bool = os.getenv("ENFORCE_EAGER", "False").lower() in ("true", "1")
    ENABLE_PREFIX_CACHING: bool = os.getenv("ENABLE_PREFIX_CACHING", "True").lower() in ("true", "1")
    DEVICE: str = os.getenv("DEVICE", "cuda")  # 'cuda' or 'cpu'

    # API Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
