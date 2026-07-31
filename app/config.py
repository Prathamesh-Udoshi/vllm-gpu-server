import os
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Enterprise vLLM Platform Configuration.
    Exposes all runtime parameters via environment variables with production defaults.
    """
    # --- Model & Engine Configuration ---
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    QUANTIZATION: Optional[str] = os.getenv("QUANTIZATION", None)  # Options: 'awq', 'gptq', 'fp8', or None
    DTYPE: str = os.getenv("DTYPE", "auto")  # Options: 'auto', 'float16', 'bfloat16'
    TENSOR_PARALLEL_SIZE: int = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    GPU_MEMORY_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
    MAX_MODEL_LEN: int = int(os.getenv("MAX_MODEL_LEN", "4096"))
    MAX_NUM_SEQS: int = int(os.getenv("MAX_NUM_SEQS", "256"))
    MAX_NUM_BATCHED_TOKENS: int = int(os.getenv("MAX_NUM_BATCHED_TOKENS", "4096"))
    SWAP_SPACE: int = int(os.getenv("SWAP_SPACE", "4"))  # CPU Swap space in GB
    ENABLE_PREFIX_CACHING: bool = os.getenv("ENABLE_PREFIX_CACHING", "True").lower() in ("true", "1")
    TRUST_REMOTE_CODE: bool = os.getenv("TRUST_REMOTE_CODE", "True").lower() in ("true", "1")
    REVISION: Optional[str] = os.getenv("REVISION", None)
    KV_CACHE_DTYPE: str = os.getenv("KV_CACHE_DTYPE", "auto")  # Options: 'auto', 'fp8'
    ENFORCE_EAGER: bool = os.getenv("ENFORCE_EAGER", "False").lower() in ("true", "1")
    DOWNLOAD_DIR: str = os.getenv("DOWNLOAD_DIR", "/workspace/.cache/huggingface")
    DEVICE: str = os.getenv("DEVICE", "cuda")  # 'cuda' or 'cpu'
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)

    # --- Security & Authentication ---
    API_KEY: Optional[str] = os.getenv("API_KEY", None)  # Optional API key protection
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")

    # --- API Server Settings ---
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "True").lower() in ("true", "1")

    @property
    def cors_origins_list(self) -> List[str]:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
