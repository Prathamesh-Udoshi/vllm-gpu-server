from prometheus_client import Counter, Histogram, Gauge

# --- Prometheus Metrics for LLM Inference ---

# TTFT: Time To First Token (Prefill latency)
TTFT_HISTOGRAM = Histogram(
    "llm_time_to_first_token_seconds",
    "Time to first token (prefill latency) in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# TPOT: Time Per Output Token (Decode latency per token)
TPOT_HISTOGRAM = Histogram(
    "llm_time_per_output_token_seconds",
    "Time per output token (decode latency per generated token) in seconds",
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
)

# Overall End-to-End Latency
REQUEST_LATENCY_HISTOGRAM = Histogram(
    "llm_request_duration_seconds",
    "Total end-to-end request duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

# Throughput Counters
PROMPT_TOKENS_COUNTER = Counter(
    "llm_prompt_tokens_total",
    "Total number of prompt (input) tokens processed"
)

COMPLETION_TOKENS_COUNTER = Counter(
    "llm_completion_tokens_total",
    "Total number of completion (output) tokens generated"
)

# Request Counters
REQUEST_COUNTER = Counter(
    "llm_requests_total",
    "Total number of API requests received",
    ["status"]  # "success" or "error"
)

# System Status Gauges
KV_CACHE_USAGE_GAUGE = Gauge(
    "llm_kv_cache_usage_fraction",
    "Fraction of allocated KV Cache blocks currently in use (0.0 to 1.0)"
)

NUM_WAITING_REQUESTS_GAUGE = Gauge(
    "llm_num_requests_waiting",
    "Number of requests currently queued in vLLM waiting for GPU/CPU memory"
)

NUM_RUNNING_REQUESTS_GAUGE = Gauge(
    "llm_num_requests_running",
    "Number of requests currently executing in the active continuous batch"
)
