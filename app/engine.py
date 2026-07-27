import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Dict, Any, List

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

from app.config import settings
from app.metrics import (
    TTFT_HISTOGRAM,
    TPOT_HISTOGRAM,
    REQUEST_LATENCY_HISTOGRAM,
    PROMPT_TOKENS_COUNTER,
    COMPLETION_TOKENS_COUNTER,
    REQUEST_COUNTER,
    KV_CACHE_USAGE_GAUGE,
    NUM_RUNNING_REQUESTS_GAUGE,
    NUM_WAITING_REQUESTS_GAUGE
)

class LLMInferenceEngine:
    """
    Wrapper around vLLM AsyncLLMEngine for production inference, metrics collection,
    and Server-Sent Events (SSE) token streaming.
    """
    def __init__(self):
        self.engine: AsyncLLMEngine | None = None
        self._stats_task: asyncio.Task | None = None

    async def initialize(self):
        """Initialize the vLLM AsyncLLMEngine with configured settings."""
        print(f"[vLLM Platform] Initializing AsyncLLMEngine for model: {settings.MODEL_NAME}")
        print(f"[vLLM Platform] Device: {settings.DEVICE} | Quantization: {settings.QUANTIZATION}")
        
        engine_args = AsyncEngineArgs(
            model=settings.MODEL_NAME,
            quantization=settings.QUANTIZATION,
            tensor_parallel_size=settings.TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=settings.GPU_MEMORY_UTILIZATION,
            max_model_len=settings.MAX_MODEL_LEN,
            max_num_seqs=settings.MAX_NUM_SEQS,
            dtype=settings.DTYPE,
            enforce_eager=settings.ENFORCE_EAGER,
            enable_prefix_caching=settings.ENABLE_PREFIX_CACHING,
            device=settings.DEVICE,
            trust_remote_code=True
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("[vLLM Platform] AsyncLLMEngine successfully initialized!")
        
        # Start background task to update vLLM metrics (KV cache usage, queue depths)
        self._stats_task = asyncio.create_task(self._monitor_engine_stats())

    async def shutdown(self):
        """Gracefully cancel background tasks on shutdown."""
        if self._stats_task:
            self._stats_task.cancel()

    async def _monitor_engine_stats(self):
        """Poll vLLM engine stats periodically to update Prometheus gauges."""
        while True:
            try:
                await asyncio.sleep(2.0)
                if self.engine:
                    # Collect stats from vLLM engine if available
                    stats = getattr(self.engine, "get_model_config", None)
                    # Update gauges if stats available from engine stats logger
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def build_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format OpenAI chat messages into a simple text prompt."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted

    async def generate_stream(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        model_name: str
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously stream generated text tokens as SSE events (`data: {...}\n\n`)
        while tracking TTFT, TPOT, throughput, and error metrics.
        """
        start_time = time.perf_counter()
        first_token_time: float | None = None
        last_token_time: float | None = None
        generated_tokens_count = 0
        prompt_tokens_count = 0

        created_timestamp = int(time.time())
        NUM_RUNNING_REQUESTS_GAUGE.inc()

        try:
            results_generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id
            )

            previous_text = ""

            async for request_output in results_generator:
                current_time = time.perf_counter()

                # Track prompt token count
                if prompt_tokens_count == 0 and hasattr(request_output, "prompt_token_ids"):
                    prompt_tokens_count = len(request_output.prompt_token_ids or [])
                    PROMPT_TOKENS_COUNTER.inc(prompt_tokens_count)

                # Process completion text output
                for output in request_output.outputs:
                    new_text = output.text[len(previous_text):]
                    previous_text = output.text

                    if new_text:
                        generated_tokens_count += 1

                        # Measure Time To First Token (TTFT)
                        if first_token_time is None:
                            first_token_time = current_time
                            ttft = first_token_time - start_time
                            TTFT_HISTOGRAM.observe(ttft)
                        else:
                            # Measure Time Per Output Token (TPOT)
                            if last_token_time:
                                tpot = current_time - last_token_time
                                TPOT_HISTOGRAM.observe(tpot)

                        last_token_time = current_time

                        # Format as OpenAI Chat Completion SSE Chunk
                        chunk_data = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": created_timestamp,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": new_text},
                                    "finish_reason": output.finish_reason
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

            # Final SSE marker
            yield "data: [DONE]\n\n"

            # Record final request metrics
            total_duration = time.perf_counter() - start_time
            REQUEST_LATENCY_HISTOGRAM.observe(total_duration)
            COMPLETION_TOKENS_COUNTER.inc(generated_tokens_count)
            REQUEST_COUNTER.labels(status="success").inc()

        except Exception as e:
            REQUEST_COUNTER.labels(status="error").inc()
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_server_error",
                    "code": 500
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            raise e

        finally:
            NUM_RUNNING_REQUESTS_GAUGE.dec()

    async def generate_non_stream(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Execute non-streaming completion request returning full JSON payload.
        """
        start_time = time.perf_counter()
        NUM_RUNNING_REQUESTS_GAUGE.inc()

        try:
            results_generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id
            )

            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            total_duration = time.perf_counter() - start_time
            REQUEST_LATENCY_HISTOGRAM.observe(total_duration)

            output_text = ""
            finish_reason = "stop"
            completion_tokens = 0
            prompt_tokens = len(final_output.prompt_token_ids) if final_output and final_output.prompt_token_ids else 0

            if final_output and final_output.outputs:
                output_text = final_output.outputs[0].text
                finish_reason = final_output.outputs[0].finish_reason or "stop"
                completion_tokens = len(final_output.outputs[0].token_ids or [])

            PROMPT_TOKENS_COUNTER.inc(prompt_tokens)
            COMPLETION_TOKENS_COUNTER.inc(completion_tokens)
            REQUEST_COUNTER.labels(status="success").inc()

            return {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": output_text
                        },
                        "finish_reason": finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
        except Exception as e:
            REQUEST_COUNTER.labels(status="error").inc()
            raise e
        finally:
            NUM_RUNNING_REQUESTS_GAUGE.dec()

# Global engine singleton instance
llm_engine = LLMInferenceEngine()
