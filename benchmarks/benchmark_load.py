import argparse
import asyncio
import json
import time
import statistics
import aiohttp

async def send_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    request_id: int
):
    """
    Send a single streaming request to vLLM endpoint and record TTFT, TPOT, total duration.
    """
    start_time = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_count = 0
    tpot_list = []

    headers = {"Content-Type": "application/json"}
    
    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": f"HTTP {response.status}: {text}"
                }

            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    current_time = time.perf_counter()
                    token_count += 1

                    if first_token_time is None:
                        first_token_time = current_time
                    else:
                        if last_token_time:
                            tpot_list.append(current_time - last_token_time)

                    last_token_time = current_time

            total_duration = time.perf_counter() - start_time
            ttft = (first_token_time - start_time) if first_token_time else total_duration

            return {
                "request_id": request_id,
                "success": True,
                "total_duration": total_duration,
                "ttft": ttft,
                "mean_tpot": statistics.mean(tpot_list) if tpot_list else 0.0,
                "token_count": token_count
            }

    except Exception as e:
        return {
            "request_id": request_id,
            "success": False,
            "error": str(e)
        }

async def run_benchmark(url: str, payload_path: str, concurrency: int, total_requests: int):
    """
    Run concurrent benchmark load test against the API.
    """
    with open(payload_path, 'r') as f:
        payload_base = json.load(f)

    # Ensure stream mode is active for TTFT/TPOT measurement
    payload_base["stream"] = True
    if "model" not in payload_base:
        payload_base["model"] = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"\n=======================================================")
    print(f" Starting LLM Concurrency Benchmark")
    print(f" Endpoint Target : {url}")
    print(f" Concurrency     : {concurrency} parallel streams")
    print(f" Total Requests  : {total_requests}")
    print(f"=======================================================\n")

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=600)

    start_bench_time = time.perf_counter()
    results = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def worker(req_id: int):
            async with semaphore:
                return await send_streaming_request(session, url, payload_base, req_id)

        tasks = [worker(i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)

    total_bench_duration = time.perf_counter() - start_bench_time

    # Process metrics
    successful_results = [r for r in results if r.get("success")]
    failed_results = [r for r in results if not r.get("success")]

    if not successful_results:
        print("❌ All requests failed during benchmark run!")
        for f in failed_results[:5]:
            print(f"  Error: {f.get('error')}")
        return

    ttft_values = [r["ttft"] for r in successful_results]
    tpot_values = [r["mean_tpot"] for r in successful_results if r["mean_tpot"] > 0]
    duration_values = [r["total_duration"] for r in successful_results]
    total_tokens = sum(r["token_count"] for r in successful_results)

    def calc_percentiles(data):
        sorted_d = sorted(data)
        n = len(sorted_d)
        p50 = sorted_d[int(n * 0.50)]
        p90 = sorted_d[int(n * 0.90)] if n >= 10 else sorted_d[-1]
        p99 = sorted_d[int(n * 0.99)] if n >= 100 else sorted_d[-1]
        return p50, p90, p99

    ttft_p50, ttft_p90, ttft_p99 = calc_percentiles(ttft_values)
    dur_p50, dur_p90, dur_p99 = calc_percentiles(duration_values)
    mean_tpot = statistics.mean(tpot_values) if tpot_values else 0.0

    print("=======================================================")
    print(" BENCHMARK RESULTS SUMMARY")
    print("=======================================================")
    print(f" Total Elapsed Time   : {total_bench_duration:.2f} seconds")
    print(f" Successful Requests  : {len(successful_results)} / {total_requests}")
    print(f" Failed Requests      : {len(failed_results)}")
    print(f" Total Tokens Generated: {total_tokens} tokens")
    print(f" Output Throughput    : {total_tokens / total_bench_duration:.2f} tokens/sec")
    print(f" Request Throughput   : {len(successful_results) / total_bench_duration:.2f} req/sec")
    print("-------------------------------------------------------")
    print(" LATENCY BREAKDOWN (TTFT & TPOT)")
    print("-------------------------------------------------------")
    print(f" Time to First Token (TTFT - Prefill):")
    print(f"   P50 : {ttft_p50*1000:.2f} ms")
    print(f"   P90 : {ttft_p90*1000:.2f} ms")
    print(f"   P99 : {ttft_p99*1000:.2f} ms")
    print(f" Time Per Output Token (TPOT - Decode):")
    print(f"   Mean: {mean_tpot*1000:.2f} ms/token ({1.0/mean_tpot if mean_tpot>0 else 0:.1f} tokens/sec per stream)")
    print(f" Total End-to-End Latency:")
    print(f"   P50 : {dur_p50:.2f} sec")
    print(f"   P90 : {dur_p90:.2f} sec")
    print(f"   P99 : {dur_p99:.2f} sec")
    print("=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Inference Platform Benchmark Tool")
    parser.add_argument("--url", type=str, default="http://localhost/v1/chat/completions", help="Target API URL")
    parser.add_argument("--payload", type=str, default="benchmarks/payload.json", help="Path to JSON payload")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent streams")
    parser.add_argument("--requests", type=int, default=50, help="Total number of requests")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args.url, args.payload, args.concurrency, args.requests))
