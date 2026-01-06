#!/usr/bin/env python3
"""
Postgres Load Test for TensorZero

Simulates inference request database writes to measure Postgres performance.
Tests the write patterns that would occur in a Postgres-only TensorZero deployment.

Uses multiprocessing to avoid Python GIL limitations.

Usage:
    uv run load_test.py --rps 1000 --duration 60 --workers 8
"""

import multiprocessing as mp
import random
import string
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import Process, Queue
from typing import Optional

import click
import orjson
import psycopg  # psycopg3 - sync driver, no asyncio overhead

# Use uuid6 for uuid7 support (Python < 3.13)
try:
    from uuid import uuid7
except ImportError:
    from uuid6 import uuid7


@dataclass
class LoadTestConfig:
    """Configuration for the load test."""
    postgres_url: str
    requests_per_second: int
    duration_seconds: int
    num_workers: int
    include_rate_limiting: bool = True
    include_experimentation: bool = True
    verbose: bool = False
    # Data sizes in KB
    input_size_kb: float = 1.5
    output_size_kb: float = 2048.0      # 2MB
    raw_request_size_kb: float = 1.0
    raw_response_size_kb: float = 2048.0  # 2MB
    # Docker monitoring
    docker_container: Optional[str] = None
    # Output format
    json_output: bool = False
    # Server-side data generation (no network transfer)
    server_side_data: bool = False
    # Disable prepared statements (required for PgBouncer transaction mode)
    no_prepared_statements: bool = False


# Pre-generate random data to avoid CPU overhead during test
RANDOM_POOL_SIZE_SMALL = 100
RANDOM_POOL_SIZE_LARGE = 10
_json_cache: dict[int, list[str]] = {}


def _get_pool_size(size_bytes: int) -> int:
    return RANDOM_POOL_SIZE_LARGE if size_bytes >= 100 * 1024 else RANDOM_POOL_SIZE_SMALL


def _generate_json_pool(size_kb: float) -> list[str]:
    """Generate a pool of pre-serialized JSON strings."""
    target_bytes = int(size_kb * 1024)
    content_length = max(100, target_bytes - 50)
    pool_size = _get_pool_size(target_bytes)

    pool = []
    for _ in range(pool_size):
        content = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=content_length))
        data = {"messages": [{"role": "user", "content": content}]}
        pool.append(orjson.dumps(data).decode('utf-8'))
    return pool


def get_random_json_str(size_kb: float) -> str:
    """Get a pre-serialized JSON string of approximately the given size."""
    cache_key = int(size_kb * 1024)
    cache_key = (cache_key // 1024) * 1024 or 1024

    if cache_key not in _json_cache:
        _json_cache[cache_key] = _generate_json_pool(size_kb)

    return random.choice(_json_cache[cache_key])


def prewarm_cache(config: LoadTestConfig) -> float:
    """Pre-warm the JSON cache and return estimated memory in MB."""
    total_mb = 0
    for size_kb in [config.input_size_kb, config.output_size_kb,
                    config.raw_request_size_kb, config.raw_response_size_kb]:
        cache_key = int(size_kb * 1024)
        cache_key = (cache_key // 1024) * 1024 or 1024
        pool_size = _get_pool_size(cache_key)
        if cache_key not in _json_cache:
            total_mb += (cache_key * pool_size) / (1024 * 1024)
        get_random_json_str(size_kb)
    return total_mb


def generate_inference_data(config: LoadTestConfig) -> tuple[dict, dict, "uuid7"]:
    """Generate realistic inference data for one simulated request."""
    inference_id = uuid7()
    episode_id = uuid7()
    model_inference_id = uuid7()
    timestamp = datetime.now(timezone.utc)

    chat_inference = {
        "id": inference_id,
        "function_name": f"function_{random.randint(1, 10)}",
        "variant_name": f"variant_{random.randint(1, 5)}",
        "episode_id": episode_id,
        "input_json": get_random_json_str(config.input_size_kb),
        "output_json": get_random_json_str(config.output_size_kb),
        "inference_params": {
            "temperature": round(random.uniform(0.0, 1.0), 2),
            "max_tokens": random.randint(100, 4000),
        },
        "tags": {
            "env": random.choice(["production", "staging", "development"]),
            "user_id": str(uuid7()),
            "request_id": str(uuid7()),
        },
        "processing_time_ms": random.randint(100, 3000),
        "timestamp": timestamp,
    }

    model_inference = {
        "id": model_inference_id,
        "inference_id": inference_id,
        "raw_request": get_random_json_str(config.raw_request_size_kb),
        "raw_response": get_random_json_str(config.raw_response_size_kb),
        "model_name": random.choice([
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
            "claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"
        ]),
        "model_provider_name": random.choice(["openai", "anthropic", "azure"]),
        "input_tokens": random.randint(100, 8000),
        "output_tokens": random.randint(50, 4000),
        "response_time_ms": random.randint(200, 5000),
        "input_messages": [{"role": "user", "content": "test message"}],
        "output": {"content": "model response content"},
        "timestamp": timestamp,
    }

    return chat_inference, model_inference, episode_id


def write_inference_sync(conn, config: LoadTestConfig) -> float:
    """Synchronous write using psycopg3. Returns DB latency in ms."""
    # Generate data outside timing to measure only DB latency
    chat_data, model_data, episode_id = generate_inference_data(config)

    start = time.perf_counter()
    with conn.cursor() as cur:
        with conn.transaction():
            # 1. Write chat_inference
            cur.execute(
                """
                INSERT INTO chat_inference
                (id, function_name, variant_name, episode_id, input, output,
                 inference_params, tags, processing_time_ms, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(chat_data["id"]),
                    chat_data["function_name"],
                    chat_data["variant_name"],
                    str(chat_data["episode_id"]),
                    chat_data["input_json"],
                    chat_data["output_json"],
                    orjson.dumps(chat_data["inference_params"]).decode(),
                    orjson.dumps(chat_data["tags"]).decode(),
                    chat_data["processing_time_ms"],
                    chat_data["timestamp"],
                ),
            )

            # 2. Write model_inference
            cur.execute(
                """
                INSERT INTO model_inference
                (id, inference_id, raw_request, raw_response, model_name,
                 model_provider_name, input_tokens, output_tokens, response_time_ms,
                 input_messages, output, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(model_data["id"]),
                    str(model_data["inference_id"]),
                    model_data["raw_request"],
                    model_data["raw_response"],
                    model_data["model_name"],
                    model_data["model_provider_name"],
                    model_data["input_tokens"],
                    model_data["output_tokens"],
                    model_data["response_time_ms"],
                    orjson.dumps(model_data["input_messages"]).decode(),
                    orjson.dumps(model_data["output"]).decode(),
                    model_data["timestamp"],
                ),
            )

            # 3. Rate limiting UPSERT
            if config.include_rate_limiting:
                rate_limit_key = f"rate_limit:{chat_data['function_name']}"
                cur.execute(
                    """
                    INSERT INTO resource_bucket (key, tickets, balance_as_of)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (key) DO UPDATE SET
                        tickets = resource_bucket.tickets - 1,
                        balance_as_of = EXCLUDED.balance_as_of
                    """,
                    (rate_limit_key, 1000, datetime.now(timezone.utc)),
                )

            # 4. Experimentation UPSERT
            if config.include_experimentation:
                cur.execute(
                    """
                    INSERT INTO variant_by_episode (function_name, episode_id, variant_name)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (function_name, episode_id) DO UPDATE SET
                        variant_name = EXCLUDED.variant_name
                    """,
                    (chat_data["function_name"], str(episode_id), chat_data["variant_name"]),
                )

    return (time.perf_counter() - start) * 1000


def write_inference_server_side(conn, config: LoadTestConfig) -> float:
    """Write using server-side data generation. Returns DB latency in ms.

    Generates all random data using SQL functions, eliminating network transfer overhead.
    This measures pure database write throughput.
    """
    # Calculate repeat counts for md5 strings (each md5 is 32 chars)
    # We want approximate sizes matching the config
    input_repeats = max(1, int(config.input_size_kb * 1024 / 32))
    output_repeats = max(1, int(config.output_size_kb * 1024 / 32))
    raw_request_repeats = max(1, int(config.raw_request_size_kb * 1024 / 32))
    raw_response_repeats = max(1, int(config.raw_response_size_kb * 1024 / 32))

    start = time.perf_counter()
    with conn.cursor() as cur:
        with conn.transaction():
            # Generate UUIDs once for reuse
            cur.execute("SELECT gen_random_uuid(), gen_random_uuid(), gen_random_uuid()")
            inference_id, episode_id, model_inference_id = cur.fetchone()

            # 1. Write chat_inference with server-generated data
            cur.execute(
                """
                INSERT INTO chat_inference
                (id, function_name, variant_name, episode_id, input, output,
                 inference_params, tags, processing_time_ms, timestamp)
                VALUES (
                    %s,
                    'function_' || (floor(random() * 10) + 1)::int,
                    'variant_' || (floor(random() * 5) + 1)::int,
                    %s,
                    jsonb_build_object('messages', jsonb_build_array(
                        jsonb_build_object('role', 'user', 'content', repeat(md5(random()::text), %s))
                    )),
                    jsonb_build_object('messages', jsonb_build_array(
                        jsonb_build_object('role', 'assistant', 'content', repeat(md5(random()::text), %s))
                    )),
                    '{"temperature": 0.7, "max_tokens": 1000}'::jsonb,
                    jsonb_build_object('env', 'production', 'user_id', gen_random_uuid()::text),
                    floor(random() * 3000)::int,
                    now()
                )
                """,
                (str(inference_id), str(episode_id), input_repeats, output_repeats),
            )

            # 2. Write model_inference with server-generated data
            cur.execute(
                """
                INSERT INTO model_inference
                (id, inference_id, raw_request, raw_response, model_name,
                 model_provider_name, input_tokens, output_tokens, response_time_ms,
                 input_messages, output, timestamp)
                VALUES (
                    %s,
                    %s,
                    repeat(md5(random()::text), %s),
                    repeat(md5(random()::text), %s),
                    (ARRAY['gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet', 'claude-3-opus'])[floor(random() * 4 + 1)::int],
                    (ARRAY['openai', 'anthropic', 'azure'])[floor(random() * 3 + 1)::int],
                    floor(random() * 8000 + 100)::int,
                    floor(random() * 4000 + 50)::int,
                    floor(random() * 5000 + 200)::int,
                    '[{"role": "user", "content": "test"}]'::jsonb,
                    '{"content": "response"}'::jsonb,
                    now()
                )
                """,
                (str(model_inference_id), str(inference_id), raw_request_repeats, raw_response_repeats),
            )

            # 3. Rate limiting UPSERT
            if config.include_rate_limiting:
                cur.execute(
                    """
                    INSERT INTO resource_bucket (key, tickets, balance_as_of)
                    VALUES (
                        'rate_limit:function_' || (floor(random() * 10) + 1)::int,
                        1000,
                        now()
                    )
                    ON CONFLICT (key) DO UPDATE SET
                        tickets = resource_bucket.tickets - 1,
                        balance_as_of = EXCLUDED.balance_as_of
                    """,
                )

            # 4. Experimentation UPSERT
            if config.include_experimentation:
                cur.execute(
                    """
                    INSERT INTO variant_by_episode (function_name, episode_id, variant_name)
                    VALUES (
                        'function_' || (floor(random() * 10) + 1)::int,
                        %s,
                        'variant_' || (floor(random() * 5) + 1)::int
                    )
                    ON CONFLICT (function_name, episode_id) DO UPDATE SET
                        variant_name = EXCLUDED.variant_name
                    """,
                    (str(episode_id),),
                )

    return (time.perf_counter() - start) * 1000


def worker_process(
    worker_id: int,
    config: LoadTestConfig,
    result_queue: Queue,
    start_event: mp.Event,
    stop_event: mp.Event,
):
    """Worker process that runs writes until stopped."""
    # Pre-warm cache in this process (only needed for client-side data)
    if not config.server_side_data:
        prewarm_cache(config)

    # Select the write function based on mode
    write_fn = write_inference_server_side if config.server_side_data else write_inference_sync

    # Connect to Postgres
    conn = connect_postgres(config)

    # Wait for start signal
    start_event.wait()

    successful = 0
    failed = 0
    latencies = []

    while not stop_event.is_set():
        try:
            latency = write_fn(conn, config)
            successful += 1
            latencies.append(latency)
        except Exception as e:
            failed += 1
            if config.verbose:
                print(f"[Worker {worker_id}] Error: {e}")

    conn.close()

    # Send results back
    result_queue.put({
        "worker_id": worker_id,
        "successful": successful,
        "failed": failed,
        "latencies": latencies,
    })


def get_docker_stats(container_name: str) -> tuple[Optional[float], Optional[float]]:
    """Get CPU and memory usage from a Docker container."""
    try:
        result = subprocess.run(
            ["docker", "stats", container_name, "--no-stream", "--format", "{{.CPUPerc}},{{.MemUsage}}"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            cpu_str = parts[0].replace("%", "")
            mem_str = parts[1].split("/")[0].strip()

            cpu = float(cpu_str)

            mem_value = float(mem_str[:-3])
            mem_unit = mem_str[-3:]
            if mem_unit == "GiB":
                mem_value *= 1024
            elif mem_unit == "KiB":
                mem_value /= 1024

            return cpu, mem_value
    except Exception:
        pass
    return None, None


def connect_postgres(config: LoadTestConfig):
    """Create a Postgres connection with appropriate settings."""
    if config.no_prepared_statements:
        # Disable prepared statements for PgBouncer transaction mode compatibility
        return psycopg.connect(config.postgres_url, prepare_threshold=0)
    return psycopg.connect(config.postgres_url)


def truncate_tables(config: LoadTestConfig):
    """Truncate all test tables before starting the load test."""
    conn = connect_postgres(config)
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE chat_inference, model_inference, resource_bucket, variant_by_episode")
        conn.commit()
    finally:
        conn.close()


def run_load_test(config: LoadTestConfig):
    """Run the load test with multiple worker processes."""

    # Truncate tables before starting
    if not config.json_output:
        print("Truncating tables...")
    truncate_tables(config)

    # Pre-warm cache in main process to measure memory (only for client-side data)
    cache_mb = 0
    if not config.server_side_data:
        if not config.json_output:
            print("Pre-generating random JSON data (pre-serialized)...")
        cache_mb = prewarm_cache(config)
        if not config.json_output:
            print(f"  Cache size per process: ~{cache_mb:.0f} MB")
            print(f"  Total cache memory ({config.num_workers} workers): ~{cache_mb * config.num_workers:.0f} MB")
    elif not config.json_output:
        print("Using server-side data generation (no network transfer)")

    data_per_request_kb = (config.input_size_kb + config.output_size_kb +
                           config.raw_request_size_kb + config.raw_response_size_kb)

    if not config.json_output:
        print(f"\n{'='*60}")
        print("POSTGRES LOAD TEST FOR TENSORZERO")
        print(f"{'='*60}")
        print(f"Target RPS:          {config.requests_per_second}")
        print(f"Duration:            {config.duration_seconds}s")
        print(f"Worker processes:    {config.num_workers}")
        print(f"Data per request:    ~{data_per_request_kb:.1f} KB ({data_per_request_kb/1024:.1f} MB)")
        print(f"Data generation:     {'server-side (SQL)' if config.server_side_data else 'client-side (network)'}")
        print(f"Rate limiting:       {'enabled' if config.include_rate_limiting else 'disabled'}")
        print(f"Experimentation:     {'enabled' if config.include_experimentation else 'disabled'}")
        if config.docker_container:
            print(f"Docker container:    {config.docker_container}")
        print(f"{'='*60}\n")

    # Create synchronization primitives
    result_queue = Queue()
    start_event = mp.Event()
    stop_event = mp.Event()

    # Start worker processes
    if not config.json_output:
        print(f"Starting {config.num_workers} worker processes...")
    workers = []
    for i in range(config.num_workers):
        p = Process(
            target=worker_process,
            args=(i, config, result_queue, start_event, stop_event),
        )
        p.start()
        workers.append(p)

    # Give workers time to connect
    time.sleep(1)

    # Collect docker stats during test
    docker_cpu_samples = []
    docker_mem_samples = []

    # Start all workers
    if not config.json_output:
        print("Starting test...")
    start_time = time.time()
    start_event.set()

    # Monitor progress
    last_progress = start_time
    while time.time() - start_time < config.duration_seconds:
        time.sleep(1)

        if config.docker_container:
            cpu, mem = get_docker_stats(config.docker_container)
            if cpu is not None:
                docker_cpu_samples.append(cpu)
            if mem is not None:
                docker_mem_samples.append(mem)

        current_time = time.time()
        if current_time - last_progress >= 5:
            elapsed = current_time - start_time
            if not config.json_output:
                progress_msg = f"  Progress: {elapsed:.0f}s elapsed"
                if config.docker_container and docker_cpu_samples:
                    progress_msg += f", CPU: {docker_cpu_samples[-1]:.1f}%"
                if config.docker_container and docker_mem_samples:
                    progress_msg += f", Mem: {docker_mem_samples[-1]:.0f}MiB"
                print(progress_msg)
            last_progress = current_time

    # Stop workers
    stop_event.set()
    actual_duration = time.time() - start_time

    if not config.json_output:
        print("Waiting for workers to finish...")
    for p in workers:
        p.join(timeout=5)

    # Collect results
    total_successful = 0
    total_failed = 0
    all_latencies = []

    while not result_queue.empty():
        result = result_queue.get()
        total_successful += result["successful"]
        total_failed += result["failed"]
        all_latencies.extend(result["latencies"])

    # Calculate metrics
    total_requests = total_successful + total_failed
    actual_rps = total_requests / actual_duration
    success_rate = (total_successful / max(total_requests, 1)) * 100
    target_achieved_pct = (actual_rps / config.requests_per_second) * 100

    # Calculate latency stats
    latency_stats = {}
    if all_latencies:
        sorted_latencies = sorted(all_latencies)
        n = len(sorted_latencies)
        latency_stats = {
            "min": sorted_latencies[0],
            "avg": sum(all_latencies) / n,
            "p50": sorted_latencies[int(n * 0.50)],
            "p95": sorted_latencies[int(n * 0.95)] if n > 20 else sorted_latencies[-1],
            "p99": sorted_latencies[int(n * 0.99)] if n > 100 else sorted_latencies[-1],
            "max": sorted_latencies[-1],
        }

    data_written_mb = (total_successful * data_per_request_kb) / 1024

    # Determine status
    if total_failed > 0:
        status = "FAILED"
    elif target_achieved_pct >= 90:
        status = "PASSED"
    elif target_achieved_pct >= 50:
        status = "DEGRADED"
    else:
        status = "FAILED"

    # JSON output mode
    if config.json_output:
        import socket
        results = {
            "hostname": socket.gethostname(),
            "config": {
                "target_rps": config.requests_per_second,
                "duration_seconds": config.duration_seconds,
                "workers": config.num_workers,
                "input_size_kb": config.input_size_kb,
                "output_size_kb": config.output_size_kb,
                "raw_request_size_kb": config.raw_request_size_kb,
                "raw_response_size_kb": config.raw_response_size_kb,
                "server_side_data": config.server_side_data,
            },
            "throughput": {
                "total_requests": total_requests,
                "successful_writes": total_successful,
                "failed_writes": total_failed,
                "actual_rps": round(actual_rps, 2),
                "target_achieved_pct": round(target_achieved_pct, 2),
                "success_rate": round(success_rate, 2),
            },
            "latency_ms": {k: round(v, 2) for k, v in latency_stats.items()} if latency_stats else {},
            "data_written_mb": round(data_written_mb, 2),
            "status": status,
            "all_latencies": all_latencies,  # Include raw latencies for aggregation
        }
        print(orjson.dumps(results).decode())
        return

    # Human-readable output
    print(f"\n{'='*60}")
    print("LOAD TEST RESULTS")
    print(f"{'='*60}")

    print(f"\nTHROUGHPUT:")
    print(f"  Total Requests:     {total_requests:,}")
    print(f"  Successful Writes:  {total_successful:,}")
    print(f"  Failed Writes:      {total_failed:,}")
    print(f"  Target RPS:         {config.requests_per_second}")
    print(f"  Actual RPS:         {actual_rps:.1f} ({target_achieved_pct:.1f}% of target)")
    print(f"  Success Rate:       {success_rate:.2f}%")

    if latency_stats:
        print(f"\nLATENCY (ms):")
        print(f"  Min:                {latency_stats['min']:.2f}")
        print(f"  Avg:                {latency_stats['avg']:.2f}")
        print(f"  P50:                {latency_stats['p50']:.2f}")
        print(f"  P95:                {latency_stats['p95']:.2f}")
        print(f"  P99:                {latency_stats['p99']:.2f}")
        print(f"  Max:                {latency_stats['max']:.2f}")

    if docker_cpu_samples or docker_mem_samples:
        print(f"\nDOCKER CONTAINER ({config.docker_container}):")
        if docker_cpu_samples:
            print(f"  CPU Avg:            {sum(docker_cpu_samples)/len(docker_cpu_samples):.1f}%")
            print(f"  CPU Max:            {max(docker_cpu_samples):.1f}%")
        if docker_mem_samples:
            print(f"  Memory Avg:         {sum(docker_mem_samples)/len(docker_mem_samples):.0f} MiB")
            print(f"  Memory Max:         {max(docker_mem_samples):.0f} MiB")

    print(f"\nDATA WRITTEN:")
    print(f"  Estimated:          {data_written_mb:.1f} MB ({data_written_mb/1024:.2f} GB)")

    print(f"\n{'='*60}")

    if status == "FAILED" and total_failed > 0:
        print(f"STATUS: FAILED - {total_failed} write errors")
    elif status == "PASSED":
        print(f"STATUS: PASSED - Achieved {target_achieved_pct:.1f}% of target RPS")
    elif status == "DEGRADED":
        print(f"STATUS: DEGRADED - Only achieved {target_achieved_pct:.1f}% of target RPS")
    else:
        print(f"STATUS: FAILED - Only achieved {target_achieved_pct:.1f}% of target RPS")

    if target_achieved_pct < 90 and latency_stats:
        print(f"\nDIAGNOSIS HINTS:")
        theoretical_max_rps = (config.num_workers * 1000) / latency_stats['avg']
        print(f"  Theoretical max RPS with {config.num_workers} workers @ {latency_stats['avg']:.1f}ms latency: {theoretical_max_rps:.0f}")

        if theoretical_max_rps < config.requests_per_second:
            needed_workers = int((config.requests_per_second * latency_stats['avg']) / 1000) + 1
            print(f"  To achieve {config.requests_per_second} RPS, try --workers {needed_workers}")

        if latency_stats['avg'] > 50:
            print(f"  High latency detected. Consider:")
            print(f"    - Check Postgres connection (network latency)")
            print(f"    - Check Postgres config (synchronous_commit=off)")
            print(f"    - Reduce data size with --output-size-kb, --raw-response-size-kb")

    print(f"{'='*60}\n")


@click.command()
@click.option(
    '--postgres-url',
    default='postgresql://postgres:postgres@localhost:5432/tensorzero_loadtest',
    help='Postgres connection URL'
)
@click.option(
    '--rps',
    default=100,
    help='Target requests per second'
)
@click.option(
    '--duration',
    default=60,
    help='Test duration in seconds'
)
@click.option(
    '--workers',
    default=8,
    help='Number of worker processes (default: 8)'
)
@click.option(
    '--no-rate-limiting',
    is_flag=True,
    help='Skip rate limiting writes'
)
@click.option(
    '--no-experimentation',
    is_flag=True,
    help='Skip experimentation writes'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Print verbose error messages'
)
@click.option(
    '--input-size-kb',
    default=1.5,
    help='Size of input JSON in KB [default: 1.5]'
)
@click.option(
    '--output-size-kb',
    default=2048.0,
    help='Size of output JSON in KB [default: 2048 (2MB)]'
)
@click.option(
    '--raw-request-size-kb',
    default=1.0,
    help='Size of raw_request JSON in KB [default: 1.0]'
)
@click.option(
    '--raw-response-size-kb',
    default=2048.0,
    help='Size of raw_response JSON in KB [default: 2048 (2MB)]'
)
@click.option(
    '--docker-container',
    default=None,
    help='Docker container name to monitor (e.g., "postgres")'
)
@click.option(
    '--json-output',
    is_flag=True,
    help='Output results as JSON (for aggregation across VMs)'
)
@click.option(
    '--server-side-data',
    is_flag=True,
    help='Generate random data in SQL (eliminates network transfer overhead)'
)
@click.option(
    '--no-prepared-statements',
    is_flag=True,
    help='Disable prepared statements (required for PgBouncer transaction mode)'
)
def main(postgres_url, rps, duration, workers, no_rate_limiting, no_experimentation,
         verbose, input_size_kb, output_size_kb, raw_request_size_kb, raw_response_size_kb,
         docker_container, json_output, server_side_data, no_prepared_statements):
    """
    Run a Postgres load test simulating TensorZero inference writes.

    Uses multiprocessing for true parallelism (no GIL limitations).
    """
    config = LoadTestConfig(
        postgres_url=postgres_url,
        requests_per_second=rps,
        duration_seconds=duration,
        num_workers=workers,
        include_rate_limiting=not no_rate_limiting,
        include_experimentation=not no_experimentation,
        verbose=verbose,
        input_size_kb=input_size_kb,
        output_size_kb=output_size_kb,
        raw_request_size_kb=raw_request_size_kb,
        raw_response_size_kb=raw_response_size_kb,
        docker_container=docker_container,
        json_output=json_output,
        server_side_data=server_side_data,
        no_prepared_statements=no_prepared_statements,
    )

    try:
        run_load_test(config)
    except KeyboardInterrupt:
        if not json_output:
            print("\n\nTest interrupted by user")
    except Exception as e:
        if not json_output:
            print(f"\n\nTest failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
