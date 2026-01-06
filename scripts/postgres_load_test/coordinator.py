#!/usr/bin/env python3
"""
Distributed Load Test Coordinator

Runs load tests across multiple VMs via SSH and aggregates results.

Usage:
    uv run coordinator.py \
        --vms "vm1.example.com,vm2.example.com,vm3.example.com" \
        --postgres-url "postgres://user:pass@host:5432/db" \
        --rps 1000 \
        --duration 60
"""

import json
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click


@dataclass
class VMResult:
    """Results from a single VM."""
    hostname: str
    success: bool
    error: Optional[str] = None
    data: Optional[dict] = None


def run_on_vm(
    vm: str,
    postgres_url: str,
    rps: int,
    duration: int,
    workers: int,
    output_size_kb: float,
    raw_response_size_kb: float,
    ssh_user: Optional[str],
    ssh_key: Optional[str],
    remote_path: str,
    server_side_data: bool = False,
    no_prepared_statements: bool = False,
) -> VMResult:
    """Run load test on a single VM via SSH."""

    ssh_target = f"{ssh_user}@{vm}" if ssh_user else vm

    # Build the remote command
    remote_cmd = (
        f"cd {remote_path} && "
        f"uv run load_test.py "
        f"--postgres-url '{postgres_url}' "
        f"--rps {rps} "
        f"--duration {duration} "
        f"--workers {workers} "
        f"--output-size-kb {output_size_kb} "
        f"--raw-response-size-kb {raw_response_size_kb} "
        f"--json-output"
    )
    if server_side_data:
        remote_cmd += " --server-side-data"
    if no_prepared_statements:
        remote_cmd += " --no-prepared-statements"

    # Build SSH command
    ssh_cmd = ["ssh"]
    if ssh_key:
        ssh_cmd.extend(["-i", ssh_key])
    ssh_cmd.extend(["-o", "StrictHostKeyChecking=no"])
    ssh_cmd.extend(["-o", "ConnectTimeout=10"])
    ssh_cmd.append(ssh_target)
    ssh_cmd.append(remote_cmd)

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=duration + 60,  # Allow extra time for setup/teardown
        )

        if result.returncode != 0:
            return VMResult(
                hostname=vm,
                success=False,
                error=f"SSH failed: {result.stderr}",
            )

        # Parse JSON output
        try:
            data = json.loads(result.stdout.strip())
            return VMResult(hostname=vm, success=True, data=data)
        except json.JSONDecodeError as e:
            return VMResult(
                hostname=vm,
                success=False,
                error=f"Failed to parse JSON: {e}\nOutput: {result.stdout[:500]}",
            )

    except subprocess.TimeoutExpired:
        return VMResult(hostname=vm, success=False, error="SSH timeout")
    except Exception as e:
        return VMResult(hostname=vm, success=False, error=str(e))


def aggregate_results(results: list[VMResult]) -> dict:
    """Aggregate results from multiple VMs."""

    successful_results = [r for r in results if r.success and r.data]
    failed_vms = [r for r in results if not r.success]

    if not successful_results:
        return {
            "success": False,
            "error": "No successful results",
            "failed_vms": [{"hostname": r.hostname, "error": r.error} for r in failed_vms],
        }

    # Aggregate throughput
    total_requests = sum(r.data["throughput"]["total_requests"] for r in successful_results)
    total_successful = sum(r.data["throughput"]["successful_writes"] for r in successful_results)
    total_failed = sum(r.data["throughput"]["failed_writes"] for r in successful_results)
    total_rps = sum(r.data["throughput"]["actual_rps"] for r in successful_results)

    # Aggregate latencies
    all_latencies = []
    for r in successful_results:
        if "all_latencies" in r.data:
            all_latencies.extend(r.data["all_latencies"])

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

    # Get config from first result
    config = successful_results[0].data.get("config", {})
    target_rps = config.get("target_rps", 0) * len(successful_results)

    # Calculate aggregate data written
    total_data_mb = sum(r.data.get("data_written_mb", 0) for r in successful_results)

    return {
        "success": True,
        "num_vms": len(successful_results),
        "failed_vms": [{"hostname": r.hostname, "error": r.error} for r in failed_vms],
        "config": {
            "target_rps_per_vm": config.get("target_rps", 0),
            "target_rps_total": target_rps,
            "duration_seconds": config.get("duration_seconds", 0),
            "workers_per_vm": config.get("workers", 0),
            "output_size_kb": config.get("output_size_kb", 0),
            "raw_response_size_kb": config.get("raw_response_size_kb", 0),
        },
        "throughput": {
            "total_requests": total_requests,
            "successful_writes": total_successful,
            "failed_writes": total_failed,
            "actual_rps": round(total_rps, 2),
            "target_achieved_pct": round((total_rps / target_rps) * 100, 2) if target_rps > 0 else 0,
            "success_rate": round((total_successful / max(total_requests, 1)) * 100, 2),
        },
        "latency_ms": {k: round(v, 2) for k, v in latency_stats.items()} if latency_stats else {},
        "data_written_mb": round(total_data_mb, 2),
        "per_vm_results": [
            {
                "hostname": r.data["hostname"],
                "actual_rps": r.data["throughput"]["actual_rps"],
                "success_rate": r.data["throughput"]["success_rate"],
                "latency_p50": r.data["latency_ms"].get("p50", 0),
            }
            for r in successful_results
        ],
    }


def print_aggregated_results(agg: dict):
    """Print aggregated results in human-readable format."""

    print(f"\n{'='*70}")
    print("DISTRIBUTED LOAD TEST RESULTS")
    print(f"{'='*70}")

    if not agg["success"]:
        print(f"\nERROR: {agg.get('error', 'Unknown error')}")
        if agg.get("failed_vms"):
            print("\nFailed VMs:")
            for vm in agg["failed_vms"]:
                print(f"  - {vm['hostname']}: {vm['error']}")
        return

    print(f"\nCONFIGURATION:")
    print(f"  VMs:                   {agg['num_vms']}")
    print(f"  Target RPS (per VM):   {agg['config']['target_rps_per_vm']}")
    print(f"  Target RPS (total):    {agg['config']['target_rps_total']}")
    print(f"  Duration:              {agg['config']['duration_seconds']}s")
    print(f"  Workers per VM:        {agg['config']['workers_per_vm']}")
    print(f"  Output size:           {agg['config']['output_size_kb']} KB")

    print(f"\nAGGREGATED THROUGHPUT:")
    print(f"  Total Requests:        {agg['throughput']['total_requests']:,}")
    print(f"  Successful Writes:     {agg['throughput']['successful_writes']:,}")
    print(f"  Failed Writes:         {agg['throughput']['failed_writes']:,}")
    print(f"  Actual RPS:            {agg['throughput']['actual_rps']:.1f} ({agg['throughput']['target_achieved_pct']:.1f}% of target)")
    print(f"  Success Rate:          {agg['throughput']['success_rate']:.2f}%")

    if agg["latency_ms"]:
        print(f"\nAGGREGATED LATENCY (ms):")
        print(f"  Min:                   {agg['latency_ms']['min']:.2f}")
        print(f"  Avg:                   {agg['latency_ms']['avg']:.2f}")
        print(f"  P50:                   {agg['latency_ms']['p50']:.2f}")
        print(f"  P95:                   {agg['latency_ms']['p95']:.2f}")
        print(f"  P99:                   {agg['latency_ms']['p99']:.2f}")
        print(f"  Max:                   {agg['latency_ms']['max']:.2f}")

    print(f"\nDATA WRITTEN:")
    print(f"  Total:                 {agg['data_written_mb']:.1f} MB ({agg['data_written_mb']/1024:.2f} GB)")

    print(f"\nPER-VM BREAKDOWN:")
    for vm in agg["per_vm_results"]:
        print(f"  {vm['hostname']}: {vm['actual_rps']:.1f} RPS, P50: {vm['latency_p50']:.1f}ms, Success: {vm['success_rate']:.1f}%")

    if agg.get("failed_vms"):
        print(f"\nFAILED VMs:")
        for vm in agg["failed_vms"]:
            print(f"  - {vm['hostname']}: {vm['error']}")

    print(f"\n{'='*70}")

    # Status
    if agg["throughput"]["failed_writes"] > 0:
        print(f"STATUS: FAILED - {agg['throughput']['failed_writes']} write errors")
    elif agg["throughput"]["target_achieved_pct"] >= 90:
        print(f"STATUS: PASSED - Achieved {agg['throughput']['target_achieved_pct']:.1f}% of target RPS")
    elif agg["throughput"]["target_achieved_pct"] >= 50:
        print(f"STATUS: DEGRADED - Only achieved {agg['throughput']['target_achieved_pct']:.1f}% of target RPS")
    else:
        print(f"STATUS: FAILED - Only achieved {agg['throughput']['target_achieved_pct']:.1f}% of target RPS")

    print(f"{'='*70}\n")


@click.command()
@click.option(
    '--vms',
    required=True,
    help='Comma-separated list of VM hostnames or IPs'
)
@click.option(
    '--postgres-url',
    required=True,
    help='Postgres connection URL'
)
@click.option(
    '--rps',
    default=100,
    help='Target RPS per VM [default: 100]'
)
@click.option(
    '--duration',
    default=60,
    help='Test duration in seconds [default: 60]'
)
@click.option(
    '--workers',
    default=8,
    help='Workers per VM [default: 8]'
)
@click.option(
    '--output-size-kb',
    default=1024.0,
    help='Output size in KB [default: 1024]'
)
@click.option(
    '--raw-response-size-kb',
    default=1024.0,
    help='Raw response size in KB [default: 1024]'
)
@click.option(
    '--ssh-user',
    default=None,
    help='SSH username (default: current user)'
)
@click.option(
    '--ssh-key',
    default=None,
    help='Path to SSH private key'
)
@click.option(
    '--remote-path',
    default='~/postgres_load_test',
    help='Path to load test script on remote VMs [default: ~/postgres_load_test]'
)
@click.option(
    '--json-output',
    is_flag=True,
    help='Output aggregated results as JSON'
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
def main(vms, postgres_url, rps, duration, workers, output_size_kb, raw_response_size_kb,
         ssh_user, ssh_key, remote_path, json_output, server_side_data, no_prepared_statements):
    """
    Run distributed Postgres load test across multiple VMs.

    Prerequisites:
    - SSH access to all VMs (key-based auth recommended)
    - load_test.py and dependencies installed on each VM at --remote-path
    - uv available on each VM
    """

    vm_list = [v.strip() for v in vms.split(',') if v.strip()]

    if not vm_list:
        print("Error: No VMs specified")
        sys.exit(1)

    if not json_output:
        print(f"\n{'='*70}")
        print("DISTRIBUTED LOAD TEST")
        print(f"{'='*70}")
        print(f"VMs:              {len(vm_list)}")
        print(f"Target RPS/VM:    {rps}")
        print(f"Target RPS total: {rps * len(vm_list)}")
        print(f"Duration:         {duration}s")
        print(f"Workers/VM:       {workers}")
        print(f"Data generation:  {'server-side (SQL)' if server_side_data else 'client-side (network)'}")
        print(f"{'='*70}\n")
        print("Starting tests on all VMs...")

    # Run tests in parallel
    results: list[VMResult] = []
    threads = []

    def run_and_collect(vm):
        result = run_on_vm(
            vm=vm,
            postgres_url=postgres_url,
            rps=rps,
            duration=duration,
            workers=workers,
            output_size_kb=output_size_kb,
            raw_response_size_kb=raw_response_size_kb,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            remote_path=remote_path,
            server_side_data=server_side_data,
            no_prepared_statements=no_prepared_statements,
        )
        results.append(result)
        if not json_output:
            status = "✓" if result.success else "✗"
            print(f"  [{status}] {vm}: {'Done' if result.success else result.error}")

    for vm in vm_list:
        t = threading.Thread(target=run_and_collect, args=(vm,))
        t.start()
        threads.append(t)

    # Wait for all to complete
    for t in threads:
        t.join()

    # Aggregate and print results
    aggregated = aggregate_results(results)

    if json_output:
        print(json.dumps(aggregated, indent=2))
    else:
        print_aggregated_results(aggregated)


if __name__ == "__main__":
    main()
