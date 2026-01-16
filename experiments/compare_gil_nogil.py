#!/usr/bin/env python3
"""
GIL vs No-GIL Saturation Cliff Comparison Benchmark

This script measures throughput and latency across thread counts to compare
behavior between GIL-enabled and free-threading Python builds.
"""

import sys
import time
import threading
import concurrent.futures
from statistics import mean, stdev
import json
import os

def mixed_workload(cpu_ms=10, io_ms=50):
    """
    Synthetic workload simulating edge AI orchestration:
    - CPU phase: computation that holds GIL (or just CPU in nogil)
    - I/O phase: sleep that releases GIL in both modes
    """
    # CPU phase (busy loop consuming CPU time)
    start = time.thread_time()
    result = 0
    target_cpu_time = cpu_ms / 1000.0
    while (time.thread_time() - start) < target_cpu_time:
        result += 1
    
    # I/O phase (releases GIL in both modes)
    time.sleep(io_ms / 1000.0)
    return result

def benchmark(thread_count, num_tasks=500, cpu_ms=10, io_ms=50):
    """
    Benchmark throughput at given thread count.
    Submit fixed number of tasks and measure completion time.
    """
    completed_latencies = []
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        # Submit all tasks with timestamps
        futures = []
        for _ in range(num_tasks):
            submit_time = time.time()
            future = executor.submit(mixed_workload, cpu_ms, io_ms)
            futures.append((future, submit_time))
        
        # Wait for all to complete
        for future, submit_time in futures:
            try:
                future.result(timeout=60)
                latency_ms = (time.time() - submit_time) * 1000
                completed_latencies.append(latency_ms)
            except Exception:
                pass
    
    total_elapsed = time.time() - start_time
    
    # Calculate metrics
    if completed_latencies:
        tps = len(completed_latencies) / total_elapsed
        completed_sorted = sorted(completed_latencies)
        p99_idx = int(len(completed_sorted) * 0.99)
        p99_latency = completed_sorted[min(p99_idx, len(completed_sorted) - 1)]
        mean_latency = mean(completed_latencies)
    else:
        tps = 0
        p99_latency = 0
        mean_latency = 0
    
    return {
        'thread_count': thread_count,
        'tps': round(tps, 2),
        'p99_latency_ms': round(p99_latency, 2),
        'mean_latency_ms': round(mean_latency, 2),
        'total_tasks': len(completed_latencies),
        'elapsed_sec': round(total_elapsed, 2)
    }

def main():
    # Print system info
    print("=" * 70)
    print("GIL vs No-GIL Saturation Cliff Benchmark")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    
    # Check GIL status
    try:
        gil_enabled = sys._is_gil_enabled()
        print(f"GIL enabled: {gil_enabled}")
        gil_status = "gil" if gil_enabled else "nogil"
    except AttributeError:
        print("GIL enabled: True (no _is_gil_enabled, pre-3.13)")
        gil_enabled = True
        gil_status = "gil"
    
    print(f"CPU count: {os.cpu_count()}")
    print("=" * 70)
    print()
    
    # Configuration (fixed task count for fair comparison)
    num_tasks = 500
    cpu_ms = 10
    io_ms = 50
    
    # Thread counts to test
    thread_counts = [1, 4, 8, 16, 32, 64, 128, 256, 512]
    
    print(f"Configuration:")
    print(f"  Tasks per test: {num_tasks}")
    print(f"  CPU phase: {cpu_ms}ms")
    print(f"  I/O phase: {io_ms}ms")
    print(f"  Thread counts: {thread_counts}")
    print()
    print("Running saturation cliff benchmark...")
    print("-" * 70)
    print(f"{'Threads':>8} | {'TPS':>10} | {'P99 (ms)':>10} | {'Mean (ms)':>10} | {'Time (s)':>8}")
    print("-" * 70)
    
    results = []
    for tc in thread_counts:
        result = benchmark(tc, num_tasks=num_tasks, cpu_ms=cpu_ms, io_ms=io_ms)
        results.append(result)
        print(f"{result['thread_count']:>8} | {result['tps']:>10,.1f} | {result['p99_latency_ms']:>10.1f} | {result['mean_latency_ms']:>10.1f} | {result['elapsed_sec']:>8.1f}")
    
    print("-" * 70)
    print()
    
    # Analysis
    optimal = max(results, key=lambda x: x['tps'])
    
    # Find worst among high thread counts (128+)
    high_thread_results = [r for r in results if r['thread_count'] >= 128]
    worst = min(high_thread_results, key=lambda x: x['tps']) if high_thread_results else results[-1]
    
    degradation = ((optimal['tps'] - worst['tps']) / optimal['tps'] * 100) if optimal['tps'] > 0 else 0
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"OPTIMAL: {optimal['thread_count']} threads -> {optimal['tps']:,.0f} TPS")
    print(f"WORST (N>=128): {worst['thread_count']} threads -> {worst['tps']:,.0f} TPS")
    print(f"DEGRADATION: {degradation:.1f}%")
    print(f"P99 at optimal: {optimal['p99_latency_ms']:.1f}ms")
    print(f"P99 at worst: {worst['p99_latency_ms']:.1f}ms")
    print("=" * 70)
    
    # Save results
    version_str = f"{sys.version_info.major}{sys.version_info.minor}"
    filename = f"results_{gil_status}_{version_str}.json"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    output = {
        'python_version': sys.version,
        'gil_enabled': gil_enabled,
        'cpu_count': os.cpu_count(),
        'config': {
            'num_tasks': num_tasks,
            'cpu_ms': cpu_ms,
            'io_ms': io_ms
        },
        'results': results,
        'summary': {
            'optimal_threads': optimal['thread_count'],
            'optimal_tps': optimal['tps'],
            'worst_threads': worst['thread_count'],
            'worst_tps': worst['tps'],
            'degradation_pct': round(degradation, 2)
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return output

if __name__ == "__main__":
    main()
