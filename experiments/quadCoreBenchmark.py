#!/usr/bin/env python3
# GIL Saturation Cliff Quad Core Benchmark.
# Simulates Raspberry Pi 4 or Jetson Nano with 4 core ARM processor.
# Demonstrates that the saturation cliff persists on multi core edge devices.

import time
import random
import statistics
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List

# Set seed for reproducibility.
SEED = 17
random.seed(SEED)

# Number of cores to simulate for Raspberry Pi 4 or Jetson Nano.
SIMULATED_CORES = 4

# Workload parameters matching single core benchmark for comparison.
CPU_ITERATIONS = 1000       # Approximately 0.1ms CPU work simulating model inference.
IO_SLEEP_MS = 0.1           # 0.1ms I/O wait simulating sensor or network latency.
TASK_COUNT = 20000          # Total tasks per configuration.
RUNS_PER_CONFIG = 3         # Multiple runs for statistical significance.

# Thread counts to test across extended range to find the cliff.
THREAD_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


@dataclass
class TaskResult:
    # Container for individual task execution metrics.
    startTime: float
    endTime: float
    
    @property
    def latency(self) -> float:
        # Calculate task latency in seconds.
        return self.endTime - self.startTime


def edgeAiTask() -> TaskResult:
    # Simulates a mixed CPU and I/O edge AI task.
    # CPU phase holds the GIL while I/O phase releases it.
    start = time.perf_counter()
    
    # CPU bound phase that holds the GIL.
    x = 0
    for _ in range(CPU_ITERATIONS):
        x += 1
    
    # I/O bound phase that releases the GIL.
    time.sleep(IO_SLEEP_MS / 1000)
    
    end = time.perf_counter()
    return TaskResult(start, end)


def pureIoTask() -> TaskResult:
    # Pure I/O task for baseline comparison without GIL contention.
    start = time.perf_counter()
    time.sleep(IO_SLEEP_MS / 1000)
    end = time.perf_counter()
    return TaskResult(start, end)


def runBenchmark(numThreads: int, taskCount: int, taskFunc) -> dict:
    # Run benchmark with specified thread count and return metrics.
    results: List[TaskResult] = []
    
    startTime = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=numThreads) as executor:
        futures = [executor.submit(taskFunc) for _ in range(taskCount)]
        for future in as_completed(futures):
            results.append(future.result())
    
    endTime = time.perf_counter()
    elapsed = endTime - startTime
    
    # Calculate latency metrics in milliseconds.
    latencies = [r.latency * 1000 for r in results]
    latencies.sort()
    
    return {
        "threads": numThreads,
        "elapsed": elapsed,
        "tps": taskCount / elapsed,
        "avgLat": statistics.mean(latencies),
        "p50Lat": latencies[len(latencies) // 2],
        "p95Lat": latencies[int(len(latencies) * 0.95)],
        "p99Lat": latencies[int(len(latencies) * 0.99)],
    }


def main():
    # Main entry point for quad core benchmark.
    
    # Apply CPU affinity constraint to simulate quad core device.
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
        print("Edge Simulation: Process pinned to 4 CPU cores (Pi 4 / Jetson Nano).")
    except (AttributeError, OSError):
        print("Warning: Could not set CPU affinity. Running on all available cores.")
    
    os.makedirs("results", exist_ok=True)
    
    print()
    print("-" * 70)
    print("GIL SATURATION CLIFF - QUAD CORE BENCHMARK")
    print("-" * 70)
    print("Configuration:")
    print(f"  Simulated Cores: {SIMULATED_CORES}.")
    print(f"  CPU Iterations: {CPU_ITERATIONS}.")
    print(f"  I/O Sleep: {IO_SLEEP_MS} ms.")
    print(f"  Tasks per config: {TASK_COUNT}.")
    print(f"  Runs per config: {RUNS_PER_CONFIG}.")
    print()
    
    # Run mixed workload experiment.
    print("Phase 1: Mixed CPU+I/O Workload")
    print("-" * 70)
    print(f"{'Threads':<8} | {'TPS':<12} | {'Avg Lat':<10} | {'P99 Lat':<10} | {'Status'}")
    print("-" * 70)
    
    mixedResults = []
    peakTps = 0
    peakThreads = 0
    
    for numThreads in THREAD_COUNTS:
        runResults = []
        for run in range(RUNS_PER_CONFIG):
            result = runBenchmark(numThreads, TASK_COUNT, edgeAiTask)
            result["run"] = run
            runResults.append(result)
            mixedResults.append(result)
        
        # Average across runs for display.
        avgTps = statistics.mean([r["tps"] for r in runResults])
        avgLat = statistics.mean([r["avgLat"] for r in runResults])
        avgP99 = statistics.mean([r["p99Lat"] for r in runResults])
        
        if avgTps > peakTps:
            peakTps = avgTps
            peakThreads = numThreads
        
        drop = ((peakTps - avgTps) / peakTps) * 100
        
        # Determine status based on degradation severity.
        if drop > 30:
            status = "CLIFF"
        elif drop > 15:
            status = "DROP"
        elif drop > 5:
            status = "DECLINE"
        else:
            status = "OK"
        
        print(f"{numThreads:<8} | {avgTps:<12,.0f} | {avgLat:<10.2f} | {avgP99:<10.2f} | {status}")
    
    # Run pure I/O baseline.
    print()
    print("Phase 2: Pure I/O Baseline")
    print("-" * 70)
    
    ioResults = []
    for numThreads in [1, 4, 16, 64, 256]:
        result = runBenchmark(numThreads, TASK_COUNT, pureIoTask)
        result["run"] = 0
        ioResults.append(result)
        print(f"{numThreads:<8} | {result['tps']:<12,.0f} | {result['avgLat']:<10.2f}")
    
    # Save results to CSV files.
    with open("results/quadcore_mixed_workload.csv", "w", newline="") as f:
        fieldnames = ["threads", "run", "tps", "avgLat", "p50Lat", "p95Lat", "p99Lat"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(mixedResults)
    
    with open("results/quadcore_io_baseline.csv", "w", newline="") as f:
        fieldnames = ["threads", "tps", "avgLat"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ioResults)
    
    # Print summary.
    print()
    print("-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    
    finalResult = [r for r in mixedResults if r["threads"] == THREAD_COUNTS[-1]]
    finalTps = statistics.mean([r["tps"] for r in finalResult])
    cliffSeverity = ((peakTps - finalTps) / peakTps) * 100
    
    print()
    print("Peak Performance:")
    print(f"  {peakTps:,.0f} TPS at {peakThreads} threads.")
    print()
    print("Cliff Analysis:")
    print(f"  Final: {finalTps:,.0f} TPS at {THREAD_COUNTS[-1]} threads.")
    print(f"  Severity: {cliffSeverity:.1f}% throughput loss.")
    print()
    print("Results saved to results/ directory.")
    
    if cliffSeverity >= 15:
        print()
        print("The OS-GIL Paradox is confirmed: cliff persists on multi-core hardware.")
    
    return 0 if cliffSeverity >= 10 else 1


if __name__ == "__main__":
    sys.exit(main())
