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
from typing import List, Dict, Tuple
from scipy import stats
import psutil

# Set seed for reproducibility.
SEED = 17
random.seed(SEED)

# Number of cores to simulate for Raspberry Pi 4 or Jetson Nano.
SIMULATED_CORES = 4

# Workload parameters matching single core benchmark for comparison.
CPU_ITERATIONS = 1000       # Approximately 0.1ms CPU work simulating model inference.
IO_SLEEP_MS = 0.1           # 0.1ms I/O wait simulating sensor or network latency.
TASK_COUNT = 20000          # Total tasks per configuration.
RUNS_PER_CONFIG = 10        # n=10 runs for statistical significance (Section 3.3).
CONFIDENCE_LEVEL = 0.95     # 95% confidence interval.

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


@dataclass
class AggregatedResult:
    # Container for aggregated results with confidence intervals.
    threads: int
    tpsMean: float
    tpsMargin: float
    tpsCV: float
    p99Pooled: float
    p99Median: float
    p99IQR: float
    nRuns: int


def computeCI(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute mean and CI margin using t-distribution (Section 3.3)."""
    n = len(data)
    if n < 2:
        return (data[0] if data else 0.0, 0.0)
    mean = statistics.mean(data)
    std = statistics.stdev(data)
    stderr = std / (n ** 0.5)
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * stderr
    return (mean, margin)


def computePooledP99(allLatencies: List[List[float]]) -> float:
    """Compute pooled P99 by aggregating per-task samples across runs (Section 3.3)."""
    pooled = []
    for runLats in allLatencies:
        pooled.extend(runLats)
    pooled.sort()
    idx = int(len(pooled) * 0.99)
    return pooled[idx] * 1000  # Convert to ms


def computeP99Stats(allLatencies: List[List[float]]) -> Tuple[float, float]:
    """Compute per-run P99 median ± IQR (Section 3.3)."""
    perRunP99 = []
    for runLats in allLatencies:
        sortedLats = sorted(runLats)
        idx = int(len(sortedLats) * 0.99)
        perRunP99.append(sortedLats[idx] * 1000)
    perRunP99.sort()
    n = len(perRunP99)
    median = statistics.median(perRunP99)
    q1 = perRunP99[n // 4]
    q3 = perRunP99[3 * n // 4]
    iqr = q3 - q1
    return (median, iqr)


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
    
    # Keep raw latencies for pooled P99.
    rawLatencies = [r.latency for r in results]
    
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
        "rawLatencies": rawLatencies,
    }


def main():
    # Main entry point for quad core benchmark.
    
    # Apply CPU affinity constraint to simulate quad core device.
    # Use psutil for cross-platform support (Windows + Linux).
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
        print("Edge Simulation: Process pinned to 4 CPU cores (Linux).")
    except (AttributeError, OSError):
        try:
            p = psutil.Process()
            p.cpu_affinity([0, 1, 2, 3])
            print("Edge Simulation: Process pinned to 4 CPU cores (psutil).")
        except Exception as e:
            print(f"Warning: Could not set CPU affinity: {e}")
            print("Running on all available cores.")
    
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
    print(f"  Runs per config: {RUNS_PER_CONFIG} (n=10 as per Section 3.3).")
    print(f"  Confidence: {CONFIDENCE_LEVEL*100:.0f}% CI using t-distribution.")
    print()
    
    # Run mixed workload experiment.
    print("Phase 1: Mixed CPU+I/O Workload")
    print("-" * 70)
    print(f"{'Threads':<8} | {'TPS (mean±CI)':<20} | {'P99 Pooled':<12} | {'P99 Med±IQR':<15} | {'Status'}")
    print("-" * 70)
    
    mixedResults = []
    aggregated: Dict[int, AggregatedResult] = {}
    peakTps = 0
    peakThreads = 0
    
    for numThreads in THREAD_COUNTS:
        runResults = []
        allLatencies = []
        for run in range(RUNS_PER_CONFIG):
            result = runBenchmark(numThreads, TASK_COUNT, edgeAiTask)
            result["run"] = run
            runResults.append(result)
            mixedResults.append(result)
            allLatencies.append(result["rawLatencies"])
        
        # Compute statistics as per Section 3.3.
        tpsList = [r["tps"] for r in runResults]
        tpsMean, tpsMargin = computeCI(tpsList, CONFIDENCE_LEVEL)
        tpsCV = (statistics.stdev(tpsList) / tpsMean * 100) if tpsMean > 0 else 0
        
        p99Pooled = computePooledP99(allLatencies)
        p99Median, p99IQR = computeP99Stats(allLatencies)
        
        aggregated[numThreads] = AggregatedResult(
            threads=numThreads,
            tpsMean=tpsMean,
            tpsMargin=tpsMargin,
            tpsCV=tpsCV,
            p99Pooled=p99Pooled,
            p99Median=p99Median,
            p99IQR=p99IQR,
            nRuns=RUNS_PER_CONFIG,
        )
        
        if tpsMean > peakTps:
            peakTps = tpsMean
            peakThreads = numThreads
        
        drop = ((peakTps - tpsMean) / peakTps) * 100
        
        # Determine status based on degradation severity.
        if drop > 30:
            status = "CLIFF"
        elif drop > 15:
            status = "DROP"
        elif drop > 5:
            status = "DECLINE"
        else:
            status = "OK"
        
        tpsStr = f"{tpsMean:,.0f}±{tpsMargin:,.0f}"
        p99IqrStr = f"{p99Median:.1f}±{p99IQR:.1f}"
        print(f"{numThreads:<8} | {tpsStr:<20} | {p99Pooled:<12.1f} | {p99IqrStr:<15} | {status}")
    
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
    
    # Save aggregated results with CI.
    with open("results/quadcore_mixed_workload_with_ci.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threads", "tps_mean", "tps_margin", "tps_cv",
                        "p99_pooled", "p99_median", "p99_iqr", "n_runs"])
        for t in sorted(aggregated.keys()):
            a = aggregated[t]
            writer.writerow([a.threads, f"{a.tpsMean:.0f}", f"{a.tpsMargin:.0f}",
                           f"{a.tpsCV:.1f}", f"{a.p99Pooled:.1f}",
                           f"{a.p99Median:.1f}", f"{a.p99IQR:.1f}", a.nRuns])
    
    with open("results/quadcore_io_baseline.csv", "w", newline="") as f:
        fieldnames = ["threads", "tps", "avgLat"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ioResults)
    
    # Print summary.
    print()
    print("-" * 70)
    print("RESULTS SUMMARY (with 95% CI)")
    print("-" * 70)
    
    peakResult = aggregated[peakThreads]
    finalResult = aggregated[max(aggregated.keys())]
    cliffSeverity = ((peakTps - finalResult.tpsMean) / peakTps) * 100
    
    print()
    print("Peak Performance:")
    print(f"  {peakTps:,.0f} ± {peakResult.tpsMargin:,.0f} TPS at {peakThreads} threads.")
    print()
    print("Cliff Analysis:")
    print(f"  Final: {finalResult.tpsMean:,.0f} ± {finalResult.tpsMargin:,.0f} TPS at {THREAD_COUNTS[-1]} threads.")
    print(f"  Severity: {cliffSeverity:.1f}% throughput loss.")
    print()
    print("P99 Latency Analysis (pooled across runs):")
    print(f"  At peak ({peakThreads} threads): {peakResult.p99Pooled:.1f} ms")
    print(f"  At final ({THREAD_COUNTS[-1]} threads): {finalResult.p99Pooled:.1f} ms")
    print(f"  Latency increase: {finalResult.p99Pooled / peakResult.p99Pooled:.1f}x")
    print()
    print("Results saved to results/ directory.")
    
    # Generate LaTeX table snippet.
    print()
    print("LaTeX Table Snippet (for Table 4 - Quad Core):")
    print("-" * 60)
    for t in [1, 32, 64, 256, 2048]:
        if t in aggregated:
            a = aggregated[t]
            print(f"{t} & {a.tpsMean:,.0f} $\\pm$ {a.tpsMargin:,.0f} & {a.p99Pooled:.1f} \\\\")
    
    if cliffSeverity >= 15:
        print()
        print("The OS-GIL Paradox is confirmed: cliff persists on multi-core hardware.")
    
    return 0 if cliffSeverity >= 10 else 1


if __name__ == "__main__":
    sys.exit(main())
