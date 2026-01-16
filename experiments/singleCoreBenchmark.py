#!/usr/bin/env python3
# GIL Saturation Cliff Single Core Benchmark.
# This is the authoritative experiment for characterizing the saturation cliff.
# Simulates a single core edge device using CPU affinity constraints.

import time
import random
import concurrent.futures
import statistics
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy import stats
import psutil

# Set seed for reproducibility.
SEED = 17
random.seed(SEED)

# Simulate single core edge device using CPU affinity.
# Use psutil for cross-platform support (Windows + Linux).
EDGE_MODE = False
try:
    # Try Linux-native method first.
    os.sched_setaffinity(0, {0})
    EDGE_MODE = True
    print("Edge Simulation: Process pinned to single CPU core (Linux).")
except (AttributeError, OSError):
    # Fall back to psutil for Windows.
    try:
        p = psutil.Process()
        p.cpu_affinity([0])
        EDGE_MODE = True
        print("Edge Simulation: Process pinned to single CPU core (psutil).")
    except Exception as e:
        print(f"Warning: Could not set CPU affinity: {e}")
        print("Running in multi core mode. Results may differ from edge devices.")

# Workload parameters tuned for visible cliff detection.
CPU_ITERATIONS = 1000    # Pure Python loop iterations holding the GIL.
IO_SLEEP_MS = 0.1        # I/O sleep duration in milliseconds.
TASK_COUNT = 20000       # Total tasks per experiment configuration.
NUM_RUNS = 10            # Number of runs for statistical significance (Section 3.3).
CONFIDENCE_LEVEL = 0.95  # 95% confidence interval.

# Thread counts to evaluate across the full range.
THREAD_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Create output directory for results.
os.makedirs("results", exist_ok=True)


@dataclass
class ExperimentResult:
    # Container for single experiment run results.
    threads: int
    runId: int
    elapsedSec: float
    throughputTps: float
    avgLatencyMs: float
    p50LatencyMs: float
    p95LatencyMs: float
    p99LatencyMs: float
    latencies: List[float] = None  # Raw latencies for pooled P99


@dataclass
class AggregatedResult:
    # Container for aggregated results with confidence intervals.
    threads: int
    tpsMean: float
    tpsMargin: float  # 95% CI margin
    tpsCV: float  # Coefficient of variation
    p99Pooled: float  # Pooled P99 across all runs
    p99Median: float  # Median of per-run P99
    p99IQR: float  # IQR of per-run P99
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
        perRunP99.append(sortedLats[idx] * 1000)  # Convert to ms
    perRunP99.sort()
    n = len(perRunP99)
    median = statistics.median(perRunP99)
    q1 = perRunP99[n // 4]
    q3 = perRunP99[3 * n // 4]
    iqr = q3 - q1
    return (median, iqr)


def mixedWorkloadTask() -> float:
    # Simulates an edge AI inference task with mixed CPU and I/O phases.
    # CPU phase holds the GIL while I/O phase releases it.
    start = time.perf_counter()
    
    # CPU phase representing model inference that holds the GIL.
    x = 0
    for _ in range(CPU_ITERATIONS):
        x += 1
    
    # I/O phase representing sensor read or network call that releases the GIL.
    time.sleep(IO_SLEEP_MS / 1000)
    
    return time.perf_counter() - start


def pureIoTask() -> float:
    # Pure I/O task for baseline comparison without GIL contention.
    start = time.perf_counter()
    time.sleep((IO_SLEEP_MS + 0.5) / 1000)
    return time.perf_counter() - start


def runExperiment(nThreads: int, taskFn, runId: int = 0) -> ExperimentResult:
    # Execute experiment with specified thread count and return metrics.
    latencies = []
    
    startTime = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=nThreads) as pool:
        futures = [pool.submit(taskFn) for _ in range(TASK_COUNT)]
        for future in concurrent.futures.as_completed(futures):
            latencies.append(future.result())
    
    elapsed = time.perf_counter() - startTime
    rawLatencies = latencies.copy()  # Keep raw for pooled P99
    latencies.sort()
    
    # Convert latencies to milliseconds.
    latenciesMs = [lat * 1000 for lat in latencies]
    
    return ExperimentResult(
        threads=nThreads,
        runId=runId,
        elapsedSec=elapsed,
        throughputTps=TASK_COUNT / elapsed,
        avgLatencyMs=statistics.mean(latenciesMs),
        p50LatencyMs=latenciesMs[len(latenciesMs) // 2],
        p95LatencyMs=latenciesMs[int(len(latenciesMs) * 0.95)],
        p99LatencyMs=latenciesMs[int(len(latenciesMs) * 0.99)],
        latencies=rawLatencies,
    )


def runFullCharacterization() -> Tuple[Dict[str, List[ExperimentResult]], Dict[int, AggregatedResult]]:
    # Run complete characterization experiment across all thread counts.
    results = {"mixed": [], "ioOnly": []}
    aggregated = {}  # Thread count -> AggregatedResult
    
    print()
    print("-" * 70)
    print("EXPERIMENT: GIL SATURATION CLIFF CHARACTERIZATION")
    print("-" * 70)
    print("Configuration:")
    print(f"  CPU Work: {CPU_ITERATIONS} iterations.")
    print(f"  I/O Sleep: {IO_SLEEP_MS} ms.")
    print(f"  Tasks: {TASK_COUNT:,}.")
    print(f"  Runs: {NUM_RUNS} (n=10 as per Section 3.3).")
    print(f"  Confidence: {CONFIDENCE_LEVEL*100:.0f}% CI using t-distribution.")
    print(f"  Edge Mode: {'Yes (single-core)' if EDGE_MODE else 'No (multi-core)'}.")
    print()
    
    # Run mixed workload experiment.
    print("Phase 1: Mixed CPU+I/O Workload")
    print("-" * 70)
    print(f"{'Threads':<8} | {'TPS (mean±CI)':<20} | {'P99 Pooled':<12} | {'P99 Med±IQR':<15} | {'Status'}")
    print("-" * 70)
    
    peakTps = 0
    peakThreads = 0
    
    for n in THREAD_COUNTS:
        runResults = []
        allLatencies = []
        for run in range(NUM_RUNS):
            result = runExperiment(n, mixedWorkloadTask, run)
            runResults.append(result)
            results["mixed"].append(result)
            allLatencies.append(result.latencies)
        
        # Compute statistics as per Section 3.3.
        tpsList = [r.throughputTps for r in runResults]
        tpsMean, tpsMargin = computeCI(tpsList, CONFIDENCE_LEVEL)
        tpsCV = (statistics.stdev(tpsList) / tpsMean * 100) if tpsMean > 0 else 0
        
        p99Pooled = computePooledP99(allLatencies)
        p99Median, p99IQR = computeP99Stats(allLatencies)
        
        aggregated[n] = AggregatedResult(
            threads=n,
            tpsMean=tpsMean,
            tpsMargin=tpsMargin,
            tpsCV=tpsCV,
            p99Pooled=p99Pooled,
            p99Median=p99Median,
            p99IQR=p99IQR,
            nRuns=NUM_RUNS,
        )
        
        if tpsMean > peakTps:
            peakTps = tpsMean
            peakThreads = n
        
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
        print(f"{n:<8} | {tpsStr:<20} | {p99Pooled:<12.1f} | {p99IqrStr:<15} | {status}")
    
    # Run pure I/O baseline for comparison.
    print()
    print("Phase 2: Pure I/O Baseline")
    print("-" * 70)
    
    for n in [1, 4, 16, 64, 256]:
        result = runExperiment(n, pureIoTask, 0)
        results["ioOnly"].append(result)
        print(f"{n:<8} | {result.throughputTps:<12,.0f} | {result.avgLatencyMs:<10.2f}")
    
    return results, aggregated


def analyzeResults(results: Dict[str, List[ExperimentResult]], aggregated: Dict[int, AggregatedResult]) -> dict:
    # Analyze and summarize experiment results.
    
    # Find peak and cliff points from aggregated data.
    peakResult = max(aggregated.values(), key=lambda x: x.tpsMean)
    peakThreads = peakResult.threads
    peakTps = peakResult.tpsMean
    peakMargin = peakResult.tpsMargin
    
    finalThreads = max(aggregated.keys())
    finalResult = aggregated[finalThreads]
    finalTps = finalResult.tpsMean
    finalMargin = finalResult.tpsMargin
    
    cliffSeverity = ((peakTps - finalTps) / peakTps) * 100
    
    print()
    print("-" * 70)
    print("RESULTS SUMMARY (with 95% CI)")
    print("-" * 70)
    print()
    print("Peak Performance:")
    print(f"  {peakTps:,.0f} ± {peakMargin:,.0f} TPS at {peakThreads} threads.")
    print()
    print("Cliff Analysis:")
    print(f"  Final: {finalTps:,.0f} ± {finalMargin:,.0f} TPS at {finalThreads} threads.")
    print(f"  Severity: {cliffSeverity:.1f}% throughput loss.")
    
    # Determine degradation onset point.
    for t in sorted(aggregated.keys()):
        drop = ((peakTps - aggregated[t].tpsMean) / peakTps) * 100
        if drop > 5:
            print(f"  Degradation begins at: {t} threads.")
            break
    
    print()
    print("P99 Latency Analysis (pooled across runs):")
    print(f"  At peak ({peakThreads} threads): {peakResult.p99Pooled:.1f} ms")
    print(f"  At final ({finalThreads} threads): {finalResult.p99Pooled:.1f} ms")
    print(f"  Latency increase: {finalResult.p99Pooled / peakResult.p99Pooled:.1f}x")
    
    print()
    if cliffSeverity >= 30:
        print("Strong cliff detected. This is a publishable result.")
    elif cliffSeverity >= 15:
        print("Moderate cliff detected. Evidence is sufficient for the paper.")
    else:
        print("Weak cliff detected. Consider stronger hardware constraints.")
    
    return {
        "peakThreads": peakThreads,
        "peakTps": peakTps,
        "peakMargin": peakMargin,
        "finalThreads": finalThreads,
        "finalTps": finalTps,
        "finalMargin": finalMargin,
        "cliffSeverity": cliffSeverity,
        "peakP99": peakResult.p99Pooled,
        "finalP99": finalResult.p99Pooled,
    }


def saveResults(results: Dict[str, List[ExperimentResult]], aggregated: Dict[int, AggregatedResult], summary: dict):
    # Save results to CSV files for figure generation.
    
    # Save raw per-run results.
    with open("results/mixed_workload.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threads", "run", "elapsed", "tps", "avg_lat", "p50_lat", "p95_lat", "p99_lat"])
        for r in results["mixed"]:
            writer.writerow([r.threads, r.runId, r.elapsedSec, r.throughputTps,
                           r.avgLatencyMs, r.p50LatencyMs, r.p95LatencyMs, r.p99LatencyMs])
    
    # Save aggregated results with CI (for Table 4).
    with open("results/mixed_workload_with_ci.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threads", "tps_mean", "tps_margin", "tps_cv", 
                        "p99_pooled", "p99_median", "p99_iqr", "n_runs"])
        for t in sorted(aggregated.keys()):
            a = aggregated[t]
            writer.writerow([a.threads, f"{a.tpsMean:.0f}", f"{a.tpsMargin:.0f}", 
                           f"{a.tpsCV:.1f}", f"{a.p99Pooled:.1f}", 
                           f"{a.p99Median:.1f}", f"{a.p99IQR:.1f}", a.nRuns])
    
    with open("results/io_baseline.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threads", "tps", "avg_lat"])
        for r in results["ioOnly"]:
            writer.writerow([r.threads, r.throughputTps, r.avgLatencyMs])
    
    # Save summary for quick reference.
    with open("results/summary.txt", "w") as f:
        f.write("GIL Saturation Cliff Experiment Summary (Single-Core)\n")
        f.write(f"Statistical Methodology: n={NUM_RUNS} runs, {CONFIDENCE_LEVEL*100:.0f}% CI\n")
        f.write("-" * 50 + "\n\n")
        f.write(f"Peak: {summary['peakTps']:,.0f} ± {summary['peakMargin']:,.0f} TPS at {summary['peakThreads']} threads.\n")
        f.write(f"Final: {summary['finalTps']:,.0f} ± {summary['finalMargin']:,.0f} TPS at {summary['finalThreads']} threads.\n")
        f.write(f"Cliff Severity: {summary['cliffSeverity']:.1f}%.\n")
        f.write(f"P99 at peak: {summary['peakP99']:.1f} ms\n")
        f.write(f"P99 at final: {summary['finalP99']:.1f} ms\n")
        f.write(f"Latency increase: {summary['finalP99']/summary['peakP99']:.1f}x\n")
    
    # Generate LaTeX table snippet.
    print()
    print("LaTeX Table Snippet (for Table 4 - Single Core):")
    print("-" * 60)
    for t in [1, 32, 64, 256, 2048]:
        if t in aggregated:
            a = aggregated[t]
            print(f"{t} & {a.tpsMean:,.0f} $\\pm$ {a.tpsMargin:,.0f} & {a.p99Pooled:.1f} \\\\")
    
    print()
    print("Results saved to results/ directory.")


def main():
    # Main entry point for the single-core benchmark.
    print()
    print("-" * 70)
    print("GIL SATURATION CLIFF - SINGLE CORE BENCHMARK")
    print(f"Statistical Rigor: n={NUM_RUNS} runs, {CONFIDENCE_LEVEL*100:.0f}% CI")
    print("-" * 70)
    
    results, aggregated = runFullCharacterization()
    summary = analyzeResults(results, aggregated)
    saveResults(results, aggregated, summary)
    
    return 0 if summary["cliffSeverity"] >= 15 else 1


if __name__ == "__main__":
    sys.exit(main())
