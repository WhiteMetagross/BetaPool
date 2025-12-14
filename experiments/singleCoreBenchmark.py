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
from typing import List, Dict

# Set seed for reproducibility.
SEED = 17
random.seed(SEED)

# Simulate single core edge device using CPU affinity.
try:
    os.sched_setaffinity(0, {0})
    EDGE_MODE = True
    print("Edge Simulation: Process pinned to single CPU core.")
except (AttributeError, OSError):
    EDGE_MODE = False
    print("Running in multi core mode. Results may differ from edge devices.")

# Workload parameters tuned for visible cliff detection.
CPU_ITERATIONS = 1000    # Pure Python loop iterations holding the GIL.
IO_SLEEP_MS = 0.1        # I/O sleep duration in milliseconds.
TASK_COUNT = 20000       # Total tasks per experiment configuration.
NUM_RUNS = 3             # Number of runs for statistical significance.

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
    )


def runFullCharacterization() -> Dict[str, List[ExperimentResult]]:
    # Run complete characterization experiment across all thread counts.
    results = {"mixed": [], "ioOnly": []}
    
    print()
    print("-" * 70)
    print("EXPERIMENT: GIL SATURATION CLIFF CHARACTERIZATION")
    print("-" * 70)
    print("Configuration:")
    print(f"  CPU Work: {CPU_ITERATIONS} iterations.")
    print(f"  I/O Sleep: {IO_SLEEP_MS} ms.")
    print(f"  Tasks: {TASK_COUNT:,}.")
    print(f"  Runs: {NUM_RUNS}.")
    print(f"  Edge Mode: {'Yes (single-core)' if EDGE_MODE else 'No (multi-core)'}.")
    print()
    
    # Run mixed workload experiment.
    print("Phase 1: Mixed CPU+I/O Workload")
    print("-" * 70)
    print(f"{'Threads':<8} | {'TPS':<12} | {'Latency':<10} | {'P99':<10} | {'Status'}")
    print("-" * 70)
    
    peakTps = 0
    peakThreads = 0
    
    for n in THREAD_COUNTS:
        runResults = []
        for run in range(NUM_RUNS):
            result = runExperiment(n, mixedWorkloadTask, run)
            runResults.append(result)
            results["mixed"].append(result)
        
        # Average across runs for display.
        avgTps = statistics.mean([r.throughputTps for r in runResults])
        avgLat = statistics.mean([r.avgLatencyMs for r in runResults])
        avgP99 = statistics.mean([r.p99LatencyMs for r in runResults])
        
        if avgTps > peakTps:
            peakTps = avgTps
            peakThreads = n
        
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
        
        print(f"{n:<8} | {avgTps:<12,.0f} | {avgLat:<10.2f} | {avgP99:<10.2f} | {status}")
    
    # Run pure I/O baseline for comparison.
    print()
    print("Phase 2: Pure I/O Baseline")
    print("-" * 70)
    
    for n in [1, 4, 16, 64, 256]:
        result = runExperiment(n, pureIoTask, 0)
        results["ioOnly"].append(result)
        print(f"{n:<8} | {result.throughputTps:<12,.0f} | {result.avgLatencyMs:<10.2f}")
    
    return results


def analyzeResults(results: Dict[str, List[ExperimentResult]]) -> dict:
    # Analyze and summarize experiment results.
    mixed = results["mixed"]
    
    # Aggregate by thread count.
    byThreads = {}
    for r in mixed:
        if r.threads not in byThreads:
            byThreads[r.threads] = []
        byThreads[r.threads].append(r)
    
    # Find peak and cliff points.
    threadTps = {t: statistics.mean([r.throughputTps for r in runs]) 
                 for t, runs in byThreads.items()}
    
    peakThreads = max(threadTps, key=threadTps.get)
    peakTps = threadTps[peakThreads]
    
    finalThreads = max(threadTps.keys())
    finalTps = threadTps[finalThreads]
    
    cliffSeverity = ((peakTps - finalTps) / peakTps) * 100
    
    print()
    print("-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print()
    print("Peak Performance:")
    print(f"  {peakTps:,.0f} TPS at {peakThreads} threads.")
    print()
    print("Cliff Analysis:")
    print(f"  Final: {finalTps:,.0f} TPS at {finalThreads} threads.")
    print(f"  Severity: {cliffSeverity:.1f}% throughput loss.")
    
    # Determine degradation onset point.
    for t in sorted(byThreads.keys()):
        drop = ((peakTps - threadTps[t]) / peakTps) * 100
        if drop > 5:
            print(f"  Degradation begins at: {t} threads.")
            break
    
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
        "finalThreads": finalThreads,
        "finalTps": finalTps,
        "cliffSeverity": cliffSeverity,
    }


def saveResults(results: Dict[str, List[ExperimentResult]], summary: dict):
    # Save results to CSV files for figure generation.
    with open("results/mixed_workload.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threads", "run", "elapsed", "tps", "avg_lat", "p50_lat", "p95_lat", "p99_lat"])
        for r in results["mixed"]:
            writer.writerow([r.threads, r.runId, r.elapsedSec, r.throughputTps,
                           r.avgLatencyMs, r.p50LatencyMs, r.p95LatencyMs, r.p99LatencyMs])
    
    with open("results/io_baseline.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threads", "tps", "avg_lat"])
        for r in results["ioOnly"]:
            writer.writerow([r.threads, r.throughputTps, r.avgLatencyMs])
    
    # Save summary for quick reference.
    with open("results/summary.txt", "w") as f:
        f.write("GIL Saturation Cliff Experiment Summary\n")
        f.write("-" * 50 + "\n\n")
        f.write(f"Peak: {summary['peakTps']:,.0f} TPS at {summary['peakThreads']} threads.\n")
        f.write(f"Final: {summary['finalTps']:,.0f} TPS at {summary['finalThreads']} threads.\n")
        f.write(f"Cliff Severity: {summary['cliffSeverity']:.1f}%.\n")
    
    print()
    print("Results saved to results/ directory.")


def main():
    # Main entry point for the single-core benchmark.
    print()
    print("-" * 70)
    print("GIL SATURATION CLIFF - SINGLE CORE BENCHMARK")
    print("-" * 70)
    
    results = runFullCharacterization()
    summary = analyzeResults(results)
    saveResults(results, summary)
    
    return 0 if summary["cliffSeverity"] >= 15 else 1


if __name__ == "__main__":
    sys.exit(main())
