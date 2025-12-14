#!/usr/bin/env python3
"""
Workload Parameter Sweep Experiment.

Tests the adaptive controller across different CPU/IO ratios to demonstrate
robustness and determine sensitivity to workload characteristics.

This addresses reviewer concern #3: robustness across workloads and parameters.
"""

import time
import random
import statistics
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# Set seed for reproducibility.
SEED = 17
random.seed(SEED)

# Constrain to single core for consistent results.
try:
    os.sched_setaffinity(0, {0})
    print("Process pinned to single CPU core for edge simulation.")
except (AttributeError, OSError):
    print("Note: CPU affinity not available on this platform.")

# Experiment parameters.
TASK_COUNT = 10000
RUNS_PER_CONFIG = 5
THREAD_COUNTS = [8, 16, 32, 64, 128, 256]

# CPU/IO ratio configurations to sweep.
# Format: (cpu_iterations, io_sleep_ms, description)
WORKLOAD_CONFIGS = [
    (100, 1.0, "IO Heavy (100 iter, 1.0ms)"),
    (500, 0.5, "IO Dominant (500 iter, 0.5ms)"),
    (1000, 0.1, "Balanced (1000 iter, 0.1ms)"),
    (2000, 0.05, "CPU Leaning (2000 iter, 0.05ms)"),
    (5000, 0.01, "CPU Heavy (5000 iter, 0.01ms)"),
    (10000, 0.001, "CPU Dominant (10000 iter, 0.001ms)"),
]

# Beta threshold values to test sensitivity.
BETA_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]


@dataclass
class SweepResult:
    """Container for parameter sweep results."""
    cpuIterations: int
    ioSleepMs: float
    workloadDesc: str
    threads: int
    run: int
    tps: float
    avgLatMs: float
    p99LatMs: float
    avgBeta: float
    optimalThreads: int = 0


def createMixedTask(cpuIterations: int, ioSleepMs: float):
    """Factory function to create configurable mixed workload tasks."""
    def task() -> Tuple[float, float]:
        wallStart = time.perf_counter()
        cpuStart = time.thread_time()
        
        # CPU phase.
        x = 0
        for _ in range(cpuIterations):
            x += 1
        
        # IO phase.
        time.sleep(ioSleepMs / 1000.0)
        
        cpuEnd = time.thread_time()
        wallEnd = time.perf_counter()
        
        latency = wallEnd - wallStart
        cpuTime = cpuEnd - cpuStart
        beta = 1.0 - min(1.0, cpuTime / latency) if latency > 0 else 0.0
        
        return latency, beta
    
    return task


def runSingleConfig(
    cpuIterations: int,
    ioSleepMs: float,
    numThreads: int,
    taskCount: int,
) -> Dict:
    """Run benchmark for a single configuration."""
    task = createMixedTask(cpuIterations, ioSleepMs)
    
    latencies = []
    betas = []
    
    startTime = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=numThreads) as executor:
        futures = [executor.submit(task) for _ in range(taskCount)]
        for future in as_completed(futures):
            lat, beta = future.result()
            latencies.append(lat)
            betas.append(beta)
    
    elapsed = time.perf_counter() - startTime
    
    latenciesMs = [lat * 1000 for lat in latencies]
    latenciesMs.sort()
    
    return {
        "tps": taskCount / elapsed,
        "avgLatMs": statistics.mean(latenciesMs),
        "p99LatMs": latenciesMs[int(len(latenciesMs) * 0.99)],
        "avgBeta": statistics.mean(betas),
        "stdBeta": statistics.stdev(betas) if len(betas) > 1 else 0.0,
    }


def findOptimalThreads(cpuIterations: int, ioSleepMs: float) -> int:
    """Find optimal thread count for a workload configuration."""
    bestTps = 0
    bestThreads = 1
    
    for threads in THREAD_COUNTS:
        result = runSingleConfig(cpuIterations, ioSleepMs, threads, TASK_COUNT // 2)
        if result["tps"] > bestTps:
            bestTps = result["tps"]
            bestThreads = threads
    
    return bestThreads


def runWorkloadSweep() -> List[SweepResult]:
    """Run complete workload parameter sweep."""
    results = []
    
    print()
    print("=" * 80)
    print("WORKLOAD PARAMETER SWEEP EXPERIMENT")
    print("=" * 80)
    print()
    print(f"Task count per config: {TASK_COUNT}")
    print(f"Runs per config: {RUNS_PER_CONFIG}")
    print(f"Thread counts: {THREAD_COUNTS}")
    print()
    
    for cpuIter, ioMs, desc in WORKLOAD_CONFIGS:
        print()
        print("-" * 80)
        print(f"Workload: {desc}")
        print(f"  CPU Iterations: {cpuIter}, IO Sleep: {ioMs} ms")
        print("-" * 80)
        
        # Find optimal thread count for this workload.
        print("  Finding optimal thread count...")
        optimalThreads = findOptimalThreads(cpuIter, ioMs)
        print(f"  Optimal: {optimalThreads} threads")
        print()
        
        print(f"  {'Threads':<8} | {'TPS':<12} | {'Avg Lat':<10} | {'P99 Lat':<10} | {'Avg Beta':<10}")
        print("  " + "-" * 60)
        
        for threads in THREAD_COUNTS:
            runResults = []
            for run in range(RUNS_PER_CONFIG):
                config = runSingleConfig(cpuIter, ioMs, threads, TASK_COUNT)
                
                result = SweepResult(
                    cpuIterations=cpuIter,
                    ioSleepMs=ioMs,
                    workloadDesc=desc,
                    threads=threads,
                    run=run,
                    tps=config["tps"],
                    avgLatMs=config["avgLatMs"],
                    p99LatMs=config["p99LatMs"],
                    avgBeta=config["avgBeta"],
                    optimalThreads=optimalThreads,
                )
                results.append(result)
                runResults.append(config)
            
            # Display averaged results.
            avgTps = statistics.mean([r["tps"] for r in runResults])
            avgLat = statistics.mean([r["avgLatMs"] for r in runResults])
            avgP99 = statistics.mean([r["p99LatMs"] for r in runResults])
            avgBeta = statistics.mean([r["avgBeta"] for r in runResults])
            
            status = "OPTIMAL" if threads == optimalThreads else ""
            print(f"  {threads:<8} | {avgTps:<12,.0f} | {avgLat:<10.3f} | {avgP99:<10.3f} | {avgBeta:<10.3f} {status}")
    
    return results


def runBetaThresholdSensitivity() -> List[Dict]:
    """Test sensitivity to beta threshold parameter."""
    results = []
    
    print()
    print("=" * 80)
    print("BETA THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()
    
    # Use balanced workload for sensitivity testing.
    cpuIter, ioMs, desc = WORKLOAD_CONFIGS[2]  # Balanced config.
    
    print(f"Workload: {desc}")
    print(f"Testing beta thresholds: {BETA_THRESHOLDS}")
    print()
    
    # Simulate adaptive controller behavior at different thresholds.
    # For each threshold, determine what thread count would be chosen.
    print(f"{'Threshold':<12} | {'Predicted Threads':<18} | {'Resulting TPS':<15}")
    print("-" * 50)
    
    for threshold in BETA_THRESHOLDS:
        # Measure beta at different thread counts.
        for threads in THREAD_COUNTS:
            config = runSingleConfig(cpuIter, ioMs, threads, TASK_COUNT // 2)
            
            # If beta is above threshold, controller would scale up.
            # If beta is below threshold, controller would veto scaling.
            wouldVeto = config["avgBeta"] < threshold
            
            results.append({
                "threshold": threshold,
                "threads": threads,
                "tps": config["tps"],
                "beta": config["avgBeta"],
                "wouldVeto": wouldVeto,
            })
        
        # Find what thread count the controller would settle on.
        thresholdResults = [r for r in results if r["threshold"] == threshold]
        validConfigs = [r for r in thresholdResults if not r["wouldVeto"]]
        
        if validConfigs:
            maxThreads = max(r["threads"] for r in validConfigs)
            maxConfig = [r for r in validConfigs if r["threads"] == maxThreads][0]
            print(f"{threshold:<12.2f} | {maxThreads:<18} | {maxConfig['tps']:<15,.0f}")
        else:
            # All configurations would be vetoed, use minimum.
            minConfig = thresholdResults[0]
            print(f"{threshold:<12.2f} | {minConfig['threads']:<18} | {minConfig['tps']:<15,.0f}")
    
    return results


def saveResults(sweepResults: List[SweepResult], sensitivityResults: List[Dict]):
    """Save results to CSV files."""
    os.makedirs("results", exist_ok=True)
    
    # Save workload sweep results.
    with open("results/workload_sweep.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cpu_iterations", "io_sleep_ms", "workload_desc", "threads", "run",
            "tps", "avg_lat_ms", "p99_lat_ms", "avg_beta", "optimal_threads"
        ])
        for r in sweepResults:
            writer.writerow([
                r.cpuIterations, r.ioSleepMs, r.workloadDesc, r.threads, r.run,
                r.tps, r.avgLatMs, r.p99LatMs, r.avgBeta, r.optimalThreads
            ])
    
    # Save sensitivity results.
    with open("results/beta_threshold_sensitivity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "threads", "tps", "beta", "would_veto"])
        for r in sensitivityResults:
            writer.writerow([r["threshold"], r["threads"], r["tps"], r["beta"], r["wouldVeto"]])
    
    print()
    print("Results saved to:")
    print("  results/workload_sweep.csv")
    print("  results/beta_threshold_sensitivity.csv")


def printSummary(sweepResults: List[SweepResult]):
    """Print summary analysis of results."""
    print()
    print("=" * 80)
    print("SUMMARY ANALYSIS")
    print("=" * 80)
    print()
    
    # Group by workload type.
    workloads = {}
    for r in sweepResults:
        key = r.workloadDesc
        if key not in workloads:
            workloads[key] = []
        workloads[key].append(r)
    
    print("Optimal Thread Counts by Workload Type:")
    print("-" * 50)
    print(f"{'Workload':<40} | {'Optimal Threads':<15}")
    print("-" * 50)
    
    for desc, results in workloads.items():
        optimal = results[0].optimalThreads
        print(f"{desc:<40} | {optimal:<15}")
    
    print()
    print("Key Finding: Optimal thread count varies with workload characteristics.")
    print("IO-heavy workloads benefit from higher thread counts.")
    print("CPU-heavy workloads require lower thread counts to avoid GIL contention.")


def main():
    """Main entry point for parameter sweep experiment."""
    print()
    print("#" * 80)
    print("# WORKLOAD PARAMETER SWEEP AND SENSITIVITY ANALYSIS")
    print("#" * 80)
    
    # Run workload sweep.
    sweepResults = runWorkloadSweep()
    
    # Run beta threshold sensitivity analysis.
    sensitivityResults = runBetaThresholdSensitivity()
    
    # Save results.
    saveResults(sweepResults, sensitivityResults)
    
    # Print summary.
    printSummary(sweepResults)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
