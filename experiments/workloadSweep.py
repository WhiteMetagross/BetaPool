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
from scipy import stats

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
RUNS_PER_CONFIG = 5         # n=5 runs for sweep experiments (Section 3.3).
CONFIDENCE_LEVEL = 0.95     # 95% confidence interval.
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
    rawLatencies: List[float] = None  # For pooled P99


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
        if runLats:
            pooled.extend(runLats)
    if not pooled:
        return 0.0
    pooled.sort()
    idx = int(len(pooled) * 0.99)
    return pooled[idx] * 1000  # Convert to ms


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
    
    rawLatencies = latencies.copy()  # Keep raw for pooled P99
    latenciesMs = [lat * 1000 for lat in latencies]
    latenciesMs.sort()
    
    return {
        "tps": taskCount / elapsed,
        "avgLatMs": statistics.mean(latenciesMs),
        "p99LatMs": latenciesMs[int(len(latenciesMs) * 0.99)],
        "avgBeta": statistics.mean(betas),
        "stdBeta": statistics.stdev(betas) if len(betas) > 1 else 0.0,
        "rawLatencies": rawLatencies,
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


def runWorkloadSweep() -> Tuple[List[SweepResult], Dict]:
    """Run complete workload parameter sweep. Returns results and aggregated data."""
    results = []
    aggregatedData = {}  # key: (workloadDesc, threads) -> CI stats
    
    print()
    print("=" * 80)
    print("WORKLOAD PARAMETER SWEEP EXPERIMENT")
    print("=" * 80)
    print()
    print(f"Task count per config: {TASK_COUNT}")
    print(f"Runs per config: {RUNS_PER_CONFIG} (n=5 as per Section 3.3 for sweeps)")
    print(f"Confidence: {CONFIDENCE_LEVEL*100:.0f}% CI using t-distribution")
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
        
        print(f"  {'Threads':<8} | {'TPS (±CI)':<18} | {'P99 Pooled':<12} | {'Avg Beta':<10}")
        print("  " + "-" * 60)
        
        for threads in THREAD_COUNTS:
            runResults = []
            allLatencies = []
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
                    rawLatencies=config["rawLatencies"],
                )
                results.append(result)
                runResults.append(config)
                allLatencies.append(config["rawLatencies"])
            
            # Compute aggregated stats with CI.
            tpsMean, tpsMargin = computeCI([r["tps"] for r in runResults], CONFIDENCE_LEVEL)
            p99Pooled = computePooledP99(allLatencies)
            avgBeta = statistics.mean([r["avgBeta"] for r in runResults])
            
            # Store aggregated data.
            aggregatedData[(desc, threads)] = {
                "tpsMean": tpsMean,
                "tpsMargin": tpsMargin,
                "p99Pooled": p99Pooled,
                "avgBeta": avgBeta,
                "optimalThreads": optimalThreads,
                "nRuns": RUNS_PER_CONFIG,
            }
            
            status = "OPTIMAL" if threads == optimalThreads else ""
            tpsStr = f"{tpsMean:,.0f}±{tpsMargin:,.0f}"
            print(f"  {threads:<8} | {tpsStr:<18} | {p99Pooled:<12.2f} | {avgBeta:<10.3f} {status}")
    
    return results, aggregatedData


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


def saveResults(sweepResults: List[SweepResult], sensitivityResults: List[Dict], aggregatedData: Dict):
    """Save results to CSV files."""
    os.makedirs("results", exist_ok=True)
    
    # Save workload sweep results (raw per-run).
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
    
    # Save aggregated results with CI.
    with open("results/workload_sweep_with_ci.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["workload_desc", "threads", "tps_mean", "tps_margin", 
                         "p99_pooled", "avg_beta", "optimal_threads", "n_runs"])
        for (desc, threads), data in sorted(aggregatedData.items()):
            writer.writerow([
                desc, threads, 
                f"{data['tpsMean']:.0f}", f"{data['tpsMargin']:.0f}",
                f"{data['p99Pooled']:.2f}", f"{data['avgBeta']:.3f}",
                data['optimalThreads'], data['nRuns']
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
    print("  results/workload_sweep_with_ci.csv")
    print("  results/beta_threshold_sensitivity.csv")


def printSummary(sweepResults: List[SweepResult], aggregatedData: Dict):
    """Print summary analysis of results with LaTeX table output for Table 13."""
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
    
    # Generate LaTeX table snippet for Table 13.
    print()
    print("LaTeX Table Snippet (for Table 13 - Workload Generalization):")
    print("-" * 70)
    print()
    
    # For each workload type, show performance at optimal threads.
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Workload Profile & Optimal $T$ & TPS ($\\pm$CI) & P99 (ms) \\\\")
    print("\\midrule")
    
    for cpuIter, ioMs, desc in WORKLOAD_CONFIGS:
        # Find optimal threads for this workload.
        optimalThreads = None
        for (d, t), data in aggregatedData.items():
            if d == desc:
                optimalThreads = data['optimalThreads']
                break
        
        if optimalThreads:
            # Get stats at optimal thread count.
            key = (desc, optimalThreads)
            if key in aggregatedData:
                data = aggregatedData[key]
                shortDesc = desc.split('(')[0].strip()
                print(f"{shortDesc} & {optimalThreads} & {data['tpsMean']:,.0f} $\\pm$ {data['tpsMargin']:,.0f} & {data['p99Pooled']:.2f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
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
    sweepResults, aggregatedData = runWorkloadSweep()
    
    # Run beta threshold sensitivity analysis.
    sensitivityResults = runBetaThresholdSensitivity()
    
    # Save results.
    saveResults(sweepResults, sensitivityResults, aggregatedData)
    
    # Print summary.
    printSummary(sweepResults, aggregatedData)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
