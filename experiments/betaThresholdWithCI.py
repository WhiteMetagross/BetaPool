#!/usr/bin/env python3
"""
Beta Threshold Sensitivity Experiment with Statistical Confidence Intervals.

This experiment generates data for Table XII of the paper:
- β_thresh Sensitivity Analysis (I/O Dominant Workload)
- Tests stability of performance across different threshold values

Uses the statistical methodology from Section 3.3:
- n=10 runs per configuration
- 95% CI using t-distribution
"""

import sys
import time
import random
import statistics
import csv
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Tuple
import psutil

# Attempt to import scipy for t-distribution
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Using approximate CI calculation.")

# Set seed for reproducibility.
SEED = 17
random.seed(SEED)

# I/O Dominant workload parameters (minimal CPU, high I/O)
CPU_ITERATIONS = 100      # Minimal CPU work
IO_SLEEP_MS = 1.0         # High I/O wait
NUM_TASKS = 10000         # Tasks per benchmark run
NUM_RUNS = 10             # n=10 runs for statistical significance
CONFIDENCE_LEVEL = 0.95

# Thread counts to test
THREAD_COUNTS = [16, 32, 64, 128, 256, 512]

# Beta threshold values to test (matching Table XII)
BETA_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


@dataclass
class RunResult:
    """Container for single run results."""
    betaThresh: float
    threads: int
    run: int
    tps: float
    avgBeta: float


@dataclass 
class ThresholdResult:
    """Container for results at a specific threshold."""
    betaThresh: float
    bestTps: float
    bestTpsMargin: float
    optimalN: int
    avgBeta: float
    avgBetaMargin: float
    nRuns: int


def computeCI(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute mean and CI margin using t-distribution."""
    n = len(data)
    if n < 2:
        return (data[0] if data else 0.0, 0.0)
    
    mean = statistics.mean(data)
    std = statistics.stdev(data)
    stderr = std / (n ** 0.5)
    
    if HAS_SCIPY:
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        t_table = {9: 2.262, 8: 2.306, 7: 2.365, 6: 2.447, 5: 2.571, 4: 2.776}
        t_value = t_table.get(n - 1, 2.262)
    
    margin = t_value * stderr
    return (mean, margin)


def ioDominantTask() -> Tuple[float, float]:
    """
    I/O dominant task with minimal CPU work.
    Used for threshold sensitivity analysis.
    
    Returns:
        (latency_seconds, beta)
    """
    wallStart = time.perf_counter()
    cpuStart = time.thread_time()
    
    # Minimal CPU phase
    x = 0
    for _ in range(CPU_ITERATIONS):
        x += 1
    
    # I/O phase (releases GIL)
    time.sleep(IO_SLEEP_MS / 1000.0)
    
    cpuEnd = time.thread_time()
    wallEnd = time.perf_counter()
    
    latency = wallEnd - wallStart
    cpuTime = cpuEnd - cpuStart
    
    # Blocking ratio
    beta = 1.0 - min(1.0, cpuTime / latency) if latency > 0 else 0.0
    
    return (latency, beta)


def runBenchmark(threadCount: int, numTasks: int) -> Tuple[float, float]:
    """
    Run benchmark with specified thread count.
    
    Returns:
        (tps, avg_beta)
    """
    betas = []
    
    startTime = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=threadCount) as executor:
        futures = [executor.submit(ioDominantTask) for _ in range(numTasks)]
        for future in as_completed(futures):
            _, beta = future.result()
            betas.append(beta)
    
    elapsed = time.perf_counter() - startTime
    
    return (numTasks / elapsed, statistics.mean(betas))


def simulateAdaptiveController(betaThresh: float) -> Dict[int, Dict]:
    """
    Simulate what thread count the adaptive controller would select
    at a given beta threshold.
    
    Returns:
        Dictionary with per-thread-count results
    """
    results = {}
    
    for threads in THREAD_COUNTS:
        tps, avgBeta = runBenchmark(threads, NUM_TASKS // 2)  # Quick scan
        
        # Controller logic: scale up if beta > threshold
        wouldAllow = avgBeta > betaThresh
        
        results[threads] = {
            "tps": tps,
            "beta": avgBeta,
            "wouldAllow": wouldAllow,
        }
    
    return results


def runExperiment() -> List[ThresholdResult]:
    """Run full threshold sensitivity experiment."""
    
    # Set CPU affinity to single core for consistency
    try:
        os.sched_setaffinity(0, {0})
        print("CPU affinity set to single core (Linux)")
    except (AttributeError, OSError):
        try:
            p = psutil.Process()
            p.cpu_affinity([0])
            print("CPU affinity set to single core (psutil)")
        except Exception as e:
            print(f"Warning: Could not set CPU affinity: {e}")
    
    print()
    print("=" * 70)
    print("Beta Threshold Sensitivity Experiment")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"Workload: I/O Dominant (CPU iter={CPU_ITERATIONS}, I/O sleep={IO_SLEEP_MS}ms)")
    print(f"Tasks per run: {NUM_TASKS}")
    print(f"Runs per config: {NUM_RUNS} (n=10 as per Section 3.3)")
    print()
    
    thresholdResults = []
    
    print(f"{'β_thresh':<10} | {'Best TPS (±CI)':<22} | {'Optimal N':<12} | {'Avg β (±CI)':<18}")
    print("-" * 70)
    
    for betaThresh in BETA_THRESHOLDS:
        # For each threshold, run multiple times to get statistics
        bestTpsPerRun = []
        optimalNPerRun = []
        allBetas = []
        
        for run in range(NUM_RUNS):
            # Find optimal N at this threshold
            bestTps = 0
            optimalN = THREAD_COUNTS[0]
            runBetas = []
            
            for threads in THREAD_COUNTS:
                tps, avgBeta = runBenchmark(threads, NUM_TASKS)
                runBetas.append(avgBeta)
                
                # Controller would allow this if beta > threshold
                if avgBeta > betaThresh and tps > bestTps:
                    bestTps = tps
                    optimalN = threads
            
            # If no configuration allowed (all betas below threshold), use minimum threads
            if bestTps == 0:
                tps, avgBeta = runBenchmark(THREAD_COUNTS[0], NUM_TASKS)
                bestTps = tps
                optimalN = THREAD_COUNTS[0]
            
            bestTpsPerRun.append(bestTps)
            optimalNPerRun.append(optimalN)
            allBetas.extend(runBetas)
        
        # Compute statistics
        tpsMean, tpsMargin = computeCI(bestTpsPerRun, CONFIDENCE_LEVEL)
        betaMean, betaMargin = computeCI(allBetas, CONFIDENCE_LEVEL)
        
        # Most common optimal N
        optimalN = max(set(optimalNPerRun), key=optimalNPerRun.count)
        
        thresholdResults.append(ThresholdResult(
            betaThresh=betaThresh,
            bestTps=tpsMean,
            bestTpsMargin=tpsMargin,
            optimalN=optimalN,
            avgBeta=betaMean,
            avgBetaMargin=betaMargin,
            nRuns=NUM_RUNS,
        ))
        
        tpsStr = f"{tpsMean:,.0f}±{tpsMargin:,.0f}"
        betaStr = f"{betaMean:.3f}±{betaMargin:.3f}"
        print(f"{betaThresh:<10} | {tpsStr:<22} | {optimalN:<12} | {betaStr:<18}")
    
    print("-" * 70)
    
    # Check stability across thresholds
    tpsValues = [r.bestTps for r in thresholdResults]
    tpsMean = statistics.mean(tpsValues)
    tpsStd = statistics.stdev(tpsValues) if len(tpsValues) > 1 else 0
    cv = (tpsStd / tpsMean * 100) if tpsMean > 0 else 0
    
    print(f"\nPerformance stability across thresholds:")
    print(f"  Mean TPS: {tpsMean:,.0f}")
    print(f"  Std TPS: {tpsStd:,.0f}")
    print(f"  Coefficient of Variation: {cv:.1f}%")
    print(f"  (Low CV indicates robustness to threshold selection)")
    
    return thresholdResults


def saveResults(results: List[ThresholdResult]):
    """Save results to CSV and JSON files."""
    os.makedirs("results", exist_ok=True)
    
    # Save as CSV
    with open("results/beta_threshold_sensitivity_with_ci.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beta_thresh", "best_tps_mean", "best_tps_margin", 
                        "optimal_n", "avg_beta_mean", "avg_beta_margin", "n_runs"])
        
        for r in results:
            writer.writerow([
                r.betaThresh,
                f"{r.bestTps:.0f}", f"{r.bestTpsMargin:.0f}",
                r.optimalN,
                f"{r.avgBeta:.3f}", f"{r.avgBetaMargin:.3f}",
                r.nRuns
            ])
    
    print(f"\nSaved: results/beta_threshold_sensitivity_with_ci.csv")
    
    # Save as JSON
    data = {
        "python_version": sys.version,
        "config": {
            "cpu_iterations": CPU_ITERATIONS,
            "io_sleep_ms": IO_SLEEP_MS,
            "tasks": NUM_TASKS,
            "runs": NUM_RUNS,
            "thread_counts": THREAD_COUNTS,
        },
        "results": [
            {
                "beta_thresh": r.betaThresh,
                "best_tps_mean": r.bestTps,
                "best_tps_margin": r.bestTpsMargin,
                "optimal_n": r.optimalN,
                "avg_beta_mean": r.avgBeta,
                "avg_beta_margin": r.avgBetaMargin,
            }
            for r in results
        ]
    }
    
    with open("results/beta_threshold_sensitivity_with_ci.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: results/beta_threshold_sensitivity_with_ci.json")


def generateLatexTable(results: List[ThresholdResult]):
    """Generate LaTeX table snippet for Table XII."""
    print()
    print("=" * 70)
    print("LaTeX Table Snippet (for Table XII)")
    print("=" * 70)
    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{$\\beta_{\\text{thresh}}$ Sensitivity Analysis (I/O Dominant Workload, mean $\\pm$ 95\\% CI, $n=10$)}")
    print("\\label{tab:threshold_sensitivity}")
    print("\\begin{tabular}{|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{$\\beta_{\\text{thresh}}$} & \\textbf{Best TPS} & \\textbf{Optimal $N$} & \\textbf{Avg $\\bar{\\beta}$} \\\\")
    print("\\hline")
    
    for r in results:
        tpsStr = f"{r.bestTps:,.0f} $\\pm$ {r.bestTpsMargin:,.0f}"
        betaStr = f"{r.avgBeta:.3f} $\\pm$ {r.avgBetaMargin:.3f}"
        print(f"{r.betaThresh} & {tpsStr} & {r.optimalN} & {betaStr} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    """Main entry point."""
    print()
    print("#" * 70)
    print("# Beta Threshold Sensitivity Experiment with Statistical CI")
    print("# Table XII of the paper")
    print("#" * 70)
    
    results = runExperiment()
    saveResults(results)
    generateLatexTable(results)
    
    print()
    print("Experiment complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
