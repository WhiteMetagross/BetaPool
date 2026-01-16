#!/usr/bin/env python3
"""
Pure I/O Baseline Experiment with Statistical Confidence Intervals.

This experiment generates data for Table V of the paper:
- Pure I/O Baseline (No GIL Contention)
- Linear scaling confirms saturation cliff is GIL-specific

Uses the statistical methodology from Section 3.3:
- n=10 runs per configuration
- 95% CI using t-distribution
- Pooled P99 latency computation
"""

import sys
import time
import random
import statistics
import csv
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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

# Experiment parameters.
IO_SLEEP_MS = 0.6     # I/O sleep duration (slightly longer than mixed workload)
NUM_TASKS = 20000     # Tasks per benchmark run
NUM_RUNS = 10         # n=10 runs for statistical significance (Section 3.3)
CONFIDENCE_LEVEL = 0.95

# Thread counts to test (matching Table V).
THREAD_COUNTS = [1, 4, 16, 64, 256]


@dataclass
class RunResult:
    """Container for single run results."""
    threads: int
    run: int
    tps: float
    avgLatencyMs: float
    p99LatencyMs: float
    rawLatencies: List[float] = field(default_factory=list)


@dataclass 
class AggregatedResult:
    """Container for aggregated results with confidence intervals."""
    threads: int
    tpsMean: float
    tpsMargin: float  # 95% CI margin
    avgLatMean: float
    avgLatMargin: float
    p99Pooled: float
    nRuns: int


def computeCI(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute mean and CI margin using t-distribution (Section 3.3).
    """
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


def computePooledP99(allLatencies: List[List[float]]) -> float:
    """Compute pooled P99 by aggregating per-task samples across runs."""
    pooled = []
    for runLats in allLatencies:
        pooled.extend(runLats)
    if not pooled:
        return 0.0
    pooled.sort()
    idx = int(len(pooled) * 0.99)
    return pooled[idx] * 1000  # Convert to ms


def pureIoTask() -> float:
    """
    Pure I/O task for baseline comparison without GIL contention.
    Only sleeps (releases GIL), no CPU computation.
    """
    start = time.perf_counter()
    time.sleep(IO_SLEEP_MS / 1000.0)
    return time.perf_counter() - start


def runBenchmark(threadCount: int, numTasks: int, runId: int) -> RunResult:
    """Run single benchmark iteration with specified thread count."""
    latencies = []
    
    startTime = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=threadCount) as executor:
        futures = [executor.submit(pureIoTask) for _ in range(numTasks)]
        for future in as_completed(futures):
            latencies.append(future.result())
    
    elapsed = time.perf_counter() - startTime
    
    # Sort latencies for percentile computation
    sortedLatencies = sorted(latencies)
    p99Idx = int(len(sortedLatencies) * 0.99)
    
    return RunResult(
        threads=threadCount,
        run=runId,
        tps=numTasks / elapsed,
        avgLatencyMs=statistics.mean(latencies) * 1000,
        p99LatencyMs=sortedLatencies[p99Idx] * 1000,
        rawLatencies=latencies,
    )


def runExperiment(coreConfig: str) -> Dict[int, AggregatedResult]:
    """
    Run full experiment for specified core configuration.
    """
    if coreConfig == "single":
        cores = [0]
        configName = "Single-Core"
    else:
        cores = [0, 1, 2, 3]
        configName = "Quad-Core"
    
    # Set CPU affinity
    try:
        os.sched_setaffinity(0, set(cores))
        print(f"CPU affinity set to cores {cores} (Linux)")
    except (AttributeError, OSError):
        try:
            p = psutil.Process()
            p.cpu_affinity(cores)
            print(f"CPU affinity set to cores {cores} (psutil)")
        except Exception as e:
            print(f"Warning: Could not set CPU affinity: {e}")
    
    print()
    print("=" * 70)
    print(f"Pure I/O Baseline Experiment: {configName}")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"I/O sleep: {IO_SLEEP_MS} ms")
    print(f"Tasks per run: {NUM_TASKS}")
    print(f"Runs per config: {NUM_RUNS} (n=10 as per Section 3.3)")
    print()
    
    results = []
    aggregated: Dict[int, AggregatedResult] = {}
    
    print(f"{'Threads':<8} | {'TPS (mean±CI)':<22} | {'Avg Lat (ms)':<18} | {'P99 Pooled':<12}")
    print("-" * 70)
    
    for threadCount in THREAD_COUNTS:
        runResults = []
        allLatencies = []
        
        for run in range(NUM_RUNS):
            result = runBenchmark(threadCount, NUM_TASKS, run)
            runResults.append(result)
            results.append(result)
            allLatencies.append(result.rawLatencies)
        
        # Compute aggregated statistics with CI
        tpsList = [r.tps for r in runResults]
        latList = [r.avgLatencyMs for r in runResults]
        
        tpsMean, tpsMargin = computeCI(tpsList, CONFIDENCE_LEVEL)
        latMean, latMargin = computeCI(latList, CONFIDENCE_LEVEL)
        p99Pooled = computePooledP99(allLatencies)
        
        aggregated[threadCount] = AggregatedResult(
            threads=threadCount,
            tpsMean=tpsMean,
            tpsMargin=tpsMargin,
            avgLatMean=latMean,
            avgLatMargin=latMargin,
            p99Pooled=p99Pooled,
            nRuns=NUM_RUNS,
        )
        
        tpsStr = f"{tpsMean:,.0f}±{tpsMargin:,.0f}"
        latStr = f"{latMean:.2f}±{latMargin:.2f}"
        print(f"{threadCount:<8} | {tpsStr:<22} | {latStr:<18} | {p99Pooled:<12.2f}")
    
    print("-" * 70)
    
    # Check for linear scaling
    tps1 = aggregated[1].tpsMean
    tps256 = aggregated[256].tpsMean
    scalingRatio = tps256 / tps1 if tps1 > 0 else 0
    print(f"Scaling ratio (256 threads / 1 thread): {scalingRatio:.1f}x")
    print("(Linear scaling would be ~256x for pure I/O)")
    
    return aggregated


def saveResults(singleCore: Dict[int, AggregatedResult], quadCore: Dict[int, AggregatedResult]):
    """Save results to CSV and JSON files."""
    os.makedirs("results", exist_ok=True)
    
    # Save combined results as CSV
    with open("results/io_baseline_with_ci.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "threads", "tps_mean", "tps_margin", 
                        "avg_lat_mean", "avg_lat_margin", "p99_pooled", "n_runs"])
        
        for t in THREAD_COUNTS:
            a = singleCore[t]
            writer.writerow(["single-core", t, f"{a.tpsMean:.0f}", f"{a.tpsMargin:.0f}",
                           f"{a.avgLatMean:.2f}", f"{a.avgLatMargin:.2f}", f"{a.p99Pooled:.2f}", a.nRuns])
        
        for t in THREAD_COUNTS:
            a = quadCore[t]
            writer.writerow(["quad-core", t, f"{a.tpsMean:.0f}", f"{a.tpsMargin:.0f}",
                           f"{a.avgLatMean:.2f}", f"{a.avgLatMargin:.2f}", f"{a.p99Pooled:.2f}", a.nRuns])
    
    print(f"\nSaved: results/io_baseline_with_ci.csv")
    
    # Save as JSON
    data = {
        "python_version": sys.version,
        "config": {"io_sleep_ms": IO_SLEEP_MS, "tasks": NUM_TASKS, "runs": NUM_RUNS},
        "single_core": {
            str(t): {
                "tps_mean": a.tpsMean,
                "tps_margin": a.tpsMargin,
                "avg_lat_mean": a.avgLatMean,
                "avg_lat_margin": a.avgLatMargin,
                "p99_pooled": a.p99Pooled,
            }
            for t, a in singleCore.items()
        },
        "quad_core": {
            str(t): {
                "tps_mean": a.tpsMean,
                "tps_margin": a.tpsMargin,
                "avg_lat_mean": a.avgLatMean,
                "avg_lat_margin": a.avgLatMargin,
                "p99_pooled": a.p99Pooled,
            }
            for t, a in quadCore.items()
        }
    }
    
    with open("results/io_baseline_with_ci.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: results/io_baseline_with_ci.json")


def generateLatexTable(singleCore: Dict[int, AggregatedResult], quadCore: Dict[int, AggregatedResult]):
    """Generate LaTeX table snippet for Table V."""
    print()
    print("=" * 70)
    print("LaTeX Table Snippet (for Table V)")
    print("=" * 70)
    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Pure I/O Baseline (No GIL Contention, mean $\\pm$ 95\\% CI, $n=10$). Linear scaling confirms saturation cliff is GIL-specific.}")
    print("\\label{tab:io_baseline}")
    print("\\begin{tabular}{|c|c|c|}")
    print("\\hline")
    print("\\textbf{Threads} & \\textbf{TPS (Single-Core)} & \\textbf{TPS (Quad-Core)} \\\\")
    print("\\hline")
    
    for t in THREAD_COUNTS:
        sc = singleCore[t]
        qc = quadCore[t]
        scStr = f"{sc.tpsMean:,.0f} $\\pm$ {sc.tpsMargin:,.0f}"
        qcStr = f"{qc.tpsMean:,.0f} $\\pm$ {qc.tpsMargin:,.0f}"
        print(f"{t} & {scStr} & {qcStr} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    """Main entry point."""
    print()
    print("#" * 70)
    print("# Pure I/O Baseline Experiment with Statistical CI")
    print("# Table V of the paper")
    print("#" * 70)
    
    # Run single-core experiment
    singleCore = runExperiment("single")
    
    # Run quad-core experiment
    quadCore = runExperiment("quad")
    
    # Save results
    saveResults(singleCore, quadCore)
    
    # Generate LaTeX table
    generateLatexTable(singleCore, quadCore)
    
    print()
    print("Experiment complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
