#!/usr/bin/env python3
"""
GIL vs No-GIL Experiment with Statistical Confidence Intervals.

This experiment generates data for Tables I and II of the paper:
- Table I: Single-Core Python 3.11 (GIL) vs 3.13t (no-GIL)
- Table II: Quad-Core Python 3.11 (GIL) vs 3.13t (no-GIL)

Uses the statistical methodology from Section 3.3:
- n=10 runs per configuration
- 95% CI using t-distribution
- Pooled P99 latency computation

Note: This script should be run twice - once with Python 3.11 (GIL)
and once with Python 3.13t (no-GIL). Results are saved to separate files.
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

# Attempt to import scipy for t-distribution; fallback to approximation
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Using approximate CI calculation.")

# Set seed for reproducibility.
SEED = 17
random.seed(SEED)

# Experiment parameters matching paper Section 2.4 (Tables I & II).
# Mixed workload: T_CPU = 10ms, T_IO = 50ms
CPU_MS = 10       # CPU phase duration in milliseconds
IO_MS = 50        # I/O phase duration in milliseconds
NUM_TASKS = 500   # Tasks per benchmark run
NUM_RUNS = 10     # n=10 runs for statistical significance (Section 3.3)
CONFIDENCE_LEVEL = 0.95

# Thread counts to test (matching Tables I and II).
THREAD_COUNTS = [1, 32, 256, 1024]


@dataclass
class RunResult:
    """Container for single run results."""
    threads: int
    run: int
    tps: float
    avgBeta: float
    p99LatencyMs: float
    rawLatencies: List[float] = field(default_factory=list)
    rawBetas: List[float] = field(default_factory=list)


@dataclass 
class AggregatedResult:
    """Container for aggregated results with confidence intervals."""
    threads: int
    tpsMean: float
    tpsMargin: float  # 95% CI margin
    betaMean: float
    betaMargin: float
    p99Pooled: float
    nRuns: int


def computeCI(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute mean and CI margin using t-distribution (Section 3.3).
    
    Returns:
        (mean, margin) where CI = mean ± margin
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
        # Approximation for t-distribution with n-1 degrees of freedom
        # Using t ≈ 2.262 for n=10, α=0.05 (two-tailed)
        t_table = {9: 2.262, 8: 2.306, 7: 2.365, 6: 2.447, 5: 2.571, 4: 2.776}
        t_value = t_table.get(n - 1, 2.262)
    
    margin = t_value * stderr
    return (mean, margin)


def computePooledP99(allLatencies: List[List[float]]) -> float:
    """
    Compute pooled P99 by aggregating per-task samples across runs (Section 3.3).
    
    Returns:
        P99 latency in milliseconds
    """
    pooled = []
    for runLats in allLatencies:
        pooled.extend(runLats)
    if not pooled:
        return 0.0
    pooled.sort()
    idx = int(len(pooled) * 0.99)
    return pooled[idx] * 1000  # Convert to ms


def mixedWorkloadTask(cpuMs: float = CPU_MS, ioMs: float = IO_MS) -> Tuple[float, float]:
    """
    Synthetic workload simulating edge AI orchestration:
    - CPU phase: computation that holds GIL
    - I/O phase: sleep that releases GIL
    
    Returns:
        (latency_seconds, beta) where beta = blocking ratio
    """
    wallStart = time.perf_counter()
    cpuStart = time.thread_time()
    
    # CPU phase (busy loop consuming CPU time)
    targetCpuTime = cpuMs / 1000.0
    result = 0
    while (time.thread_time() - cpuStart) < targetCpuTime:
        result += 1
    
    # I/O phase (releases GIL)
    time.sleep(ioMs / 1000.0)
    
    cpuEnd = time.thread_time()
    wallEnd = time.perf_counter()
    
    latency = wallEnd - wallStart
    cpuTime = cpuEnd - cpuStart
    
    # Blocking ratio: fraction of time NOT on CPU
    beta = 1.0 - min(1.0, cpuTime / latency) if latency > 0 else 0.0
    
    return (latency, beta)


def runBenchmark(threadCount: int, numTasks: int, runId: int) -> RunResult:
    """
    Run single benchmark iteration with specified thread count.
    
    Returns:
        RunResult with TPS, beta, and latency data
    """
    latencies = []
    betas = []
    
    startTime = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=threadCount) as executor:
        futures = [executor.submit(mixedWorkloadTask) for _ in range(numTasks)]
        for future in as_completed(futures):
            lat, beta = future.result()
            latencies.append(lat)
            betas.append(beta)
    
    elapsed = time.perf_counter() - startTime
    
    # Sort latencies for percentile computation
    sortedLatencies = sorted(latencies)
    p99Idx = int(len(sortedLatencies) * 0.99)
    p99LatencyMs = sortedLatencies[p99Idx] * 1000
    
    return RunResult(
        threads=threadCount,
        run=runId,
        tps=numTasks / elapsed,
        avgBeta=statistics.mean(betas),
        p99LatencyMs=p99LatencyMs,
        rawLatencies=latencies,
        rawBetas=betas,
    )


def runExperiment(coreConfig: str) -> Dict[int, AggregatedResult]:
    """
    Run full experiment for specified core configuration.
    
    Args:
        coreConfig: "single" for 1 core, "quad" for 4 cores
        
    Returns:
        Dictionary mapping thread count to aggregated results
    """
    # Set CPU affinity based on configuration
    if coreConfig == "single":
        cores = [0]
        configName = "Single-Core"
    else:
        cores = [0, 1, 2, 3]
        configName = "Quad-Core"
    
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
    
    # Detect GIL status
    try:
        gilEnabled = sys._is_gil_enabled()
        gilStatus = "GIL" if gilEnabled else "no-GIL"
    except AttributeError:
        gilEnabled = True
        gilStatus = "GIL"
    
    print()
    print("=" * 70)
    print(f"GIL vs No-GIL Experiment: {configName}")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"GIL status: {gilStatus}")
    print(f"Configuration: T_CPU={CPU_MS}ms, T_IO={IO_MS}ms")
    print(f"Tasks per run: {NUM_TASKS}")
    print(f"Runs per config: {NUM_RUNS} (n=10 as per Section 3.3)")
    print(f"Confidence level: {CONFIDENCE_LEVEL*100:.0f}%")
    print()
    
    results = []
    aggregated: Dict[int, AggregatedResult] = {}
    
    print(f"{'Threads':<8} | {'TPS (mean±CI)':<20} | {'Beta (mean±CI)':<18} | {'P99 Pooled':<12}")
    print("-" * 70)
    
    for threadCount in THREAD_COUNTS:
        runResults = []
        allLatencies = []
        allBetas = []
        
        for run in range(NUM_RUNS):
            result = runBenchmark(threadCount, NUM_TASKS, run)
            runResults.append(result)
            results.append(result)
            allLatencies.append(result.rawLatencies)
            allBetas.extend(result.rawBetas)
        
        # Compute aggregated statistics with CI (Section 3.3)
        tpsList = [r.tps for r in runResults]
        betaList = [r.avgBeta for r in runResults]
        
        tpsMean, tpsMargin = computeCI(tpsList, CONFIDENCE_LEVEL)
        betaMean, betaMargin = computeCI(betaList, CONFIDENCE_LEVEL)
        p99Pooled = computePooledP99(allLatencies)
        
        aggregated[threadCount] = AggregatedResult(
            threads=threadCount,
            tpsMean=tpsMean,
            tpsMargin=tpsMargin,
            betaMean=betaMean,
            betaMargin=betaMargin,
            p99Pooled=p99Pooled,
            nRuns=NUM_RUNS,
        )
        
        tpsStr = f"{tpsMean:.1f}±{tpsMargin:.1f}"
        betaStr = f"{betaMean:.2f}±{betaMargin:.2f}"
        print(f"{threadCount:<8} | {tpsStr:<20} | {betaStr:<18} | {p99Pooled:<12.2f}")
    
    # Compute degradation from peak
    peakTps = max(a.tpsMean for a in aggregated.values())
    finalTps = aggregated[THREAD_COUNTS[-1]].tpsMean
    degradation = ((peakTps - finalTps) / peakTps) * 100
    
    print("-" * 70)
    print(f"Peak TPS: {peakTps:.1f}")
    print(f"Final TPS (at {THREAD_COUNTS[-1]} threads): {finalTps:.1f}")
    print(f"Degradation: {degradation:.1f}%")
    
    return aggregated


def saveResults(singleCore: Dict[int, AggregatedResult], quadCore: Dict[int, AggregatedResult]):
    """Save results to CSV and JSON files."""
    os.makedirs("results", exist_ok=True)
    
    # Detect runtime for filename
    try:
        gilEnabled = sys._is_gil_enabled()
        runtime = "gil" if gilEnabled else "nogil"
    except AttributeError:
        runtime = "gil"
    
    version = f"{sys.version_info.major}{sys.version_info.minor}"
    
    # Save single-core results
    filename = f"results/gilnogil_{runtime}_py{version}_singlecore.json"
    data = {
        "python_version": sys.version,
        "runtime": runtime,
        "config": {"cpu_ms": CPU_MS, "io_ms": IO_MS, "tasks": NUM_TASKS, "runs": NUM_RUNS},
        "results": {
            str(t): {
                "tps_mean": a.tpsMean,
                "tps_margin": a.tpsMargin,
                "beta_mean": a.betaMean,
                "beta_margin": a.betaMargin,
                "p99_pooled": a.p99Pooled,
            }
            for t, a in singleCore.items()
        }
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {filename}")
    
    # Save quad-core results
    filename = f"results/gilnogil_{runtime}_py{version}_quadcore.json"
    data["results"] = {
        str(t): {
            "tps_mean": a.tpsMean,
            "tps_margin": a.tpsMargin,
            "beta_mean": a.betaMean,
            "beta_margin": a.betaMargin,
            "p99_pooled": a.p99Pooled,
        }
        for t, a in quadCore.items()
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filename}")


def generateLatexTables(singleCore: Dict[int, AggregatedResult], quadCore: Dict[int, AggregatedResult]):
    """Generate LaTeX table snippets for Tables I and II."""
    
    # Detect runtime
    try:
        gilEnabled = sys._is_gil_enabled()
        runtime = "3.11" if gilEnabled else "3.13t"
    except AttributeError:
        runtime = "3.11"
    
    print()
    print("=" * 70)
    print("LaTeX Table Snippet (for Tables I and II)")
    print("=" * 70)
    print()
    print(f"% Data for Python {runtime}")
    print()
    
    # Table I (Single-Core)
    print("% Table I: Single-Core")
    print("\\begin{tabular}{|c|c|c|}")
    print("\\hline")
    print(f"\\textbf{{Threads}} & \\textbf{{{runtime} TPS}} & \\textbf{{{runtime} $\\bar{{\\beta}}$}} \\\\")
    print("\\hline")
    
    peakTps = max(a.tpsMean for a in singleCore.values())
    for t in THREAD_COUNTS:
        a = singleCore[t]
        tpsStr = f"{a.tpsMean:.1f} $\\pm$ {a.tpsMargin:.1f}"
        betaStr = f"{a.betaMean:.2f} $\\pm$ {a.betaMargin:.2f}"
        if a.tpsMean == peakTps:
            tpsStr = f"\\textbf{{{a.tpsMean:.1f}}} $\\pm$ {a.tpsMargin:.1f}"
        print(f"{t} & {tpsStr} & {betaStr} \\\\")
    
    finalTps = singleCore[THREAD_COUNTS[-1]].tpsMean
    degradation = ((peakTps - finalTps) / peakTps) * 100
    print("\\hline")
    print(f"\\textbf{{Degradation}} & \\textbf{{{degradation:.1f}\\%}} & -- \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print()
    
    # Table II (Quad-Core)
    print("% Table II: Quad-Core")
    print("\\begin{tabular}{|c|c|c|}")
    print("\\hline")
    print(f"\\textbf{{Threads}} & \\textbf{{{runtime} TPS}} & \\textbf{{{runtime} $\\bar{{\\beta}}$}} \\\\")
    print("\\hline")
    
    peakTps = max(a.tpsMean for a in quadCore.values())
    for t in THREAD_COUNTS:
        a = quadCore[t]
        tpsStr = f"{a.tpsMean:.1f} $\\pm$ {a.tpsMargin:.1f}"
        betaStr = f"{a.betaMean:.2f} $\\pm$ {a.betaMargin:.2f}"
        if a.tpsMean == peakTps:
            tpsStr = f"\\textbf{{{a.tpsMean:.1f}}} $\\pm$ {a.tpsMargin:.1f}"
        print(f"{t} & {tpsStr} & {betaStr} \\\\")
    
    finalTps = quadCore[THREAD_COUNTS[-1]].tpsMean
    change = ((finalTps - peakTps) / peakTps) * 100
    changeStr = f"+{change:.1f}\\%" if change > 0 else f"{change:.1f}\\%"
    print("\\hline")
    print(f"\\textbf{{Change}} & \\textbf{{{changeStr}}} & -- \\\\")
    print("\\hline")
    print("\\end{tabular}")


def main():
    """Main entry point."""
    print()
    print("#" * 70)
    print("# GIL vs No-GIL Experiment with Statistical CI")
    print("# Tables I and II of the paper")
    print("#" * 70)
    
    # Run single-core experiment
    singleCore = runExperiment("single")
    
    # Run quad-core experiment
    quadCore = runExperiment("quad")
    
    # Save results
    saveResults(singleCore, quadCore)
    
    # Generate LaTeX tables
    generateLatexTables(singleCore, quadCore)
    
    print()
    print("Experiment complete!")
    print("Run this script with both Python 3.11 and Python 3.13t to get")
    print("complete data for Tables I and II.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
