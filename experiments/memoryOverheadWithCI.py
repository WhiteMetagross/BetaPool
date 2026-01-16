#!/usr/bin/env python3
"""
Memory Overhead Experiment with Statistical Confidence Intervals.

This experiment generates data for Table IX of the paper:
- Memory Overhead: ThreadPool vs ProcessPool
- RSS (Resident Set Size) measurements

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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
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

# Experiment parameters.
CPU_ITERATIONS = 1000
IO_SLEEP_MS = 0.1
NUM_TASKS = 5000
NUM_RUNS = 10         # n=10 runs for statistical significance
CONFIDENCE_LEVEL = 0.95

# Worker counts to test (matching Table IX).
THREADPOOL_WORKERS = [4, 32, 64]
PROCESSPOOL_WORKERS = [4, 8, 16]


@dataclass
class MemoryResult:
    """Container for single run results."""
    strategy: str
    workers: int
    run: int
    memoryMb: float
    overheadMb: float
    tps: float


@dataclass 
class AggregatedResult:
    """Container for aggregated results with confidence intervals."""
    strategy: str
    workers: int
    memoryMean: float
    memoryMargin: float
    overheadMean: float
    overheadMargin: float
    tpsMean: float
    tpsMargin: float
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


def getMemoryUsageMb() -> float:
    """Get current process memory usage (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def mixedTask() -> float:
    """Standard mixed CPU+IO task."""
    start = time.perf_counter()
    
    # CPU phase
    x = 0
    for _ in range(CPU_ITERATIONS):
        x += 1
    
    # IO phase
    time.sleep(IO_SLEEP_MS / 1000.0)
    
    return time.perf_counter() - start


def mixedTaskForProcess(args=None) -> float:
    """Mixed task suitable for multiprocessing (must be picklable)."""
    start = time.perf_counter()
    
    x = 0
    for _ in range(CPU_ITERATIONS):
        x += 1
    
    time.sleep(IO_SLEEP_MS / 1000.0)
    
    return time.perf_counter() - start


def runThreadPoolBenchmark(numWorkers: int, numTasks: int) -> Tuple[float, float, float]:
    """
    Run ThreadPoolExecutor benchmark with memory measurement.
    
    Returns:
        (memory_mb, overhead_mb, tps)
    """
    # Measure baseline memory
    import gc
    gc.collect()
    time.sleep(0.1)
    memBefore = getMemoryUsageMb()
    
    startTime = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=numWorkers) as executor:
        # Give pool time to spawn threads
        time.sleep(0.1)
        memDuring = getMemoryUsageMb()
        
        futures = [executor.submit(mixedTask) for _ in range(numTasks)]
        for future in as_completed(futures):
            future.result()
    
    elapsed = time.perf_counter() - startTime
    
    return (memDuring, memDuring - memBefore, numTasks / elapsed)


def runProcessPoolBenchmark(numWorkers: int, numTasks: int) -> Tuple[float, float, float]:
    """
    Run ProcessPoolExecutor benchmark with memory measurement.
    
    Returns:
        (memory_mb, overhead_mb, tps)
    """
    import gc
    gc.collect()
    time.sleep(0.1)
    memBefore = getMemoryUsageMb()
    
    startTime = time.perf_counter()
    
    with ProcessPoolExecutor(max_workers=numWorkers) as executor:
        # Give pool time to spawn processes
        time.sleep(0.5)
        memDuring = getMemoryUsageMb()
        
        futures = [executor.submit(mixedTaskForProcess) for _ in range(numTasks)]
        for future in as_completed(futures):
            future.result()
    
    elapsed = time.perf_counter() - startTime
    
    return (memDuring, memDuring - memBefore, numTasks / elapsed)


def runExperiment() -> Dict[str, Dict[int, AggregatedResult]]:
    """Run full experiment for both ThreadPool and ProcessPool."""
    
    print()
    print("=" * 70)
    print("Memory Overhead Experiment: ThreadPool vs ProcessPool")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"Tasks per run: {NUM_TASKS}")
    print(f"Runs per config: {NUM_RUNS} (n=10 as per Section 3.3)")
    print()
    
    results = {"ThreadPool": {}, "ProcessPool": {}}
    allResults = []
    
    # Test ThreadPool
    print("Testing: ThreadPoolExecutor")
    print("-" * 70)
    print(f"{'Workers':<10} | {'Memory (MB)':<18} | {'Overhead (MB)':<18} | {'TPS':<15}")
    print("-" * 70)
    
    for workers in THREADPOOL_WORKERS:
        runResults = []
        
        for run in range(NUM_RUNS):
            memory, overhead, tps = runThreadPoolBenchmark(workers, NUM_TASKS)
            runResults.append({
                "memory": memory,
                "overhead": overhead,
                "tps": tps
            })
            allResults.append(MemoryResult(
                strategy="ThreadPool",
                workers=workers,
                run=run,
                memoryMb=memory,
                overheadMb=overhead,
                tps=tps,
            ))
        
        # Compute aggregated statistics
        memMean, memMargin = computeCI([r["memory"] for r in runResults])
        ovhMean, ovhMargin = computeCI([r["overhead"] for r in runResults])
        tpsMean, tpsMargin = computeCI([r["tps"] for r in runResults])
        
        results["ThreadPool"][workers] = AggregatedResult(
            strategy="ThreadPool",
            workers=workers,
            memoryMean=memMean,
            memoryMargin=memMargin,
            overheadMean=ovhMean,
            overheadMargin=ovhMargin,
            tpsMean=tpsMean,
            tpsMargin=tpsMargin,
            nRuns=NUM_RUNS,
        )
        
        memStr = f"{memMean:.1f}±{memMargin:.1f}"
        ovhStr = f"{ovhMean:.1f}±{ovhMargin:.1f}"
        tpsStr = f"{tpsMean:.1f}±{tpsMargin:.1f}"
        print(f"{workers:<10} | {memStr:<18} | {ovhStr:<18} | {tpsStr:<15}")
    
    # Test ProcessPool
    print()
    print("Testing: ProcessPoolExecutor")
    print("-" * 70)
    print(f"{'Workers':<10} | {'Memory (MB)':<18} | {'Overhead (MB)':<18} | {'TPS':<15}")
    print("-" * 70)
    
    for workers in PROCESSPOOL_WORKERS:
        runResults = []
        
        for run in range(NUM_RUNS):
            memory, overhead, tps = runProcessPoolBenchmark(workers, NUM_TASKS)
            runResults.append({
                "memory": memory,
                "overhead": overhead,
                "tps": tps
            })
            allResults.append(MemoryResult(
                strategy="ProcessPool",
                workers=workers,
                run=run,
                memoryMb=memory,
                overheadMb=overhead,
                tps=tps,
            ))
        
        # Compute aggregated statistics
        memMean, memMargin = computeCI([r["memory"] for r in runResults])
        ovhMean, ovhMargin = computeCI([r["overhead"] for r in runResults])
        tpsMean, tpsMargin = computeCI([r["tps"] for r in runResults])
        
        results["ProcessPool"][workers] = AggregatedResult(
            strategy="ProcessPool",
            workers=workers,
            memoryMean=memMean,
            memoryMargin=memMargin,
            overheadMean=ovhMean,
            overheadMargin=ovhMargin,
            tpsMean=tpsMean,
            tpsMargin=tpsMargin,
            nRuns=NUM_RUNS,
        )
        
        memStr = f"{memMean:.1f}±{memMargin:.1f}"
        ovhStr = f"{ovhMean:.1f}±{ovhMargin:.1f}"
        tpsStr = f"{tpsMean:.1f}±{tpsMargin:.1f}"
        print(f"{workers:<10} | {memStr:<18} | {ovhStr:<18} | {tpsStr:<15}")
    
    print("-" * 70)
    
    # Calculate memory overhead ratio
    threadMem = results["ThreadPool"][32].memoryMean
    processMem = results["ProcessPool"][8].memoryMean
    ratio = processMem / threadMem if threadMem > 0 else 0
    print(f"\nMemory ratio (ProcessPool-8 / ThreadPool-32): {ratio:.1f}x")
    
    return results


def saveResults(results: Dict[str, Dict[int, AggregatedResult]]):
    """Save results to CSV and JSON files."""
    os.makedirs("results", exist_ok=True)
    
    # Save as CSV
    with open("results/memory_overhead_with_ci.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "workers", "memory_mean", "memory_margin",
                        "overhead_mean", "overhead_margin", "tps_mean", "tps_margin", "n_runs"])
        
        for strategy in ["ThreadPool", "ProcessPool"]:
            workerList = THREADPOOL_WORKERS if strategy == "ThreadPool" else PROCESSPOOL_WORKERS
            for workers in workerList:
                a = results[strategy][workers]
                writer.writerow([
                    strategy, workers,
                    f"{a.memoryMean:.1f}", f"{a.memoryMargin:.1f}",
                    f"{a.overheadMean:.1f}", f"{a.overheadMargin:.1f}",
                    f"{a.tpsMean:.1f}", f"{a.tpsMargin:.1f}",
                    a.nRuns
                ])
    
    print(f"\nSaved: results/memory_overhead_with_ci.csv")
    
    # Save as JSON
    data = {
        "python_version": sys.version,
        "config": {"cpu_iterations": CPU_ITERATIONS, "io_sleep_ms": IO_SLEEP_MS, 
                   "tasks": NUM_TASKS, "runs": NUM_RUNS},
        "results": {}
    }
    
    for strategy in ["ThreadPool", "ProcessPool"]:
        workerList = THREADPOOL_WORKERS if strategy == "ThreadPool" else PROCESSPOOL_WORKERS
        data["results"][strategy] = {}
        for workers in workerList:
            a = results[strategy][workers]
            data["results"][strategy][str(workers)] = {
                "memory_mean": a.memoryMean,
                "memory_margin": a.memoryMargin,
                "overhead_mean": a.overheadMean,
                "overhead_margin": a.overheadMargin,
                "tps_mean": a.tpsMean,
                "tps_margin": a.tpsMargin,
            }
    
    with open("results/memory_overhead_with_ci.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: results/memory_overhead_with_ci.json")


def generateLatexTable(results: Dict[str, Dict[int, AggregatedResult]]):
    """Generate LaTeX table snippet for Table IX."""
    print()
    print("=" * 70)
    print("LaTeX Table Snippet (for Table IX)")
    print("=" * 70)
    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Memory Overhead: ThreadPool vs ProcessPool (mean $\\pm$ 95\\% CI, $n=10$). Memory values are RSS (Resident Set Size).}")
    print("\\label{tab:memory_comparison}")
    print("\\begin{tabular}{|l|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Strategy} & \\textbf{Workers} & \\textbf{Memory (MB)} & \\textbf{Overhead (MB)} & \\textbf{TPS} \\\\")
    print("\\hline")
    
    for workers in THREADPOOL_WORKERS:
        a = results["ThreadPool"][workers]
        memStr = f"{a.memoryMean:.1f} $\\pm$ {a.memoryMargin:.1f}"
        ovhStr = f"{a.overheadMean:.1f} $\\pm$ {a.overheadMargin:.1f}"
        tpsStr = f"{a.tpsMean:.1f} $\\pm$ {a.tpsMargin:.1f}"
        print(f"ThreadPool & {workers} & {memStr} & {ovhStr} & {tpsStr} \\\\")
    
    print("\\hline")
    
    for workers in PROCESSPOOL_WORKERS:
        a = results["ProcessPool"][workers]
        memStr = f"{a.memoryMean:.1f} $\\pm$ {a.memoryMargin:.1f}"
        ovhStr = f"{a.overheadMean:.1f} $\\pm$ {a.overheadMargin:.1f}"
        tpsStr = f"{a.tpsMean:.1f} $\\pm$ {a.tpsMargin:.1f}"
        print(f"ProcessPool & {workers} & {memStr} & {ovhStr} & {tpsStr} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    """Main entry point."""
    print()
    print("#" * 70)
    print("# Memory Overhead Experiment with Statistical CI")
    print("# Table IX of the paper")
    print("#" * 70)
    
    results = runExperiment()
    saveResults(results)
    generateLatexTable(results)
    
    print()
    print("Experiment complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
