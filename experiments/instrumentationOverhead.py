#!/usr/bin/env python3
"""
Instrumentation Overhead Benchmark.

Measures the CPU and wall time overhead introduced by the blocking ratio
instrumentation (time.thread_time() and time.time() calls).

This addresses reviewer concern #2: instrumentation overhead disclosure.
"""

import time
import statistics
import csv
import os
from dataclasses import dataclass
from typing import List

# Number of iterations for statistical significance.
ITERATIONS = 1000000
WARMUP_ITERATIONS = 10000


@dataclass
class OverheadResult:
    """Container for overhead measurement results."""
    operation: str
    iterations: int
    totalTimeUs: float
    meanTimeUs: float
    stdTimeUs: float
    medianTimeUs: float
    p99TimeUs: float


def measureTimeDotTime() -> List[float]:
    """Measure overhead of time.time() calls."""
    measurements = []
    
    # Warmup phase.
    for _ in range(WARMUP_ITERATIONS):
        _ = time.time()
    
    # Measurement phase.
    for _ in range(ITERATIONS):
        start = time.perf_counter_ns()
        _ = time.time()
        end = time.perf_counter_ns()
        measurements.append((end - start) / 1000.0)  # Convert to microseconds.
    
    return measurements


def measureThreadTime() -> List[float]:
    """Measure overhead of time.thread_time() calls."""
    measurements = []
    
    # Warmup phase.
    for _ in range(WARMUP_ITERATIONS):
        _ = time.thread_time()
    
    # Measurement phase.
    for _ in range(ITERATIONS):
        start = time.perf_counter_ns()
        _ = time.thread_time()
        end = time.perf_counter_ns()
        measurements.append((end - start) / 1000.0)  # Convert to microseconds.
    
    return measurements


def measureCombinedInstrumentation() -> List[float]:
    """
    Measure overhead of the complete instrumentation pattern used in the executor.
    
    This matches the wrapper pattern in adaptiveExecutor.py:
        wallStart = time.time()
        cpuStart = time.thread_time()
        ... task execution ...
        cpuEnd = time.thread_time()
        wallEnd = time.time()
        beta = 1.0 - (cpuEnd - cpuStart) / (wallEnd - wallStart)
    """
    measurements = []
    
    # Warmup phase.
    for _ in range(WARMUP_ITERATIONS):
        wallStart = time.time()
        cpuStart = time.thread_time()
        cpuEnd = time.thread_time()
        wallEnd = time.time()
        if wallEnd > wallStart:
            _ = 1.0 - (cpuEnd - cpuStart) / (wallEnd - wallStart)
    
    # Measurement phase.
    for _ in range(ITERATIONS):
        start = time.perf_counter_ns()
        
        wallStart = time.time()
        cpuStart = time.thread_time()
        cpuEnd = time.thread_time()
        wallEnd = time.time()
        if wallEnd > wallStart:
            _ = 1.0 - (cpuEnd - cpuStart) / (wallEnd - wallStart)
        
        end = time.perf_counter_ns()
        measurements.append((end - start) / 1000.0)  # Convert to microseconds.
    
    return measurements


def measureNoopTask() -> List[float]:
    """Measure baseline overhead of a no-op task for comparison."""
    measurements = []
    
    def noopTask():
        pass
    
    # Warmup phase.
    for _ in range(WARMUP_ITERATIONS):
        noopTask()
    
    # Measurement phase.
    for _ in range(ITERATIONS):
        start = time.perf_counter_ns()
        noopTask()
        end = time.perf_counter_ns()
        measurements.append((end - start) / 1000.0)
    
    return measurements


def analyzeResults(measurements: List[float], operation: str) -> OverheadResult:
    """Compute statistics from measurement data."""
    measurements.sort()
    
    return OverheadResult(
        operation=operation,
        iterations=len(measurements),
        totalTimeUs=sum(measurements),
        meanTimeUs=statistics.mean(measurements),
        stdTimeUs=statistics.stdev(measurements) if len(measurements) > 1 else 0.0,
        medianTimeUs=measurements[len(measurements) // 2],
        p99TimeUs=measurements[int(len(measurements) * 0.99)],
    )


def main():
    """Run instrumentation overhead benchmark."""
    print()
    print("=" * 70)
    print("INSTRUMENTATION OVERHEAD BENCHMARK")
    print("=" * 70)
    print()
    print(f"Iterations: {ITERATIONS:,}")
    print(f"Warmup: {WARMUP_ITERATIONS:,}")
    print()
    
    results = []
    
    # Measure each operation.
    print("Measuring time.time() overhead...")
    timeMeasurements = measureTimeDotTime()
    results.append(analyzeResults(timeMeasurements, "time.time()"))
    
    print("Measuring time.thread_time() overhead...")
    threadTimeMeasurements = measureThreadTime()
    results.append(analyzeResults(threadTimeMeasurements, "time.thread_time()"))
    
    print("Measuring combined instrumentation overhead...")
    combinedMeasurements = measureCombinedInstrumentation()
    results.append(analyzeResults(combinedMeasurements, "Combined (full pattern)"))
    
    print("Measuring no-op baseline...")
    noopMeasurements = measureNoopTask()
    results.append(analyzeResults(noopMeasurements, "No-op baseline"))
    
    # Display results.
    print()
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)
    print()
    print(f"{'Operation':<25} | {'Mean (μs)':<10} | {'Std (μs)':<10} | {'Median (μs)':<12} | {'P99 (μs)':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r.operation:<25} | {r.meanTimeUs:<10.4f} | {r.stdTimeUs:<10.4f} | {r.medianTimeUs:<12.4f} | {r.p99TimeUs:<10.4f}")
    
    # Calculate net instrumentation overhead.
    combinedResult = results[2]
    noopResult = results[3]
    netOverhead = combinedResult.medianTimeUs - noopResult.medianTimeUs
    
    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print()
    print(f"Net instrumentation overhead (median): {netOverhead:.4f} μs per task")
    print(f"Net instrumentation overhead (mean): {combinedResult.meanTimeUs - noopResult.meanTimeUs:.4f} μs per task")
    print()
    
    # Context: typical CPU work in experiments.
    typicalCpuMs = 0.1  # 100 microseconds = 0.1 ms.
    overheadPct = (netOverhead / 1000.0) / typicalCpuMs * 100
    print(f"For typical T_CPU = {typicalCpuMs} ms workload:")
    print(f"  Overhead represents {overheadPct:.2f}% of CPU work.")
    print()
    
    # Save results to CSV.
    os.makedirs("results", exist_ok=True)
    with open("results/instrumentation_overhead.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["operation", "iterations", "mean_us", "std_us", "median_us", "p99_us"])
        for r in results:
            writer.writerow([r.operation, r.iterations, r.meanTimeUs, r.stdTimeUs, r.medianTimeUs, r.p99TimeUs])
    
    print("Results saved to results/instrumentation_overhead.csv")
    
    return 0


if __name__ == "__main__":
    exit(main())
