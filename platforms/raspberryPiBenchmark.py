#!/usr/bin/env python3
# Raspberry Pi 4 GIL Saturation Cliff Benchmark.
# Designed to run natively on Raspberry Pi 4B (4 core ARM Cortex A72).
# This script demonstrates the saturation cliff phenomenon on real edge hardware.

import time
import random
import statistics
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List

# Set seed for reproducibility.
SEED = 17
random.seed(SEED)

# Configuration for Raspberry Pi 4 hardware.
CPU_ITERATIONS = 1000       # Approximately 0.1ms CPU work simulating model inference.
IO_SLEEP_MS = 0.1           # 0.1ms I/O wait simulating sensor or network latency.
TASK_COUNT = 20000          # Total tasks per configuration.
RUNS_PER_CONFIG = 3         # Multiple runs for statistical significance.

# Thread counts to evaluate for cliff detection.
THREAD_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Output directory for results.
RESULTS_DIR = "results"


@dataclass
class TaskResult:
    """Container for individual task execution metrics."""
    startTime: float
    endTime: float
    cpuTime: float
    
    @property
    def latency(self) -> float:
        """Calculate task latency in seconds."""
        return self.endTime - self.startTime
    
    @property
    def blockingRatio(self) -> float:
        """Calculate blocking ratio for workload classification."""
        wallTime = self.endTime - self.startTime
        if wallTime <= 0:
            return 0.0
        return 1.0 - min(1.0, self.cpuTime / wallTime)


def edgeAiTask() -> TaskResult:
    """
    Simulate a mixed CPU and I/O edge AI task.
    
    CPU Phase: Represents model inference and preprocessing that holds the GIL.
    I/O Phase: Represents sensor reads or network calls that release the GIL.
    """
    start = time.perf_counter()
    cpuStart = time.thread_time()
    
    # CPU-bound phase holding the GIL.
    x = 0
    for _ in range(CPU_ITERATIONS):
        x += 1
    
    # I/O-bound phase releasing the GIL.
    time.sleep(IO_SLEEP_MS / 1000)
    
    cpuEnd = time.thread_time()
    end = time.perf_counter()
    
    return TaskResult(start, end, cpuEnd - cpuStart)


def pureIoTask() -> TaskResult:
    """Pure I/O task for baseline comparison without GIL contention."""
    start = time.perf_counter()
    cpuStart = time.thread_time()
    
    time.sleep(IO_SLEEP_MS / 1000)
    
    cpuEnd = time.thread_time()
    end = time.perf_counter()
    
    return TaskResult(start, end, cpuEnd - cpuStart)


def runBenchmark(numThreads: int, taskCount: int, taskFunc) -> dict:
    """Execute benchmark with specified thread count and return metrics."""
    results: List[TaskResult] = []
    
    startTime = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=numThreads) as executor:
        futures = [executor.submit(taskFunc) for _ in range(taskCount)]
        for future in as_completed(futures):
            results.append(future.result())
    
    endTime = time.perf_counter()
    elapsed = endTime - startTime
    
    # Calculate latency metrics in milliseconds.
    latencies = [r.latency * 1000 for r in results]
    latencies.sort()
    
    # Calculate average blocking ratio for workload characterization.
    betas = [r.blockingRatio for r in results]
    
    return {
        "threads": numThreads,
        "elapsed": elapsed,
        "tps": taskCount / elapsed,
        "avgLat": statistics.mean(latencies),
        "p50Lat": latencies[len(latencies) // 2],
        "p95Lat": latencies[int(len(latencies) * 0.95)],
        "p99Lat": latencies[int(len(latencies) * 0.99)],
        "avgBeta": statistics.mean(betas),
    }


def getCpuInfo() -> dict:
    """Retrieve CPU information for the Raspberry Pi."""
    info = {"model": "Unknown", "cores": 0, "architecture": "Unknown"}
    
    try:
        with open("/proc/cpuinfo", "r") as f:
            content = f.read()
            
        for line in content.split("\n"):
            if "model name" in line.lower() or "Model" in line:
                info["model"] = line.split(":")[1].strip()
            if "processor" in line.lower():
                info["cores"] += 1
                
        import platform
        info["architecture"] = platform.machine()
        
    except Exception as e:
        print(f"Warning: Could not read CPU info: {e}")
    
    return info


def getMemoryInfo() -> dict:
    """Retrieve memory information for the Raspberry Pi."""
    info = {"totalMb": 0, "availableMb": 0}
    
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    info["totalMb"] = int(line.split()[1]) // 1024
                if "MemAvailable" in line:
                    info["availableMb"] = int(line.split()[1]) // 1024
    except Exception as e:
        print(f"Warning: Could not read memory info: {e}")
    
    return info


def main():
    """Main entry point for Raspberry Pi benchmark."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Print system information.
    print("-" * 70)
    print("GIL Saturation Cliff Benchmark - Raspberry Pi 4")
    print("-" * 70)
    
    cpuInfo = getCpuInfo()
    memInfo = getMemoryInfo()
    
    print(f"CPU: {cpuInfo['model']} ({cpuInfo['cores']} cores)")
    print(f"Architecture: {cpuInfo['architecture']}")
    print(f"Memory: {memInfo['totalMb']} MB total, {memInfo['availableMb']} MB available")
    print(f"Python: {__import__('sys').version}")
    print()
    print("Configuration:")
    print(f"  CPU Iterations: {CPU_ITERATIONS}")
    print(f"  I/O Sleep: {IO_SLEEP_MS} ms")
    print(f"  Tasks per config: {TASK_COUNT}")
    print(f"  Runs per config: {RUNS_PER_CONFIG}")
    print("-" * 70)
    print()
    
    # Run mixed workload experiment.
    print("Phase 1: Mixed CPU+I/O Workload")
    print("-" * 70)
    print(f"{'Threads':<8} | {'TPS':<12} | {'Avg Lat':<10} | {'P99 Lat':<10} | {'Beta':<6} | {'Status'}")
    print("-" * 70)
    
    mixedResults = []
    peakTps = 0
    peakThreads = 0
    
    for numThreads in THREAD_COUNTS:
        runResults = []
        for run in range(RUNS_PER_CONFIG):
            result = runBenchmark(numThreads, TASK_COUNT, edgeAiTask)
            result["run"] = run
            runResults.append(result)
            mixedResults.append(result)
        
        # Average across runs.
        avgTps = statistics.mean([r["tps"] for r in runResults])
        avgLat = statistics.mean([r["avgLat"] for r in runResults])
        avgP99 = statistics.mean([r["p99Lat"] for r in runResults])
        avgBeta = statistics.mean([r["avgBeta"] for r in runResults])
        
        if avgTps > peakTps:
            peakTps = avgTps
            peakThreads = numThreads
        
        drop = ((peakTps - avgTps) / peakTps) * 100
        
        if drop > 30:
            status = "CLIFF"
        elif drop > 15:
            status = "DROP"
        elif drop > 5:
            status = "DECLINE"
        else:
            status = "OK"
        
        print(f"{numThreads:<8} | {avgTps:<12,.0f} | {avgLat:<10.2f} | {avgP99:<10.2f} | {avgBeta:<6.2f} | {status}")
    
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
    
    # Save results to CSV.
    mixedCsvPath = os.path.join(RESULTS_DIR, "raspberrypi_mixed_workload.csv")
    with open(mixedCsvPath, "w", newline="") as f:
        fieldnames = ["threads", "run", "tps", "avgLat", "p50Lat", "p95Lat", "p99Lat", "avgBeta"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(mixedResults)
    
    ioCsvPath = os.path.join(RESULTS_DIR, "raspberrypi_io_baseline.csv")
    with open(ioCsvPath, "w", newline="") as f:
        fieldnames = ["threads", "tps", "avgLat"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ioResults)
    
    # Print summary.
    print()
    print("-" * 70)
    print("Results Summary")
    print("-" * 70)
    
    finalResult = [r for r in mixedResults if r["threads"] == THREAD_COUNTS[-1]]
    finalTps = statistics.mean([r["tps"] for r in finalResult])
    cliffSeverity = ((peakTps - finalTps) / peakTps) * 100
    
    print(f"Peak Throughput: {peakTps:,.0f} TPS at {peakThreads} threads")
    print(f"Final Throughput: {finalTps:,.0f} TPS at {THREAD_COUNTS[-1]} threads")
    print(f"Cliff Severity: {cliffSeverity:.1f}% degradation")
    print()
    print(f"Results saved to:")
    print(f"  {mixedCsvPath}")
    print(f"  {ioCsvPath}")
    
    if cliffSeverity >= 30:
        print()
        print("Strong cliff detected. This is a publishable result.")
    elif cliffSeverity >= 15:
        print()
        print("Moderate cliff detected. Evidence supports the saturation hypothesis.")
    
    return 0


if __name__ == "__main__":
    exit(main())
