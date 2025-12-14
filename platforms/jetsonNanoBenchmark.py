#!/usr/bin/env python3
# NVIDIA Jetson Nano GIL Saturation Cliff Benchmark.
# Designed to run natively on Jetson Nano (4 core ARM Cortex A57).
# Includes GPU availability detection for future CUDA workload extensions.

import time
import random
import statistics
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

# Set seed for reproducibility.
SEED = 17
random.seed(SEED)

# Configuration for Jetson Nano hardware.
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
    
    CPU Phase: Represents model inference preprocessing that holds the GIL.
    I/O Phase: Represents sensor reads or network calls that release the GIL.
    
    Note: On Jetson, actual GPU inference would release the GIL during CUDA calls.
    This benchmark isolates the Python orchestration layer performance.
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


def getJetsonInfo() -> dict:
    """Retrieve Jetson-specific hardware information."""
    info = {
        "model": "Unknown",
        "cores": 0,
        "architecture": "Unknown",
        "l4tVersion": "Unknown",
        "cudaAvailable": False,
        "gpuName": "Unknown",
    }
    
    # Get CPU information.
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
    
    # Check for Jetson-specific L4T version.
    try:
        if os.path.exists("/etc/nv_tegra_release"):
            with open("/etc/nv_tegra_release", "r") as f:
                info["l4tVersion"] = f.read().strip()[:50]
    except Exception:
        pass
    
    # Check for CUDA availability.
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info["cudaAvailable"] = True
            info["gpuName"] = result.stdout.strip()
    except Exception:
        pass
    
    # Alternative CUDA check via tegrastats.
    if not info["cudaAvailable"]:
        try:
            if os.path.exists("/usr/bin/tegrastats"):
                info["cudaAvailable"] = True
                info["gpuName"] = "Tegra GPU (Maxwell)"
        except Exception:
            pass
    
    return info


def getMemoryInfo() -> dict:
    """Retrieve memory information."""
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


def getThermalInfo() -> Optional[float]:
    """Read current CPU temperature for Jetson thermal monitoring."""
    thermalPaths = [
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone0/temp",
    ]
    
    for path in thermalPaths:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    temp = int(f.read().strip()) / 1000.0
                    return temp
        except Exception:
            continue
    
    return None


def main():
    """Main entry point for Jetson Nano benchmark."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Print system information.
    print("-" * 70)
    print("GIL Saturation Cliff Benchmark - NVIDIA Jetson Nano")
    print("-" * 70)
    
    jetsonInfo = getJetsonInfo()
    memInfo = getMemoryInfo()
    initialTemp = getThermalInfo()
    
    print(f"CPU: {jetsonInfo['model']} ({jetsonInfo['cores']} cores)")
    print(f"Architecture: {jetsonInfo['architecture']}")
    print(f"GPU: {jetsonInfo['gpuName']}")
    print(f"CUDA Available: {jetsonInfo['cudaAvailable']}")
    print(f"Memory: {memInfo['totalMb']} MB total, {memInfo['availableMb']} MB available")
    if initialTemp:
        print(f"CPU Temperature: {initialTemp:.1f} C")
    print(f"Python: {sys.version}")
    print()
    print("Configuration:")
    print(f"  CPU Iterations: {CPU_ITERATIONS}")
    print(f"  I/O Sleep: {IO_SLEEP_MS} ms")
    print(f"  Tasks per config: {TASK_COUNT}")
    print(f"  Runs per config: {RUNS_PER_CONFIG}")
    print("-" * 70)
    print()
    
    # Thermal warning for Jetson.
    if initialTemp and initialTemp > 60:
        print("Warning: CPU temperature is elevated. Results may be affected by throttling.")
        print("Consider allowing the device to cool before running benchmarks.")
        print()
    
    # Run mixed workload experiment.
    print("Phase 1: Mixed CPU+I/O Workload (Python Orchestration Layer)")
    print("-" * 70)
    print(f"{'Threads':<8} | {'TPS':<12} | {'Avg Lat':<10} | {'P99 Lat':<10} | {'Beta':<6} | {'Temp':<6} | {'Status'}")
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
        
        currentTemp = getThermalInfo()
        tempStr = f"{currentTemp:.0f}C" if currentTemp else "N/A"
        
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
        
        print(f"{numThreads:<8} | {avgTps:<12,.0f} | {avgLat:<10.2f} | {avgP99:<10.2f} | {avgBeta:<6.2f} | {tempStr:<6} | {status}")
    
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
    mixedCsvPath = os.path.join(RESULTS_DIR, "jetson_mixed_workload.csv")
    with open(mixedCsvPath, "w", newline="") as f:
        fieldnames = ["threads", "run", "tps", "avgLat", "p50Lat", "p95Lat", "p99Lat", "avgBeta"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(mixedResults)
    
    ioCsvPath = os.path.join(RESULTS_DIR, "jetson_io_baseline.csv")
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
    
    finalTemp = getThermalInfo()
    
    print(f"Peak Throughput: {peakTps:,.0f} TPS at {peakThreads} threads")
    print(f"Final Throughput: {finalTps:,.0f} TPS at {THREAD_COUNTS[-1]} threads")
    print(f"Cliff Severity: {cliffSeverity:.1f}% degradation")
    if initialTemp and finalTemp:
        print(f"Temperature Change: {initialTemp:.1f}C -> {finalTemp:.1f}C")
    print()
    print(f"Results saved to:")
    print(f"  {mixedCsvPath}")
    print(f"  {ioCsvPath}")
    
    if cliffSeverity >= 30:
        print()
        print("Strong cliff detected. This result demonstrates GIL saturation on Jetson.")
    elif cliffSeverity >= 15:
        print()
        print("Moderate cliff detected. Evidence supports the saturation hypothesis.")
    
    # Note about GPU offloading.
    if jetsonInfo["cudaAvailable"]:
        print()
        print("Note: This benchmark measures Python orchestration layer performance only.")
        print("Actual GPU inference calls (TensorRT, CUDA) release the GIL and scale better.")
        print("The saturation cliff affects the Python code that coordinates GPU operations.")
    
    return 0


if __name__ == "__main__":
    exit(main())
