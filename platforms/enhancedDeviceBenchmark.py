#!/usr/bin/env python3
"""
Enhanced Real Device Benchmark with Resource Metrics.

Designed to run on actual Raspberry Pi 4, Raspberry Pi Zero, or Jetson Nano.
Collects comprehensive resource metrics including memory usage and power
consumption estimates.

This addresses reviewer concern #4: real device experiments and resource metrics.
"""

import time
import random
import statistics
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from threading import Thread, Event
import platform

# Set seed for reproducibility.
SEED = 17
random.seed(SEED)

# Experiment parameters.
CPU_ITERATIONS = 1000
IO_SLEEP_MS = 0.1
TASK_COUNT = 10000
RUNS_PER_CONFIG = 5  # More runs for statistical confidence.
THREAD_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Results directory.
RESULTS_DIR = "results"


@dataclass
class ResourceMetrics:
    """Container for resource usage metrics during benchmark."""
    memoryUsedMb: float
    memoryPeakMb: float
    cpuPercentAvg: float
    cpuPercentPeak: float
    cpuTempCelsius: Optional[float] = None
    powerEstimateWatts: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Container for complete benchmark results with statistics."""
    platform: str
    threads: int
    runs: int
    tpsMean: float
    tpsStd: float
    tpsConfInterval: Tuple[float, float]
    avgLatMsMean: float
    avgLatMsStd: float
    p99LatMsMean: float
    p99LatMsStd: float
    avgBeta: float
    resources: ResourceMetrics


def getPlatformInfo() -> Dict:
    """Detect and return platform information."""
    info = {
        "name": "Unknown",
        "model": "Unknown",
        "cores": os.cpu_count() or 1,
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }
    
    # Try to detect Raspberry Pi model.
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip()
            info["model"] = model
            if "Pi 4" in model:
                info["name"] = "RaspberryPi4"
            elif "Pi Zero" in model:
                info["name"] = "RaspberryPiZero"
            elif "Pi 3" in model:
                info["name"] = "RaspberryPi3"
            else:
                info["name"] = "RaspberryPi"
    except FileNotFoundError:
        pass
    
    # Try to detect Jetson.
    try:
        with open("/etc/nv_tegra_release", "r") as f:
            info["name"] = "JetsonNano"
            info["model"] = "NVIDIA Jetson Nano"
    except FileNotFoundError:
        pass
    
    # Detect memory.
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    info["memory_mb"] = int(line.split()[1]) // 1024
                    break
    except:
        info["memory_mb"] = 0
    
    return info


def getMemoryUsageMb() -> float:
    """Get current process memory usage in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status", "r") as f:
            for line in f:
                if "VmRSS" in line:
                    return int(line.split()[1]) / 1024
    except:
        pass
    
    # Fallback using psutil if available.
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def getCpuTemperature() -> Optional[float]:
    """Get CPU temperature in Celsius (Raspberry Pi specific)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read().strip()) / 1000.0
    except:
        return None


def estimatePowerConsumption(cpuPercent: float, platform: str) -> Optional[float]:
    """
    Estimate power consumption based on CPU utilization.
    
    Power models based on published measurements:
    - Raspberry Pi 4 idle: ~2.7W, load: ~6.4W
    - Raspberry Pi Zero: ~0.4W idle, ~1.0W load
    - Jetson Nano: ~2.0W idle, ~10W load (with GPU)
    
    Returns estimated power in Watts.
    """
    power_models = {
        "RaspberryPi4": (2.7, 6.4),
        "RaspberryPi3": (1.5, 5.0),
        "RaspberryPiZero": (0.4, 1.0),
        "JetsonNano": (2.0, 10.0),
    }
    
    if platform not in power_models:
        return None
    
    idle, load = power_models[platform]
    return idle + (load - idle) * (cpuPercent / 100.0)


class ResourceMonitor:
    """Background thread for monitoring resource usage during benchmarks."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.stopEvent = Event()
        self.memoryReadings = []
        self.cpuReadings = []
        self.tempReadings = []
        self.monitorThread = None
    
    def start(self):
        """Start the resource monitor."""
        self.stopEvent.clear()
        self.memoryReadings = []
        self.cpuReadings = []
        self.tempReadings = []
        self.monitorThread = Thread(target=self._monitorLoop, daemon=True)
        self.monitorThread.start()
    
    def stop(self) -> ResourceMetrics:
        """Stop the monitor and return collected metrics."""
        self.stopEvent.set()
        if self.monitorThread:
            self.monitorThread.join(timeout=1.0)
        
        memAvg = statistics.mean(self.memoryReadings) if self.memoryReadings else 0
        memPeak = max(self.memoryReadings) if self.memoryReadings else 0
        cpuAvg = statistics.mean(self.cpuReadings) if self.cpuReadings else 0
        cpuPeak = max(self.cpuReadings) if self.cpuReadings else 0
        tempAvg = statistics.mean(self.tempReadings) if self.tempReadings else None
        
        return ResourceMetrics(
            memoryUsedMb=memAvg,
            memoryPeakMb=memPeak,
            cpuPercentAvg=cpuAvg,
            cpuPercentPeak=cpuPeak,
            cpuTempCelsius=tempAvg,
        )
    
    def _monitorLoop(self):
        """Main monitoring loop."""
        while not self.stopEvent.is_set():
            self.memoryReadings.append(getMemoryUsageMb())
            
            # Simple CPU usage estimation.
            try:
                with open("/proc/stat", "r") as f:
                    line = f.readline()
                    parts = line.split()
                    idle = int(parts[4])
                    total = sum(int(p) for p in parts[1:])
                    
                time.sleep(0.05)
                
                with open("/proc/stat", "r") as f:
                    line = f.readline()
                    parts = line.split()
                    idle2 = int(parts[4])
                    total2 = sum(int(p) for p in parts[1:])
                
                idleDelta = idle2 - idle
                totalDelta = total2 - total
                cpuPercent = 100.0 * (1.0 - idleDelta / totalDelta) if totalDelta > 0 else 0
                self.cpuReadings.append(cpuPercent)
            except:
                pass
            
            temp = getCpuTemperature()
            if temp is not None:
                self.tempReadings.append(temp)
            
            time.sleep(self.interval)


def mixedTask() -> Tuple[float, float]:
    """Mixed CPU+IO task returning latency and blocking ratio."""
    wallStart = time.perf_counter()
    cpuStart = time.thread_time()
    
    # CPU phase.
    x = 0
    for _ in range(CPU_ITERATIONS):
        x += 1
    
    # IO phase.
    time.sleep(IO_SLEEP_MS / 1000.0)
    
    cpuEnd = time.thread_time()
    wallEnd = time.perf_counter()
    
    latency = wallEnd - wallStart
    cpuTime = cpuEnd - cpuStart
    beta = 1.0 - min(1.0, cpuTime / latency) if latency > 0 else 0.0
    
    return latency, beta


def runSingleBenchmark(numThreads: int, taskCount: int) -> Dict:
    """Run a single benchmark iteration."""
    latencies = []
    betas = []
    
    startTime = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=numThreads) as executor:
        futures = [executor.submit(mixedTask) for _ in range(taskCount)]
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
        "p50LatMs": latenciesMs[len(latenciesMs) // 2],
        "p95LatMs": latenciesMs[int(len(latenciesMs) * 0.95)],
        "p99LatMs": latenciesMs[int(len(latenciesMs) * 0.99)],
        "avgBeta": statistics.mean(betas),
    }


def computeConfidenceInterval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval using t-distribution approximation."""
    n = len(data)
    if n < 2:
        return (data[0], data[0]) if data else (0.0, 0.0)
    
    mean = statistics.mean(data)
    std = statistics.stdev(data)
    se = std / (n ** 0.5)
    
    # t-value for 95% CI with n-1 degrees of freedom (approximation).
    t_values = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    t = t_values.get(n, 1.96)  # Default to z-score for large n.
    
    margin = t * se
    return (mean - margin, mean + margin)


def runFullBenchmark(platformInfo: Dict) -> List[BenchmarkResult]:
    """Run complete benchmark across all thread counts."""
    results = []
    
    print()
    print("Running benchmark with statistical analysis...")
    print(f"{'Threads':<8} | {'TPS Mean':<12} | {'95% CI':<20} | {'P99 Lat (ms)':<12} | {'Beta':<6}")
    print("-" * 75)
    
    peakTps = 0
    peakThreads = 0
    
    for numThreads in THREAD_COUNTS:
        monitor = ResourceMonitor()
        runResults = []
        
        monitor.start()
        
        for run in range(RUNS_PER_CONFIG):
            result = runSingleBenchmark(numThreads, TASK_COUNT)
            runResults.append(result)
        
        resources = monitor.stop()
        
        # Compute statistics.
        tpsList = [r["tps"] for r in runResults]
        tpsMean = statistics.mean(tpsList)
        tpsStd = statistics.stdev(tpsList) if len(tpsList) > 1 else 0
        tpsCI = computeConfidenceInterval(tpsList)
        
        avgLatList = [r["avgLatMs"] for r in runResults]
        p99LatList = [r["p99LatMs"] for r in runResults]
        betaList = [r["avgBeta"] for r in runResults]
        
        resources.powerEstimateWatts = estimatePowerConsumption(
            resources.cpuPercentAvg, platformInfo["name"]
        )
        
        result = BenchmarkResult(
            platform=platformInfo["name"],
            threads=numThreads,
            runs=RUNS_PER_CONFIG,
            tpsMean=tpsMean,
            tpsStd=tpsStd,
            tpsConfInterval=tpsCI,
            avgLatMsMean=statistics.mean(avgLatList),
            avgLatMsStd=statistics.stdev(avgLatList) if len(avgLatList) > 1 else 0,
            p99LatMsMean=statistics.mean(p99LatList),
            p99LatMsStd=statistics.stdev(p99LatList) if len(p99LatList) > 1 else 0,
            avgBeta=statistics.mean(betaList),
            resources=resources,
        )
        results.append(result)
        
        if tpsMean > peakTps:
            peakTps = tpsMean
            peakThreads = numThreads
        
        # Format CI for display.
        ciStr = f"[{tpsCI[0]:,.0f}, {tpsCI[1]:,.0f}]"
        print(f"{numThreads:<8} | {tpsMean:<12,.0f} | {ciStr:<20} | {result.p99LatMsMean:<12.2f} | {result.avgBeta:<6.2f}")
    
    return results


def saveResults(results: List[BenchmarkResult], platformInfo: Dict):
    """Save benchmark results to CSV files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    platformName = platformInfo["name"].lower()
    
    # Save main results.
    mainPath = os.path.join(RESULTS_DIR, f"{platformName}_benchmark.csv")
    with open(mainPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "platform", "threads", "runs", "tps_mean", "tps_std", "tps_ci_low", "tps_ci_high",
            "avg_lat_mean", "avg_lat_std", "p99_lat_mean", "p99_lat_std", "avg_beta",
            "memory_used_mb", "memory_peak_mb", "cpu_avg_pct", "cpu_peak_pct",
            "temp_celsius", "power_estimate_watts"
        ])
        for r in results:
            writer.writerow([
                r.platform, r.threads, r.runs, r.tpsMean, r.tpsStd,
                r.tpsConfInterval[0], r.tpsConfInterval[1],
                r.avgLatMsMean, r.avgLatMsStd, r.p99LatMsMean, r.p99LatMsStd, r.avgBeta,
                r.resources.memoryUsedMb, r.resources.memoryPeakMb,
                r.resources.cpuPercentAvg, r.resources.cpuPercentPeak,
                r.resources.cpuTempCelsius, r.resources.powerEstimateWatts
            ])
    
    print(f"\nResults saved to {mainPath}")


def printSummary(results: List[BenchmarkResult], platformInfo: Dict):
    """Print summary analysis of results."""
    print()
    print("=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)
    print()
    print(f"Platform: {platformInfo['model']}")
    print(f"Cores: {platformInfo['cores']}")
    print(f"Memory: {platformInfo.get('memory_mb', 'Unknown')} MB")
    print()
    
    # Find peak and cliff.
    peakResult = max(results, key=lambda r: r.tpsMean)
    finalResult = results[-1]
    cliffSeverity = ((peakResult.tpsMean - finalResult.tpsMean) / peakResult.tpsMean) * 100
    
    print("Performance Analysis:")
    print(f"  Peak Throughput: {peakResult.tpsMean:,.0f} TPS at {peakResult.threads} threads")
    print(f"  Peak 95% CI: [{peakResult.tpsConfInterval[0]:,.0f}, {peakResult.tpsConfInterval[1]:,.0f}]")
    print(f"  Final Throughput: {finalResult.tpsMean:,.0f} TPS at {finalResult.threads} threads")
    print(f"  Cliff Severity: {cliffSeverity:.1f}% degradation")
    print()
    
    print("Resource Metrics at Peak:")
    print(f"  Memory Usage: {peakResult.resources.memoryUsedMb:.1f} MB (peak: {peakResult.resources.memoryPeakMb:.1f} MB)")
    print(f"  CPU Utilization: {peakResult.resources.cpuPercentAvg:.1f}% avg, {peakResult.resources.cpuPercentPeak:.1f}% peak")
    if peakResult.resources.cpuTempCelsius:
        print(f"  CPU Temperature: {peakResult.resources.cpuTempCelsius:.1f}°C")
    if peakResult.resources.powerEstimateWatts:
        print(f"  Estimated Power: {peakResult.resources.powerEstimateWatts:.2f} W")
    print()
    
    print("Resource Metrics at Cliff (overprovisioned):")
    print(f"  Memory Usage: {finalResult.resources.memoryUsedMb:.1f} MB (peak: {finalResult.resources.memoryPeakMb:.1f} MB)")
    print(f"  CPU Utilization: {finalResult.resources.cpuPercentAvg:.1f}% avg, {finalResult.resources.cpuPercentPeak:.1f}% peak")
    if finalResult.resources.powerEstimateWatts:
        print(f"  Estimated Power: {finalResult.resources.powerEstimateWatts:.2f} W")


def main():
    """Main entry point for enhanced real device benchmark."""
    print()
    print("#" * 75)
    print("# ENHANCED REAL DEVICE BENCHMARK WITH RESOURCE METRICS")
    print("#" * 75)
    print()
    
    # Detect platform.
    platformInfo = getPlatformInfo()
    print(f"Platform: {platformInfo['name']}")
    print(f"Model: {platformInfo['model']}")
    print(f"Architecture: {platformInfo['architecture']}")
    print(f"Cores: {platformInfo['cores']}")
    print(f"Memory: {platformInfo.get('memory_mb', 'Unknown')} MB")
    print(f"Python: {platformInfo['python_version']}")
    print()
    print("Benchmark Configuration:")
    print(f"  CPU Iterations: {CPU_ITERATIONS}")
    print(f"  IO Sleep: {IO_SLEEP_MS} ms")
    print(f"  Tasks per config: {TASK_COUNT}")
    print(f"  Runs per config: {RUNS_PER_CONFIG}")
    print(f"  Thread counts: {THREAD_COUNTS}")
    
    # Run benchmark.
    results = runFullBenchmark(platformInfo)
    
    # Save results.
    saveResults(results, platformInfo)
    
    # Print summary.
    printSummary(results, platformInfo)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
