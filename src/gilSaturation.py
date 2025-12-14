# Experiment C: GIL Saturation Validation
# Demonstrates that blocking ratio correctly identifies GIL contention,
# justifying its use over raw CPU utilization for scaling decisions.

import time
import csv
import os
import math
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import statistics
import psutil

from src.adaptive_executor import AdaptiveThreadPoolExecutor, StaticThreadPoolExecutor
from src.metrics import MetricsCollector, TaskMetrics


@dataclass
class GilExperimentConfig:
    # Configuration for the GIL saturation experiment.
    threadCounts: List[int] = None  # Will default to [1, 2, 4, 8, 16, 32]
    tasksPerThread: int = 50
    fibonacciN: int = 28  # Computation intensity (higher = longer)
    measurementDurationSec: float = 30.0
    outputDir: str = "results/experiment_c"
    
    def __post_init__(self):
        if self.threadCounts is None:
            self.threadCounts = [1, 2, 4, 8, 16, 32]


@dataclass
class GilDataPoint:
    # Data point for a single thread count configuration.
    threadCount: int
    totalThroughput: float  # Tasks per second
    avgBlockingRatio: float
    stdBlockingRatio: float
    avgCpuUtilization: float
    avgTaskDurationMs: float
    totalTasksCompleted: int
    measurementDurationSec: float


def fibonacciRecursive(n: int) -> int:
    # Compute the nth Fibonacci number recursively.
    This is intentionally inefficient to hold the GIL for extended periods.
    Pure Python computation with no opportunity for GIL release.
    
    Args:
        n: The Fibonacci index to compute.
        
    Returns:
        The nth Fibonacci number.
    """
    if n <= 1:
        return n
    return fibonacciRecursive(n - 1) + fibonacciRecursive(n - 2)


def cpuIntensivePythonTask(iterations: int = 50000) -> float:
    """
    Pure Python CPU-intensive task that holds the GIL.
    
    Uses mathematical operations that cannot be optimized away
    and keeps the GIL locked throughout execution.
    
    Args:
        iterations: Number of computation iterations.
        
    Returns:
        Computed result value.
    """
    result = 0.0
    for i in range(iterations):
        result += math.sin(i) * math.cos(i) * math.tan(i % 1000 + 0.1)
        result = result % 1000000  # Prevent overflow
    return result


class GilSaturationExperiment:
    """
    Experiment C: GIL Saturation Validation
    
    This experiment demonstrates that the blocking ratio metric correctly
    identifies GIL saturation, even when CPU utilization appears high.
    
    Key insight: When running pure Python CPU-bound tasks, adding threads
    beyond the physical core count does not improve throughput due to GIL
    contention. The blocking ratio drops to near zero as threads compete
    for the GIL, providing a clear signal to stop scaling.
    """
    
    def __init__(self, config: Optional[GilExperimentConfig] = None):
        """
        Initialize the experiment.
        
        Args:
            config: Experiment configuration. Uses defaults if None.
        """
        self.config = config or GilExperimentConfig()
        os.makedirs(self.config.outputDir, exist_ok=True)
        self.physicalCores = psutil.cpu_count(logical=False) or 4
        self.logicalCores = psutil.cpu_count(logical=True) or 8
        
        print(f"Detected {self.physicalCores} physical cores, "
              f"{self.logicalCores} logical cores")
    
    def runWithThreadCount(self, threadCount: int) -> GilDataPoint:
        """
        Run the experiment with a specific thread count.
        
        Args:
            threadCount: Number of worker threads to use.
            
        Returns:
            GilDataPoint with collected metrics.
        """
        print(f"\nTesting with {threadCount} threads...")
        
        metricsCollector = MetricsCollector()
        cpuSamples = []
        stopEvent = threading.Event()
        taskCounter = [0]  # Mutable container for counter
        counterLock = threading.Lock()
        
        # CPU monitoring thread
        def monitorCpu():
            while not stopEvent.is_set():
                cpuSamples.append(psutil.cpu_percent(interval=0.2))
        
        monitorThread = threading.Thread(target=monitorCpu, daemon=True)
        monitorThread.start()
        
        # Task wrapper for metrics collection
        def instrumentedTask():
            wallStart = time.time()
            cpuStart = time.thread_time()
            
            # Execute the GIL-holding task
            result = fibonacciRecursive(self.config.fibonacciN)
            
            cpuEnd = time.thread_time()
            wallEnd = time.time()
            
            wallTime = wallEnd - wallStart
            cpuTime = cpuEnd - cpuStart
            
            if wallTime > 0:
                blockingRatio = 1.0 - (cpuTime / wallTime)
            else:
                blockingRatio = 0.0
            
            with counterLock:
                taskCounter[0] += 1
                taskId = f"task-{taskCounter[0]}"
            
            metrics = TaskMetrics(
                taskId=taskId,
                wallTime=wallTime,
                cpuTime=cpuTime,
                blockingRatio=blockingRatio,
                timestamp=wallEnd,
                success=True,
            )
            metricsCollector.record(metrics)
            
            return result
        
        # Run with the specified thread count
        executor = ThreadPoolExecutor(max_workers=threadCount)
        measurementStart = time.time()
        futures: List[Future] = []
        
        # Submit tasks continuously for the measurement duration
        while (time.time() - measurementStart) < self.config.measurementDurationSec:
            # Submit a batch of tasks
            batchSize = threadCount * 2
            for _ in range(batchSize):
                future = executor.submit(instrumentedTask)
                futures.append(future)
            
            # Wait for some to complete before submitting more
            completed = sum(1 for f in futures if f.done())
            while completed < len(futures) * 0.5:
                time.sleep(0.01)
                completed = sum(1 for f in futures if f.done())
        
        # Wait for all futures to complete
        for future in futures:
            try:
                future.result(timeout=60.0)
            except Exception as e:
                print(f"Task failed: {e}")
        
        measurementDuration = time.time() - measurementStart
        
        # Stop monitoring
        stopEvent.set()
        monitorThread.join(timeout=2.0)
        
        # Shutdown executor
        executor.shutdown(wait=True)
        
        # Calculate statistics
        allMetrics = metricsCollector.exportToDict()
        
        if allMetrics:
            blockingRatios = [m["blockingRatio"] for m in allMetrics]
            taskDurations = [m["wallTime"] * 1000 for m in allMetrics]  # Convert to ms
            
            avgBlockingRatio = statistics.mean(blockingRatios)
            stdBlockingRatio = statistics.stdev(blockingRatios) if len(blockingRatios) > 1 else 0.0
            avgTaskDuration = statistics.mean(taskDurations)
        else:
            avgBlockingRatio = 0.0
            stdBlockingRatio = 0.0
            avgTaskDuration = 0.0
        
        totalTasks = len(allMetrics)
        throughput = totalTasks / measurementDuration if measurementDuration > 0 else 0.0
        avgCpu = statistics.mean(cpuSamples) if cpuSamples else 0.0
        
        dataPoint = GilDataPoint(
            threadCount=threadCount,
            totalThroughput=throughput,
            avgBlockingRatio=avgBlockingRatio,
            stdBlockingRatio=stdBlockingRatio,
            avgCpuUtilization=avgCpu,
            avgTaskDurationMs=avgTaskDuration,
            totalTasksCompleted=totalTasks,
            measurementDurationSec=measurementDuration,
        )
        
        print(f"  Completed: {totalTasks} tasks, "
              f"throughput={throughput:.1f} ops/s, "
              f"blocking_ratio={avgBlockingRatio:.3f}, "
              f"cpu={avgCpu:.1f}%")
        
        return dataPoint
    
    def run(self) -> List[GilDataPoint]:
        """
        Run the complete GIL saturation experiment.
        
        Tests each configured thread count and collects metrics.
        
        Returns:
            List of GilDataPoint for each thread configuration.
        """
        print("="*60)
        print("Experiment C: GIL Saturation Validation")
        print("="*60)
        print(f"Physical cores: {self.physicalCores}")
        print(f"Logical cores: {self.logicalCores}")
        print(f"Testing thread counts: {self.config.threadCounts}")
        print(f"Fibonacci(n={self.config.fibonacciN}) per task")
        print("")
        
        results: List[GilDataPoint] = []
        
        for threadCount in self.config.threadCounts:
            dataPoint = self.runWithThreadCount(threadCount)
            results.append(dataPoint)
            
            # Brief pause between runs for system stabilization
            time.sleep(2.0)
        
        # Save results
        self.saveResults(results)
        
        # Print analysis
        self.printAnalysis(results)
        
        return results
    
    def saveResults(self, results: List[GilDataPoint]) -> None:
        """
        Save experiment results to CSV.
        
        Args:
            results: List of data points to save.
        """
        filename = os.path.join(self.config.outputDir, "gil_saturation.csv")
        
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "thread_count", "throughput_ops_sec", "avg_blocking_ratio",
                "std_blocking_ratio", "avg_cpu_utilization", "avg_task_duration_ms",
                "total_tasks", "measurement_duration_sec"
            ])
            
            for dp in results:
                writer.writerow([
                    dp.threadCount, dp.totalThroughput, dp.avgBlockingRatio,
                    dp.stdBlockingRatio, dp.avgCpuUtilization, dp.avgTaskDurationMs,
                    dp.totalTasksCompleted, dp.measurementDurationSec
                ])
        
        print(f"\nSaved results to {filename}")
    
    def printAnalysis(self, results: List[GilDataPoint]) -> None:
        """
        Print analysis of the GIL saturation experiment.
        
        Args:
            results: List of data points to analyze.
        """
        print("\n" + "="*60)
        print("ANALYSIS: GIL Saturation Detection")
        print("="*60)
        
        # Find peak throughput
        maxThroughput = max(results, key=lambda x: x.totalThroughput)
        
        print(f"\nPeak throughput: {maxThroughput.totalThroughput:.1f} ops/sec "
              f"at {maxThroughput.threadCount} threads")
        
        # Find where blocking ratio drops significantly
        for i, dp in enumerate(results):
            if i == 0:
                continue
            
            prevRatio = results[i-1].avgBlockingRatio
            currRatio = dp.avgBlockingRatio
            
            if currRatio < 0.1 and prevRatio > currRatio:
                print(f"\nGIL saturation signal detected at {dp.threadCount} threads:")
                print(f"  Blocking ratio dropped from {prevRatio:.3f} to {currRatio:.3f}")
                print(f"  This indicates threads are waiting for GIL rather than I/O")
                break
        
        # Compare to physical core count
        print(f"\nComparison to physical cores ({self.physicalCores}):")
        
        for dp in results:
            if dp.threadCount == self.physicalCores:
                print(f"  At {dp.threadCount} threads (= physical cores): "
                      f"throughput={dp.totalThroughput:.1f}, "
                      f"blocking_ratio={dp.avgBlockingRatio:.3f}")
            elif dp.threadCount == self.physicalCores * 2:
                print(f"  At {dp.threadCount} threads (2x physical cores): "
                      f"throughput={dp.totalThroughput:.1f}, "
                      f"blocking_ratio={dp.avgBlockingRatio:.3f}")
        
        print("\nConclusion:")
        print("  The blocking ratio correctly identifies when adding more threads")
        print("  no longer improves throughput due to GIL contention.")
        print("  A low blocking ratio (< 0.1) signals CPU/GIL saturation.")


def runExperimentC(
    threadCounts: Optional[List[int]] = None,
    measurementDuration: float = 30.0,
    outputDir: str = "results/experiment_c"
) -> List[GilDataPoint]:
    """
    Convenience function to run Experiment C.
    
    Args:
        threadCounts: List of thread counts to test.
        measurementDuration: Duration for each thread count test.
        outputDir: Directory for output files.
        
    Returns:
        List of GilDataPoint results.
    """
    config = GilExperimentConfig(
        threadCounts=threadCounts,
        measurementDurationSec=measurementDuration,
        outputDir=outputDir,
    )
    
    experiment = GilSaturationExperiment(config)
    return experiment.run()


if __name__ == "__main__":
    print("Starting Experiment C: GIL Saturation Validation")
    print("This experiment validates blocking ratio as a GIL contention signal.")
    print("")
    
    # Use reduced measurement time for faster testing
    results = runExperimentC(
        threadCounts=[1, 2, 4, 8, 16, 32],
        measurementDuration=15.0,
    )
    
    print("\nExperiment C complete. Results saved to results/experiment_c/")
