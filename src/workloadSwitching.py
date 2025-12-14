# Experiment A: Square Wave Stress Test
# Validates controller response to abrupt workload phase transitions
# between I/O-bound and CPU-bound operation modes.

import time
import csv
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
import threading
import psutil
import numpy as np

from src.adaptive_executor import AdaptiveThreadPoolExecutor, StaticThreadPoolExecutor, ControllerConfig
from src.workloads import WorkloadGenerator


class WorkloadPhase(Enum):
    """Workload phase identifiers for the square wave experiment."""
    IO_BOUND = "io_bound"
    CPU_BOUND = "cpu_bound"
    MIXED = "mixed"


@dataclass
class ExperimentConfig:
    """Configuration for the square wave experiment."""
    totalDurationSec: float = 180.0
    phaseDurationSec: float = 60.0
    taskSubmissionRatePerSec: float = 50.0
    ioTaskDurationMs: float = 50.0
    cpuTaskIterations: int = 100000
    matrixSize: int = 100
    useNumpyCpu: bool = True  # True = NumPy (releases GIL), False = pure Python
    samplingIntervalSec: float = 0.5
    outputDir: str = "results/experiment_a"


@dataclass
class DataPoint:
    """Single data point collected during the experiment."""
    timestamp: float
    elapsedSec: float
    phase: str
    activeThreads: int
    throughputRps: float
    cpuUtilization: float
    avgBlockingRatio: float
    p50LatencyMs: float
    p99LatencyMs: float
    pendingTasks: int


class SquareWaveExperiment:
    """
    Experiment A: Square Wave Stress Test
    
    This experiment runs three phases:
    1. I/O Phase (0-60s): Tasks that primarily sleep, simulating network waits.
    2. CPU Phase (60-120s): Tasks that perform computation.
    3. Recovery Phase (120-180s): Return to I/O-bound workload.
    
    The purpose is to validate that the adaptive controller:
    - Scales up thread count during I/O phases
    - Scales down during CPU phases to avoid GIL contention
    - Responds quickly to phase transitions without thrashing
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Initialize the experiment.
        
        Args:
            config: Experiment configuration. Uses defaults if None.
        """
        self.config = config or ExperimentConfig()
        self.dataPoints: List[DataPoint] = []
        self.startTime: float = 0.0
        self.stopEvent = threading.Event()
        
        # Ensure output directory exists
        os.makedirs(self.config.outputDir, exist_ok=True)
    
    def getCurrentPhase(self, elapsedSec: float) -> WorkloadPhase:
        """
        Determine the current workload phase based on elapsed time.
        
        Args:
            elapsedSec: Seconds elapsed since experiment start.
            
        Returns:
            Current WorkloadPhase.
        """
        phaseDuration = self.config.phaseDurationSec
        
        if elapsedSec < phaseDuration:
            return WorkloadPhase.IO_BOUND
        elif elapsedSec < 2 * phaseDuration:
            return WorkloadPhase.CPU_BOUND
        else:
            return WorkloadPhase.IO_BOUND  # Recovery phase
    
    def getTaskForPhase(self, phase: WorkloadPhase):
        """
        Get the appropriate task generator for the current phase.
        
        Args:
            phase: Current workload phase.
            
        Returns:
            Callable task generator.
        """
        if phase == WorkloadPhase.IO_BOUND:
            return WorkloadGenerator.ioTask(self.config.ioTaskDurationMs)
        elif phase == WorkloadPhase.CPU_BOUND:
            if self.config.useNumpyCpu:
                return WorkloadGenerator.cpuTaskNumpy(self.config.matrixSize)
            else:
                return WorkloadGenerator.cpuTaskPython(self.config.cpuTaskIterations)
        else:
            return WorkloadGenerator.mixedTask(
                self.config.ioTaskDurationMs / 2,
                self.config.cpuTaskIterations // 2
            )
    
    def runWithExecutor(
        self,
        executor,
        executorName: str
    ) -> List[DataPoint]:
        """
        Run the experiment with a specific executor.
        
        Args:
            executor: The executor to test (adaptive or static).
            executorName: Name for logging and output files.
            
        Returns:
            List of collected data points.
        """
        dataPoints: List[DataPoint] = []
        self.startTime = time.time()
        pendingFutures: List[Future] = []
        
        # Metrics collection thread
        metricsLock = threading.Lock()
        
        print(f"[{executorName}] Starting experiment...")
        print(f"[{executorName}] Duration: {self.config.totalDurationSec}s")
        print(f"[{executorName}] Phase duration: {self.config.phaseDurationSec}s")
        
        # Task submission thread
        def submitTasks():
            interval = 1.0 / self.config.taskSubmissionRatePerSec
            lastSubmitTime = time.time()
            
            while not self.stopEvent.is_set():
                currentTime = time.time()
                elapsed = currentTime - self.startTime
                
                if elapsed >= self.config.totalDurationSec:
                    break
                
                phase = self.getCurrentPhase(elapsed)
                task = self.getTaskForPhase(phase)
                
                try:
                    future = executor.submit(task)
                    with metricsLock:
                        pendingFutures.append(future)
                except Exception as e:
                    print(f"[{executorName}] Task submission error: {e}")
                
                # Rate limiting
                sleepTime = interval - (time.time() - lastSubmitTime)
                if sleepTime > 0:
                    time.sleep(sleepTime)
                lastSubmitTime = time.time()
        
        # Metrics sampling thread
        def sampleMetrics():
            lastSampleTime = time.time()
            completedCount = 0
            
            while not self.stopEvent.is_set():
                currentTime = time.time()
                elapsed = currentTime - self.startTime
                
                if elapsed >= self.config.totalDurationSec:
                    break
                
                # Count completed futures
                with metricsLock:
                    newCompleted = sum(1 for f in pendingFutures if f.done())
                
                # Calculate throughput
                sampleInterval = currentTime - lastSampleTime
                if sampleInterval > 0:
                    throughput = (newCompleted - completedCount) / sampleInterval
                else:
                    throughput = 0.0
                completedCount = newCompleted
                lastSampleTime = currentTime
                
                # Get executor metrics
                metrics = executor.getMetrics()
                
                phase = self.getCurrentPhase(elapsed)
                
                dataPoint = DataPoint(
                    timestamp=currentTime,
                    elapsedSec=elapsed,
                    phase=phase.value,
                    activeThreads=metrics.get("currentThreads", 0),
                    throughputRps=metrics.get("throughput", throughput),
                    cpuUtilization=psutil.cpu_percent(interval=None),
                    avgBlockingRatio=metrics.get("avgBlockingRatio", 0.0),
                    p50LatencyMs=metrics.get("p50Latency", 0.0) * 1000,
                    p99LatencyMs=metrics.get("p99Latency", 0.0) * 1000,
                    pendingTasks=len(pendingFutures) - completedCount,
                )
                
                dataPoints.append(dataPoint)
                
                # Progress reporting
                if int(elapsed) % 10 == 0 and elapsed > 0:
                    print(f"[{executorName}] t={elapsed:.0f}s, phase={phase.value}, "
                          f"threads={dataPoint.activeThreads}, "
                          f"throughput={dataPoint.throughputRps:.1f}")
                
                time.sleep(self.config.samplingIntervalSec)
        
        # Start threads
        submitThread = threading.Thread(target=submitTasks, name="TaskSubmitter")
        metricsThread = threading.Thread(target=sampleMetrics, name="MetricsSampler")
        
        submitThread.start()
        metricsThread.start()
        
        # Wait for experiment duration
        time.sleep(self.config.totalDurationSec + 1.0)
        self.stopEvent.set()
        
        submitThread.join(timeout=5.0)
        metricsThread.join(timeout=5.0)
        
        # Final metrics
        finalMetrics = executor.getMetrics()
        print(f"[{executorName}] Experiment complete.")
        print(f"[{executorName}] Total tasks: {finalMetrics.get('totalTasks', 0)}")
        print(f"[{executorName}] Scale-up events: {finalMetrics.get('scaleUpCount', 'N/A')}")
        print(f"[{executorName}] Scale-down events: {finalMetrics.get('scaleDownCount', 'N/A')}")
        
        self.stopEvent.clear()
        
        return dataPoints
    
    def run(
        self,
        minWorkers: int = 4,
        maxWorkers: int = 64,
        staticSmallPool: int = 4,
        staticLargePool: int = 50,
    ) -> Dict[str, List[DataPoint]]:
        """
        Run the complete experiment with all executor configurations.
        
        Args:
            minWorkers: Minimum workers for adaptive executor.
            maxWorkers: Maximum workers for adaptive executor.
            staticSmallPool: Worker count for small static baseline.
            staticLargePool: Worker count for large static baseline.
            
        Returns:
            Dictionary mapping executor name to collected data points.
        """
        results = {}
        
        # Run with adaptive executor
        print("\n" + "="*60)
        print("Running with Adaptive Executor")
        print("="*60)
        
        controllerConfig = ControllerConfig(
            monitorIntervalSec=0.5,
            betaHighThreshold=0.7,
            betaLowThreshold=0.3,
            scaleUpStep=2,
            scaleDownStep=1,
        )
        
        with AdaptiveThreadPoolExecutor(
            minWorkers=minWorkers,
            maxWorkers=maxWorkers,
            config=controllerConfig,
        ) as executor:
            results["adaptive"] = self.runWithExecutor(executor, "Adaptive")
        
        # Reset and run with static small pool
        print("\n" + "="*60)
        print(f"Running with Static Executor (workers={staticSmallPool})")
        print("="*60)
        
        with StaticThreadPoolExecutor(workers=staticSmallPool) as executor:
            results["static_small"] = self.runWithExecutor(executor, "Static-Small")
        
        # Reset and run with static large pool
        print("\n" + "="*60)
        print(f"Running with Static Executor (workers={staticLargePool})")
        print("="*60)
        
        with StaticThreadPoolExecutor(workers=staticLargePool) as executor:
            results["static_large"] = self.runWithExecutor(executor, "Static-Large")
        
        # Save results
        self.saveResults(results)
        
        return results
    
    def saveResults(self, results: Dict[str, List[DataPoint]]) -> None:
        """
        Save experiment results to CSV files.
        
        Args:
            results: Dictionary of results from each executor.
        """
        for executorName, dataPoints in results.items():
            filename = os.path.join(
                self.config.outputDir,
                f"square_wave_{executorName}.csv"
            )
            
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "elapsed_sec", "phase", "active_threads",
                    "throughput_rps", "cpu_utilization", "avg_blocking_ratio",
                    "p50_latency_ms", "p99_latency_ms", "pending_tasks"
                ])
                
                for dp in dataPoints:
                    writer.writerow([
                        dp.timestamp, dp.elapsedSec, dp.phase, dp.activeThreads,
                        dp.throughputRps, dp.cpuUtilization, dp.avgBlockingRatio,
                        dp.p50LatencyMs, dp.p99LatencyMs, dp.pendingTasks
                    ])
            
            print(f"Saved results to {filename}")


def runExperimentA(
    duration: float = 180.0,
    phaseDuration: float = 60.0,
    outputDir: str = "results/experiment_a"
) -> Dict[str, List[DataPoint]]:
    """
    Convenience function to run Experiment A with default settings.
    
    Args:
        duration: Total experiment duration in seconds.
        phaseDuration: Duration of each phase in seconds.
        outputDir: Directory for output files.
        
    Returns:
        Dictionary of results from each executor configuration.
    """
    config = ExperimentConfig(
        totalDurationSec=duration,
        phaseDurationSec=phaseDuration,
        outputDir=outputDir,
    )
    
    experiment = SquareWaveExperiment(config)
    return experiment.run()


if __name__ == "__main__":
    # Run experiment with reduced duration for testing
    print("Starting Experiment A: Square Wave Stress Test")
    print("This experiment tests controller response to workload phase changes.")
    print("")
    
    results = runExperimentA(
        duration=90.0,  # Reduced for faster testing
        phaseDuration=30.0,
    )
    
    print("\nExperiment A complete. Results saved to results/experiment_a/")
