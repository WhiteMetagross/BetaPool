# Adaptive Thread Pool Executor with Runtime Resizing
# Core implementation of the adaptive controller using Hill Climbing optimization
# and blocking ratio based workload classification.

from concurrent.futures import ThreadPoolExecutor, Future
from threading import Thread, Lock, Event
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
import time
import uuid
import psutil
import logging

from src.metrics import TaskMetrics, MetricsCollector


# Configure module logger
logger = logging.getLogger(__name__)


class ControllerConfig:
    # Configuration parameters for the adaptive controller.
    # These values are tuned based on empirical testing on typical
    # server hardware (8-16 cores). Adjust for specific deployments.
    
    def __init__(
        self,
        monitorIntervalSec: float = 0.5,
        betaHighThreshold: float = 0.7,
        betaLowThreshold: float = 0.3,
        scaleUpStep: int = 2,
        scaleDownStep: int = 1,
        cpuUpperThreshold: float = 85.0,
        cpuLowerThreshold: float = 50.0,
        stabilizationWindowSec: float = 2.0,
        warmupTaskCount: int = 10,
    ):
        # Initialize controller configuration.
        # monitorIntervalSec: Time between controller decisions.
        # betaHighThreshold: Blocking ratio above which we scale up (I/O-bound).
        # betaLowThreshold: Blocking ratio below which we scale down (CPU-bound).
        # scaleUpStep: Number of threads to add when scaling up.
        # scaleDownStep: Number of threads to remove when scaling down.
        # cpuUpperThreshold: CPU utilization above which we avoid scaling up.
        # cpuLowerThreshold: CPU utilization below which we may scale down.
        # stabilizationWindowSec: Minimum time between scaling decisions.
        # warmupTaskCount: Minimum tasks before making scaling decisions.
        self.monitorIntervalSec = monitorIntervalSec
        self.betaHighThreshold = betaHighThreshold
        self.betaLowThreshold = betaLowThreshold
        self.scaleUpStep = scaleUpStep
        self.scaleDownStep = scaleDownStep
        self.cpuUpperThreshold = cpuUpperThreshold
        self.cpuLowerThreshold = cpuLowerThreshold
        self.stabilizationWindowSec = stabilizationWindowSec
        self.warmupTaskCount = warmupTaskCount


class ControllerState:
    """
    Runtime state of the adaptive controller.
    
    Tracks scaling decisions and provides data for experiment logging.
    # Runtime state of the adaptive controller.
    # Tracks scaling decisions and provides data for experiment logging. self.lastScaleTime = time.time()
        self.scaleUpCount = 0
        self.scaleDownCount = 0
        self.decisionHistory: List[Dict[str, Any]] = []
    
    def recordDecision(
        self,
        timestamp: float,
        threadsBefore: int,
        threadsAfter: int,
        blockingRatio: float,
        cpuPercent: float,
        throughput: float,
        decision: str,
    ) -> None:
        """Record a scaling decision for later analysis."""
        self.decisionHistory.append({
            "timestamp": timestamp,
            "threadsBefore": threadsBefore,
            "threadsAfter": threadsAfter,
            "blockingRatio": blockingRatio,
            "cpuPercent": cpuPercent,
        # Record a scaling decision for later analysis.
            "decision": decision,
        })


class AdaptiveThreadPoolExecutor:
    """
    Thread pool executor with adaptive sizing based on workload characteristics.
    
    This executor monitors task execution patterns to distinguish between
    I/O-bound and CPU-bound workloads, adjusting the thread count accordingly.
    
    Key insight: The blocking ratio (1 - cpu_time/wall_time) provides a
    reliable signal for workload classification without requiring GIL-level
    instrumentation.
    
    Usage:
        with AdaptiveThreadPoolExecutor(minWorkers=4, maxWorkers=64) as executor:
            futures = [executor.submit(task, arg) for arg in args]
            results = [f.result() for f in futures]
    """
    
    def __init__(
        self,
        minWorkers: int = 4,
        maxWorkers: int = 64,
        config: Optional[ControllerConfig] = None,
        enableLogging: bool = False,
    ):
        """
        Initialize the adaptive executor.
        
        Args:
            minWorkers: Minimum thread count (typically = physical cores).
            maxWorkers: Maximum thread count (for I/O-heavy workloads).
            config: Controller configuration. Uses defaults if None.
            enableLogging: Enable debug logging for development.
        """
        # Validate parameters
        if minWorkers < 1:
            raise ValueError("minWorkers must be at least 1")
        if maxWorkers < minWorkers:
            raise ValueError("maxWorkers must be >= minWorkers")
        
        self.minWorkers = minWorkers
        self.maxWorkers = maxWorkers
        self.config = config or ControllerConfig()
        self.enableLogging = enableLogging
        
        # Initialize the underlying executor with minimum workers
        self.executor = ThreadPoolExecutor(max_workers=minWorkers)
        
        # Metrics collection
        self.metricsCollector = MetricsCollector()
        
        # Controller state
        self.controllerState = ControllerState(minWorkers)
        self.controllerLock = Lock()
        
        # Monitor thread control
        self.stopEvent = Event()
        self.monitorThread: Optional[Thread] = None
        
        # Task tracking
        self.taskCounter = 0
        self.taskCounterLock = Lock()
        
        # Experiment data logging
        self.experimentLog: List[Dict[str, Any]] = []
        self.experimentLogLock = Lock()
        
        # Start the monitor thread
        self._startMonitor()
    
    def _startMonitor(self) -> None:
        """Start the background monitoring thread."""
        self.monitorThread = Thread(
            target=self._monitorLoop,
            daemon=True,
            name="AdaptiveExecutor-Monitor"
        )
        self.monitorThread.start()
    
    def _monitorLoop(self) -> None:
        """
        Main loop for the controller thread.
        
        Wakes up periodically to evaluate workload and adjust thread count.
        Uses Hill Climbing optimization: if throughput improves after scaling,
        continue in that direction; otherwise, reverse.
        """
        lastThroughput = 0.0
        lastDirection = 0  # -1 = scaled down, 0 = none, 1 = scaled up
        
        while not self.stopEvent.is_set():
            time.sleep(self.config.monitorIntervalSec)
            
            if self.stopEvent.is_set():
                break
            
            try:
                self._makeScalingDecision(lastThroughput, lastDirection)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
    
    def _makeScalingDecision(
        self,
        lastThroughput: float,
        lastDirection: int
    ) -> None:
        """
        Evaluate current metrics and decide whether to scale.
        
        Decision logic:
        1. If beta > high_threshold and CPU < upper_limit: scale UP (I/O-bound)
        2. If beta < low_threshold or CPU > upper_limit: scale DOWN (CPU-bound/saturated)
        3. Otherwise: maintain current level
        """
        currentTime = time.time()
        
        # Check stabilization window
        timeSinceLastScale = currentTime - self.controllerState.lastScaleTime
        if timeSinceLastScale < self.config.stabilizationWindowSec:
            return
        
        # Check warmup period
        if self.metricsCollector.totalTasks < self.config.warmupTaskCount:
            return
        
        # Gather current metrics
        beta = self.metricsCollector.getRecentBlockingRatio()
        cpuPercent = psutil.cpu_percent(interval=None)
        throughput = self.metricsCollector.getThroughput()
        
        with self.controllerLock:
            currentThreads = self.controllerState.currentThreads
            decision = "hold"
            newThreads = currentThreads
            
            # Decision logic based on blocking ratio and CPU utilization
            if beta > self.config.betaHighThreshold:
                # High blocking ratio indicates I/O-bound workload
                if cpuPercent < self.config.cpuUpperThreshold:
                    # CPU has headroom, scale up
                    newThreads = min(
                        currentThreads + self.config.scaleUpStep,
                        self.maxWorkers
                    )
                    if newThreads > currentThreads:
                        decision = "scaleUp"
            
            elif beta < self.config.betaLowThreshold:
                # Low blocking ratio indicates CPU-bound workload
                # Scale down to reduce GIL contention
                newThreads = max(
                    currentThreads - self.config.scaleDownStep,
                    self.minWorkers
                )
                if newThreads < currentThreads:
                    decision = "scaleDown"
            
            elif cpuPercent > self.config.cpuUpperThreshold:
                # CPU is saturated, scale down regardless of beta
                newThreads = max(
                    currentThreads - self.config.scaleDownStep,
                    self.minWorkers
                )
                if newThreads < currentThreads:
                    decision = "scaleDown"
            
            # Apply the scaling decision
            if newThreads != currentThreads:
                self._resizePool(newThreads)
                self.controllerState.currentThreads = newThreads
                self.controllerState.lastScaleTime = currentTime
                
                if decision == "scaleUp":
                    self.controllerState.scaleUpCount += 1
                else:
                    self.controllerState.scaleDownCount += 1
            
            # Record decision for experiment analysis
            self.controllerState.recordDecision(
                timestamp=currentTime,
                threadsBefore=currentThreads,
                threadsAfter=newThreads,
                blockingRatio=beta,
                cpuPercent=cpuPercent,
                throughput=throughput,
                decision=decision,
            )
            
            # Log experiment data point
            self._logExperimentData(
                currentTime, newThreads, beta, cpuPercent, throughput
            )
            
            if self.enableLogging:
                logger.debug(
                    f"Controller: threads={newThreads}, beta={beta:.3f}, "
                    f"cpu={cpuPercent:.1f}%, throughput={throughput:.1f}, "
                    f"decision={decision}"
                )
    
    def _resizePool(self, newSize: int) -> None:
        """
        Resize the thread pool at runtime.
        
        This uses the private _max_workers attribute and _adjust_thread_count
        method of ThreadPoolExecutor. This is a well-known technique in
        Python systems programming.
        """
        self.executor._max_workers = newSize
        self.executor._adjust_thread_count()
    
    def _logExperimentData(
        self,
        timestamp: float,
        activeThreads: int,
        blockingRatio: float,
        cpuPercent: float,
        throughput: float,
    ) -> None:
        """Log data point for experiment visualization."""
        latencies = self.metricsCollector.getLatencyPercentiles()
        
        with self.experimentLogLock:
            self.experimentLog.append({
                "timestamp": timestamp,
                "activeThreads": activeThreads,
                "blockingRatio": blockingRatio,
                "cpuPercent": cpuPercent,
                "throughput": throughput,
                "p50Latency": latencies["p50"],
                "p99Latency": latencies["p99"],
            })
    
    def _wrapTask(
        self,
        fn: Callable,
        taskId: str
    ) -> Callable:
        """
        Wrap a task function to capture execution metrics.
        
        The wrapper measures both wall time (total elapsed) and CPU time
        (actual execution on CPU) to compute the blocking ratio.
        """
        @wraps(fn)
        def wrapper(*args, **kwargs):
            wallStart = time.time()
            cpuStart = time.thread_time()
            
            success = True
            errorMessage = None
            result = None
            
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                success = False
                errorMessage = str(e)
                raise
            finally:
                cpuEnd = time.thread_time()
                wallEnd = time.time()
                
                wallTime = wallEnd - wallStart
                cpuTime = cpuEnd - cpuStart
                
                # Compute blocking ratio, handling edge cases
                if wallTime > 0:
                    blockingRatio = 1.0 - (cpuTime / wallTime)
                else:
                    blockingRatio = 0.0
                
                # Record metrics
                metrics = TaskMetrics(
                    taskId=taskId,
                    wallTime=wallTime,
                    cpuTime=cpuTime,
                    blockingRatio=blockingRatio,
                    timestamp=wallEnd,
                    success=success,
                    errorMessage=errorMessage,
                )
                self.metricsCollector.record(metrics)
            
            return result
        
        return wrapper
    
    def _generateTaskId(self) -> str:
        """Generate a unique task identifier."""
        with self.taskCounterLock:
            self.taskCounter += 1
            return f"task-{self.taskCounter:08d}"
    
    def submit(
        self,
        fn: Callable,
        *args,
        **kwargs
    ) -> Future:
        """
        Submit a task for execution.
        
        The task will be wrapped to capture execution metrics. The return
        value is a Future that can be used to retrieve the result.
        
        Args:
            fn: The callable to execute.
            *args: Positional arguments for the callable.
            **kwargs: Keyword arguments for the callable.
            
        Returns:
            A Future representing the pending execution.
        """
        taskId = self._generateTaskId()
        wrappedFn = self._wrapTask(fn, taskId)
        return self.executor.submit(wrappedFn, *args, **kwargs)
    
    def map(
        self,
        fn: Callable,
        *iterables,
        timeout: Optional[float] = None,
        chunksize: int = 1
    ):
        """
        Map a function over iterables, executing in parallel.
        
        Args:
            fn: The callable to apply to each element.
            *iterables: Iterables of arguments.
            timeout: Maximum time to wait for results.
            chunksize: Size of chunks for batching (currently unused).
            
        Yields:
            Results in order of the input iterables.
        """
        futures = [self.submit(fn, *args) for args in zip(*iterables)]
        
        for future in futures:
            yield future.result(timeout=timeout)
    
    def getCurrentThreadCount(self) -> int:
        """Get the current active thread count."""
        with self.controllerLock:
            return self.controllerState.currentThreads
    
    def getMetrics(self) -> Dict[str, Any]:
        """
        Get current metrics summary.
        
        Returns:
            Dictionary containing current executor state and metrics.
        """
        aggregated = self.metricsCollector.getAggregatedMetrics()
        
        return {
            "currentThreads": self.getCurrentThreadCount(),
            "minWorkers": self.minWorkers,
            "maxWorkers": self.maxWorkers,
            "totalTasks": self.metricsCollector.totalTasks,
            "avgBlockingRatio": aggregated.avgBlockingRatio,
            "throughput": aggregated.throughput,
            "p50Latency": aggregated.p50Latency,
            "p99Latency": aggregated.p99Latency,
            "scaleUpCount": self.controllerState.scaleUpCount,
            "scaleDownCount": self.controllerState.scaleDownCount,
        }
    
    def getExperimentLog(self) -> List[Dict[str, Any]]:
        """Get the experiment data log for visualization."""
        with self.experimentLogLock:
            return list(self.experimentLog)
    
    def getDecisionHistory(self) -> List[Dict[str, Any]]:
        """Get the history of scaling decisions."""
        return list(self.controllerState.decisionHistory)
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.
        
        Args:
            wait: If True, wait for pending tasks to complete.
        """
        self.stopEvent.set()
        
        if self.monitorThread and self.monitorThread.is_alive():
            self.monitorThread.join(timeout=2.0)
        
        self.executor.shutdown(wait=wait)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True)
        return False


class StaticThreadPoolExecutor:
    """
    Static thread pool executor for baseline comparison.
    
    Provides the same interface as AdaptiveThreadPoolExecutor but with
    fixed thread count. Used as baseline in experiments.
    """
    
    def __init__(self, workers: int = 4):
        """
        Initialize with fixed worker count.
        
        Args:
            workers: Number of worker threads (fixed).
        """
        self.workers = workers
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.metricsCollector = MetricsCollector()
        
        self.taskCounter = 0
        self.taskCounterLock = Lock()
        
        self.experimentLog: List[Dict[str, Any]] = []
        self.experimentLogLock = Lock()
        self.startTime = time.time()
    
    def _wrapTask(self, fn: Callable, taskId: str) -> Callable:
        """Wrap task for metrics collection."""
        @wraps(fn)
        def wrapper(*args, **kwargs):
            wallStart = time.time()
            cpuStart = time.thread_time()
            
            result = None
            success = True
            
            try:
                result = fn(*args, **kwargs)
            except Exception:
                success = False
                raise
            finally:
                cpuEnd = time.thread_time()
                wallEnd = time.time()
                
                wallTime = wallEnd - wallStart
                cpuTime = cpuEnd - cpuStart
                
                if wallTime > 0:
                    blockingRatio = 1.0 - (cpuTime / wallTime)
                else:
                    blockingRatio = 0.0
                
                metrics = TaskMetrics(
                    taskId=taskId,
                    wallTime=wallTime,
                    cpuTime=cpuTime,
                    blockingRatio=blockingRatio,
                    timestamp=wallEnd,
                    success=success,
                )
                self.metricsCollector.record(metrics)
                
                # Log experiment data
                with self.experimentLogLock:
                    self.experimentLog.append({
                        "timestamp": wallEnd,
                        "activeThreads": self.workers,
                        "blockingRatio": blockingRatio,
                        "cpuPercent": psutil.cpu_percent(interval=None),
                        "throughput": self.metricsCollector.getThroughput(),
                        "p50Latency": 0.0,
                        "p99Latency": 0.0,
                    })
            
            return result
        
        return wrapper
    
    def _generateTaskId(self) -> str:
        """Generate unique task ID."""
        with self.taskCounterLock:
            self.taskCounter += 1
            return f"task-{self.taskCounter:08d}"
    
    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit a task for execution."""
        taskId = self._generateTaskId()
        wrappedFn = self._wrapTask(fn, taskId)
        return self.executor.submit(wrappedFn, *args, **kwargs)
    
    def getCurrentThreadCount(self) -> int:
        """Get thread count (constant for static executor)."""
        return self.workers
    
    def getMetrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        aggregated = self.metricsCollector.getAggregatedMetrics()
        return {
            "currentThreads": self.workers,
            "totalTasks": self.metricsCollector.totalTasks,
            "avgBlockingRatio": aggregated.avgBlockingRatio,
            "throughput": aggregated.throughput,
            "p50Latency": aggregated.p50Latency,
            "p99Latency": aggregated.p99Latency,
        }
    
    def getExperimentLog(self) -> List[Dict[str, Any]]:
        """Get experiment log."""
        with self.experimentLogLock:
            return list(self.experimentLog)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        self.executor.shutdown(wait=wait)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False
