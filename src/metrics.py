# Task Metrics and Instrumentation Module
# Provides fine-grained measurement of task execution characteristics
# including CPU time, wall time, and blocking ratio calculation.

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import deque
from threading import Lock
import time
import statistics


@dataclass
class TaskMetrics:
    # Container for individual task execution metrics.
    # The blocking ratio is computed as the fraction of wall time spent
    # waiting (I/O, locks, sleep) rather than executing on CPU:
    #     beta = 1.0 - (cpu_time / wall_time)
    # A blocking ratio near 1.0 indicates I/O-bound behavior.
    # A blocking ratio near 0.0 indicates CPU-bound behavior.
    taskId: str
    wallTime: float  # Total elapsed time in seconds
    cpuTime: float   # Thread CPU time in seconds
    blockingRatio: float  # Computed: 1 - (cpuTime / wallTime)
    timestamp: float  # Unix timestamp when task completed
    success: bool = True
    errorMessage: Optional[str] = None
    
    def __post_init__(self):
        # Ensure blocking ratio is within valid bounds
        self.blockingRatio = max(0.0, min(1.0, self.blockingRatio))


@dataclass
class AggregatedMetrics:
    # Aggregated metrics over a time window for controller decisions.
    windowStart: float
    windowEnd: float
    taskCount: int
    avgBlockingRatio: float
    stdBlockingRatio: float
    avgWallTime: float
    avgCpuTime: float
    throughput: float  # Tasks per second
    p50Latency: float
    p99Latency: float


class MetricsCollector:
    """
    Thread-safe collector for task execution metrics.
    
    Maintains a rolling window of recent task metrics for use by the
    adaptive controller. Uses a lock-protected deque for thread safety.
    """
    
    def __init__(self, windowSize: int = 100, maxHistory: int = 10000):
        """
        Initialize the metrics collector.
        
        Args:
            windowSize: Number of recent tasks to consider for rolling average.
            maxHistory: Maximum number of metrics to retain in history.
        """
        self.windowSize = windowSize
        self.maxHistory = maxHistory
        self.recentMetrics: deque = deque(maxlen=maxHistory)
        self.lock = Lock()
        self.startTime = time.time()
        
        # Counters for throughput calculation
        self.totalTasks = 0
        self.windowTaskCount = 0
        self.lastWindowReset = time.time()
    
    def record(self, metrics: TaskMetrics) -> None:
        """
        Record a completed task's metrics.
        
        Args:
            metrics: TaskMetrics instance containing execution data.
        """
        with self.lock:
            self.recentMetrics.append(metrics)
            self.totalTasks += 1
            self.windowTaskCount += 1
    
    def getRecentBlockingRatio(self, n: Optional[int] = None) -> float:
        """
        Compute the average blocking ratio over recent tasks.
        
        Args:
            n: Number of recent tasks to consider. Defaults to windowSize.
            
        Returns:
            Average blocking ratio, or 0.5 if no tasks recorded.
        """
        n = n or self.windowSize
        with self.lock:
            if not self.recentMetrics:
                return 0.5  # Neutral default when no data
            
            recent = list(self.recentMetrics)[-n:]
            if not recent:
                return 0.5
            
            return statistics.mean(m.blockingRatio for m in recent)
    
    def getBlockingRatioStd(self, n: Optional[int] = None) -> float:
        """
        Compute the standard deviation of blocking ratio.
        
        Args:
            n: Number of recent tasks to consider.
            
        Returns:
            Standard deviation, or 0.0 if insufficient data.
        """
        n = n or self.windowSize
        with self.lock:
            if len(self.recentMetrics) < 2:
                return 0.0
            
            recent = list(self.recentMetrics)[-n:]
            if len(recent) < 2:
                return 0.0
            
            return statistics.stdev(m.blockingRatio for m in recent)
    
    def getThroughput(self) -> float:
        """
        Compute current throughput in tasks per second.
        
        Returns:
            Tasks completed per second over the measurement window.
        """
        with self.lock:
            currentTime = time.time()
            elapsed = currentTime - self.lastWindowReset
            if elapsed < 0.1:  # Avoid division by near-zero
                return 0.0
            
            throughput = self.windowTaskCount / elapsed
            
            # Reset window if enough time has passed
            if elapsed > 1.0:
                self.windowTaskCount = 0
                self.lastWindowReset = currentTime
            
            return throughput
    
    def getLatencyPercentiles(self, n: Optional[int] = None) -> Dict[str, float]:
        """
        Compute latency percentiles from recent tasks.
        
        Args:
            n: Number of recent tasks to consider.
            
        Returns:
            Dictionary with p50, p90, p99 latency values.
        """
        n = n or self.windowSize
        with self.lock:
            if not self.recentMetrics:
                return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
            
            recent = list(self.recentMetrics)[-n:]
            latencies = sorted(m.wallTime for m in recent)
            
            def percentile(data: List[float], p: float) -> float:
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = f + 1 if f + 1 < len(data) else f
                return data[f] + (data[c] - data[f]) * (k - f)
            
            return {
                "p50": percentile(latencies, 50),
                "p90": percentile(latencies, 90),
                "p99": percentile(latencies, 99),
            }
    
    def getAggregatedMetrics(self, n: Optional[int] = None) -> AggregatedMetrics:
        """
        Get comprehensive aggregated metrics over recent tasks.
        
        Args:
            n: Number of recent tasks to consider.
            
        Returns:
            AggregatedMetrics instance with all computed statistics.
        """
        n = n or self.windowSize
        with self.lock:
            currentTime = time.time()
            
            if not self.recentMetrics:
                return AggregatedMetrics(
                    windowStart=self.startTime,
                    windowEnd=currentTime,
                    taskCount=0,
                    avgBlockingRatio=0.5,
                    stdBlockingRatio=0.0,
                    avgWallTime=0.0,
                    avgCpuTime=0.0,
                    throughput=0.0,
                    p50Latency=0.0,
                    p99Latency=0.0,
                )
            
            recent = list(self.recentMetrics)[-n:]
            latencies = sorted(m.wallTime for m in recent)
            
            def percentile(data: List[float], p: float) -> float:
                if not data:
                    return 0.0
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = f + 1 if f + 1 < len(data) else f
                return data[f] + (data[c] - data[f]) * (k - f)
            
            elapsed = currentTime - self.lastWindowReset
            throughput = self.windowTaskCount / elapsed if elapsed > 0.1 else 0.0
            
            return AggregatedMetrics(
                windowStart=recent[0].timestamp if recent else self.startTime,
                windowEnd=currentTime,
                taskCount=len(recent),
                avgBlockingRatio=statistics.mean(m.blockingRatio for m in recent),
                stdBlockingRatio=statistics.stdev(m.blockingRatio for m in recent) if len(recent) > 1 else 0.0,
                avgWallTime=statistics.mean(m.wallTime for m in recent),
                avgCpuTime=statistics.mean(m.cpuTime for m in recent),
                throughput=throughput,
                p50Latency=percentile(latencies, 50),
                p99Latency=percentile(latencies, 99),
            )
    
    def reset(self) -> None:
        """Clear all collected metrics."""
        with self.lock:
            self.recentMetrics.clear()
            self.totalTasks = 0
            self.windowTaskCount = 0
            self.lastWindowReset = time.time()
            self.startTime = time.time()
    
    def exportToDict(self) -> List[Dict[str, Any]]:
        """
        Export all metrics as a list of dictionaries for serialization.
        
        Returns:
            List of metric dictionaries suitable for CSV/JSON export.
        """
        with self.lock:
            return [
                {
                    "taskId": m.taskId,
                    "wallTime": m.wallTime,
                    "cpuTime": m.cpuTime,
                    "blockingRatio": m.blockingRatio,
                    "timestamp": m.timestamp,
                    "success": m.success,
                }
                for m in self.recentMetrics
            ]
