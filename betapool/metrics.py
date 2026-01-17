"""
Task Metrics and Instrumentation Module

Provides fine-grained measurement of task execution characteristics
including CPU time, wall time, and blocking ratio calculation.

The blocking ratio (beta) is the core metric used by the adaptive controller
to distinguish between I/O-bound and CPU-bound workloads:

    beta = 1.0 - (cpu_time / wall_time)

- beta near 1.0: I/O-bound (thread is waiting, safe to add more threads)
- beta near 0.0: CPU-bound (thread is computing, adding threads causes GIL contention)

Author: Mridankan Mandal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import deque
from threading import Lock
import time
import statistics


@dataclass
class TaskMetrics:
    """
    Container for individual task execution metrics.
    
    The blocking ratio is computed as the fraction of wall time spent
    waiting (I/O, locks, sleep) rather than executing on CPU.
    
    Attributes:
        task_id: Unique identifier for the task.
        wall_time: Total elapsed time in seconds.
        cpu_time: Thread CPU time in seconds.
        blocking_ratio: Computed as 1 - (cpu_time / wall_time).
        timestamp: Unix timestamp when task completed.
        success: Whether the task completed successfully.
        error_message: Error message if task failed.
    """
    task_id: str
    wall_time: float
    cpu_time: float
    blocking_ratio: float
    timestamp: float
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Ensure blocking ratio is within valid bounds [0, 1]."""
        self.blocking_ratio = max(0.0, min(1.0, self.blocking_ratio))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "wall_time": self.wall_time,
            "cpu_time": self.cpu_time,
            "blocking_ratio": self.blocking_ratio,
            "timestamp": self.timestamp,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class AggregatedMetrics:
    """
    Aggregated metrics over a time window for controller decisions.
    
    Attributes:
        window_start: Start timestamp of the measurement window.
        window_end: End timestamp of the measurement window.
        task_count: Number of tasks in the window.
        avg_blocking_ratio: Mean blocking ratio.
        std_blocking_ratio: Standard deviation of blocking ratio.
        avg_wall_time: Mean wall clock time per task.
        avg_cpu_time: Mean CPU time per task.
        throughput: Tasks completed per second.
        p50_latency: Median latency.
        p99_latency: 99th percentile latency.
    """
    window_start: float
    window_end: float
    task_count: int
    avg_blocking_ratio: float
    std_blocking_ratio: float
    avg_wall_time: float
    avg_cpu_time: float
    throughput: float
    p50_latency: float
    p99_latency: float


class MetricsCollector:
    """
    Thread-safe collector for task execution metrics.
    
    Maintains a rolling window of recent task metrics for use by the
    adaptive controller. Uses a lock-protected deque for thread safety.
    
    Example:
        collector = MetricsCollector(window_size=100)
        collector.record(metrics)
        avg_beta = collector.get_recent_blocking_ratio()
    """
    
    def __init__(self, window_size: int = 100, max_history: int = 10000):
        """
        Initialize the metrics collector.
        
        Args:
            window_size: Number of recent tasks to consider for rolling average.
            max_history: Maximum number of metrics to retain in history.
        """
        self.window_size = window_size
        self.max_history = max_history
        self._recent_metrics: deque = deque(maxlen=max_history)
        self._lock = Lock()
        self._start_time = time.time()
        
        # Counters for throughput calculation
        self.total_tasks = 0
        self._window_task_count = 0
        self._last_window_reset = time.time()
    
    def record(self, metrics: TaskMetrics) -> None:
        """
        Record a completed task's metrics.
        
        Args:
            metrics: TaskMetrics instance containing execution data.
        """
        with self._lock:
            self._recent_metrics.append(metrics)
            self.total_tasks += 1
            self._window_task_count += 1
    
    def get_recent_blocking_ratio(self, n: Optional[int] = None) -> float:
        """
        Compute the average blocking ratio over recent tasks.
        
        Args:
            n: Number of recent tasks to consider. Defaults to window_size.
            
        Returns:
            Average blocking ratio, or 0.5 if no tasks recorded.
        """
        n = n or self.window_size
        with self._lock:
            if not self._recent_metrics:
                return 0.5  # Neutral default when no data
            
            recent = list(self._recent_metrics)[-n:]
            if not recent:
                return 0.5
            
            return statistics.mean(m.blocking_ratio for m in recent)
    
    def get_blocking_ratio_std(self, n: Optional[int] = None) -> float:
        """
        Compute the standard deviation of blocking ratio.
        
        Args:
            n: Number of recent tasks to consider.
            
        Returns:
            Standard deviation, or 0.0 if insufficient data.
        """
        n = n or self.window_size
        with self._lock:
            if len(self._recent_metrics) < 2:
                return 0.0
            
            recent = list(self._recent_metrics)[-n:]
            if len(recent) < 2:
                return 0.0
            
            return statistics.stdev(m.blocking_ratio for m in recent)
    
    def get_throughput(self) -> float:
        """
        Compute current throughput in tasks per second.
        
        Returns:
            Tasks completed per second over the measurement window.
        """
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self._last_window_reset
            if elapsed < 0.1:  # Avoid division by near-zero
                return 0.0
            
            throughput = self._window_task_count / elapsed
            
            # Reset window if enough time has passed
            if elapsed > 1.0:
                self._window_task_count = 0
                self._last_window_reset = current_time
            
            return throughput
    
    def get_latency_percentiles(self, n: Optional[int] = None) -> Dict[str, float]:
        """
        Compute latency percentiles from recent tasks.
        
        Args:
            n: Number of recent tasks to consider.
            
        Returns:
            Dictionary with p50, p90, p99 latency values in seconds.
        """
        n = n or self.window_size
        with self._lock:
            if not self._recent_metrics:
                return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
            
            recent = list(self._recent_metrics)[-n:]
            latencies = sorted(m.wall_time for m in recent)
            
            def percentile(data: List[float], p: float) -> float:
                if not data:
                    return 0.0
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = f + 1 if f + 1 < len(data) else f
                return data[f] + (data[c] - data[f]) * (k - f)
            
            return {
                "p50": percentile(latencies, 50),
                "p90": percentile(latencies, 90),
                "p99": percentile(latencies, 99),
            }
    
    def get_aggregated_metrics(self, n: Optional[int] = None) -> AggregatedMetrics:
        """
        Get comprehensive aggregated metrics over recent tasks.
        
        Args:
            n: Number of recent tasks to consider.
            
        Returns:
            AggregatedMetrics instance with all computed statistics.
        """
        n = n or self.window_size
        with self._lock:
            current_time = time.time()
            
            if not self._recent_metrics:
                return AggregatedMetrics(
                    window_start=self._start_time,
                    window_end=current_time,
                    task_count=0,
                    avg_blocking_ratio=0.5,
                    std_blocking_ratio=0.0,
                    avg_wall_time=0.0,
                    avg_cpu_time=0.0,
                    throughput=0.0,
                    p50_latency=0.0,
                    p99_latency=0.0,
                )
            
            recent = list(self._recent_metrics)[-n:]
            latencies = sorted(m.wall_time for m in recent)
            
            def percentile(data: List[float], p: float) -> float:
                if not data:
                    return 0.0
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = f + 1 if f + 1 < len(data) else f
                return data[f] + (data[c] - data[f]) * (k - f)
            
            elapsed = current_time - self._last_window_reset
            throughput = self._window_task_count / elapsed if elapsed > 0.1 else 0.0
            
            return AggregatedMetrics(
                window_start=recent[0].timestamp if recent else self._start_time,
                window_end=current_time,
                task_count=len(recent),
                avg_blocking_ratio=statistics.mean(m.blocking_ratio for m in recent),
                std_blocking_ratio=statistics.stdev(m.blocking_ratio for m in recent) if len(recent) > 1 else 0.0,
                avg_wall_time=statistics.mean(m.wall_time for m in recent),
                avg_cpu_time=statistics.mean(m.cpu_time for m in recent),
                throughput=throughput,
                p50_latency=percentile(latencies, 50),
                p99_latency=percentile(latencies, 99),
            )
    
    def reset(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._recent_metrics.clear()
            self.total_tasks = 0
            self._window_task_count = 0
            self._last_window_reset = time.time()
            self._start_time = time.time()
    
    def export_to_list(self) -> List[Dict[str, Any]]:
        """
        Export all metrics as a list of dictionaries for serialization.
        
        Returns:
            List of metric dictionaries suitable for CSV/JSON export.
        """
        with self._lock:
            return [m.to_dict() for m in self._recent_metrics]
