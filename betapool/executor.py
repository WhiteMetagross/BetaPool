"""
Adaptive Thread Pool Executor with Runtime Resizing

Core implementation of the BetaPool algorithm using blocking ratio
based workload classification and GIL Safety Veto mechanism.

The key insight is the blocking ratio (beta):
    beta = 1 - (cpu_time / wall_time)

When beta is high (> 0.7), threads are waiting on I/O, safe to add more.
When beta is low (< 0.3), threads are CPU-bound, adding more causes GIL contention.

Author: Mridankan Mandal
"""

from concurrent.futures import ThreadPoolExecutor, Future
from threading import Thread, Lock, Event
from typing import Callable, Any, Optional, Dict, List, TypeVar, Iterator
from functools import wraps
from dataclasses import dataclass, field
import time
import logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from betapool.metrics import TaskMetrics, MetricsCollector


# Configure module logger
logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ControllerConfig:
    """
    Configuration parameters for the adaptive controller.
    
    These values are tuned based on empirical testing. Adjust for specific
    deployments and hardware configurations.
    
    Attributes:
        monitor_interval_sec: Time between controller decisions.
        beta_high_threshold: Blocking ratio above which we scale up (I/O-bound).
        beta_low_threshold: Blocking ratio below which we scale down (CPU-bound).
        scale_up_step: Number of threads to add when scaling up.
        scale_down_step: Number of threads to remove when scaling down.
        cpu_upper_threshold: CPU utilization above which we avoid scaling up.
        cpu_lower_threshold: CPU utilization below which we may scale down.
        stabilization_window_sec: Minimum time between scaling decisions.
        warmup_task_count: Minimum tasks before making scaling decisions.
    """
    monitor_interval_sec: float = 0.5
    beta_high_threshold: float = 0.7
    beta_low_threshold: float = 0.3
    scale_up_step: int = 2
    scale_down_step: int = 1
    cpu_upper_threshold: float = 85.0
    cpu_lower_threshold: float = 50.0
    stabilization_window_sec: float = 2.0
    warmup_task_count: int = 10


@dataclass
class ControllerState:
    """
    Runtime state of the adaptive controller.
    
    Tracks scaling decisions and provides data for monitoring and analysis.
    
    Attributes:
        current_threads: Current number of active worker threads.
        last_scale_time: Timestamp of the last scaling decision.
        scale_up_count: Total number of scale-up decisions made.
        scale_down_count: Total number of scale-down decisions made.
        decision_history: List of all scaling decisions for analysis.
    """
    current_threads: int
    last_scale_time: float = field(default_factory=time.time)
    scale_up_count: int = 0
    scale_down_count: int = 0
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_decision(
        self,
        timestamp: float,
        threads_before: int,
        threads_after: int,
        blocking_ratio: float,
        cpu_percent: float,
        throughput: float,
        decision: str,
    ) -> None:
        """
        Record a scaling decision for later analysis.
        
        Args:
            timestamp: Time of decision.
            threads_before: Thread count before decision.
            threads_after: Thread count after decision.
            blocking_ratio: Current blocking ratio.
            cpu_percent: Current CPU utilization.
            throughput: Current throughput.
            decision: Decision made (scaleUp, scaleDown, hold).
        """
        self.decision_history.append({
            "timestamp": timestamp,
            "threads_before": threads_before,
            "threads_after": threads_after,
            "blocking_ratio": blocking_ratio,
            "cpu_percent": cpu_percent,
            "throughput": throughput,
            "decision": decision,
        })


class AdaptiveThreadPoolExecutor:
    """
    Thread pool executor with adaptive sizing based on workload characteristics.
    
    This executor monitors task execution patterns to distinguish between
    I/O-bound and CPU-bound workloads, adjusting the thread count accordingly
    using the GIL Safety Veto mechanism.
    
    Key insight: The blocking ratio (1 - cpu_time/wall_time) provides a
    reliable signal for workload classification without requiring GIL-level
    instrumentation.
    
    Example:
        with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
            futures = [executor.submit(task, arg) for arg in args]
            results = [f.result() for f in futures]
    
    Attributes:
        min_workers: Minimum thread count (typically = physical cores).
        max_workers: Maximum thread count (for I/O-heavy workloads).
        config: Controller configuration parameters.
    """
    
    def __init__(
        self,
        min_workers: int = 4,
        max_workers: int = 64,
        config: Optional[ControllerConfig] = None,
        enable_logging: bool = False,
    ):
        """
        Initialize the adaptive executor.
        
        Args:
            min_workers: Minimum thread count (typically = physical cores).
            max_workers: Maximum thread count (for I/O-heavy workloads).
            config: Controller configuration. Uses defaults if None.
            enable_logging: Enable debug logging for development.
            
        Raises:
            ValueError: If min_workers < 1 or max_workers < min_workers.
        """
        # Validate parameters
        if min_workers < 1:
            raise ValueError("min_workers must be at least 1")
        if max_workers < min_workers:
            raise ValueError("max_workers must be >= min_workers")
        
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.config = config or ControllerConfig()
        self._enable_logging = enable_logging
        
        # Initialize the underlying executor with minimum workers
        self._executor = ThreadPoolExecutor(max_workers=min_workers)
        
        # Metrics collection
        self._metrics_collector = MetricsCollector()
        
        # Controller state
        self._controller_state = ControllerState(current_threads=min_workers)
        self._controller_lock = Lock()
        
        # Monitor thread control
        self._stop_event = Event()
        self._monitor_thread: Optional[Thread] = None
        
        # Task tracking
        self._task_counter = 0
        self._task_counter_lock = Lock()
        
        # Experiment data logging
        self._experiment_log: List[Dict[str, Any]] = []
        self._experiment_log_lock = Lock()
        
        # Start the monitor thread
        self._start_monitor()
    
    def _start_monitor(self) -> None:
        """Start the background monitoring thread."""
        self._monitor_thread = Thread(
            target=self._monitor_loop,
            daemon=True,
            name="BetaPool-Monitor"
        )
        self._monitor_thread.start()
    
    def _monitor_loop(self) -> None:
        """
        Main loop for the controller thread.
        
        Wakes up periodically to evaluate workload and adjust thread count.
        Implements the GIL Safety Veto mechanism: if blocking ratio indicates
        CPU-bound work, refuse to add threads regardless of queue pressure.
        """
        while not self._stop_event.is_set():
            time.sleep(self.config.monitor_interval_sec)
            
            if self._stop_event.is_set():
                break
            
            try:
                self._make_scaling_decision()
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
    
    def _get_cpu_percent(self) -> float:
        """Get current CPU utilization percentage."""
        if PSUTIL_AVAILABLE:
            return psutil.cpu_percent(interval=None)
        return 50.0  # Default if psutil not available
    
    def _make_scaling_decision(self) -> None:
        """
        Evaluate current metrics and decide whether to scale.
        
        Decision logic (GIL Safety Veto):
        1. If beta > high_threshold and CPU < upper_limit: scale UP (I/O-bound)
        2. If beta < low_threshold or CPU > upper_limit: scale DOWN (CPU-bound/saturated)
        3. Otherwise: maintain current level (VETO further scaling)
        """
        current_time = time.time()
        
        # Check stabilization window
        time_since_last_scale = current_time - self._controller_state.last_scale_time
        if time_since_last_scale < self.config.stabilization_window_sec:
            return
        
        # Check warmup period
        if self._metrics_collector.total_tasks < self.config.warmup_task_count:
            return
        
        # Gather current metrics
        beta = self._metrics_collector.get_recent_blocking_ratio()
        cpu_percent = self._get_cpu_percent()
        throughput = self._metrics_collector.get_throughput()
        
        with self._controller_lock:
            current_threads = self._controller_state.current_threads
            decision = "hold"
            new_threads = current_threads
            
            # Decision logic based on blocking ratio and CPU utilization
            if beta > self.config.beta_high_threshold:
                # High blocking ratio indicates I/O-bound workload
                if cpu_percent < self.config.cpu_upper_threshold:
                    # CPU has headroom, scale up
                    new_threads = min(
                        current_threads + self.config.scale_up_step,
                        self.max_workers
                    )
                    if new_threads > current_threads:
                        decision = "scaleUp"
            
            elif beta < self.config.beta_low_threshold:
                # Low blocking ratio indicates CPU-bound workload
                # GIL Safety Veto: scale down to reduce GIL contention
                new_threads = max(
                    current_threads - self.config.scale_down_step,
                    self.min_workers
                )
                if new_threads < current_threads:
                    decision = "scaleDown"
            
            elif cpu_percent > self.config.cpu_upper_threshold:
                # CPU is saturated, scale down regardless of beta
                new_threads = max(
                    current_threads - self.config.scale_down_step,
                    self.min_workers
                )
                if new_threads < current_threads:
                    decision = "scaleDown"
            
            # Apply the scaling decision
            if new_threads != current_threads:
                self._resize_pool(new_threads)
                self._controller_state.current_threads = new_threads
                self._controller_state.last_scale_time = current_time
                
                if decision == "scaleUp":
                    self._controller_state.scale_up_count += 1
                else:
                    self._controller_state.scale_down_count += 1
            
            # Record decision for analysis
            self._controller_state.record_decision(
                timestamp=current_time,
                threads_before=current_threads,
                threads_after=new_threads,
                blocking_ratio=beta,
                cpu_percent=cpu_percent,
                throughput=throughput,
                decision=decision,
            )
            
            # Log experiment data point
            self._log_experiment_data(
                current_time, new_threads, beta, cpu_percent, throughput
            )
            
            if self._enable_logging:
                logger.debug(
                    f"Controller: threads={new_threads}, beta={beta:.3f}, "
                    f"cpu={cpu_percent:.1f}%, throughput={throughput:.1f}, "
                    f"decision={decision}"
                )
    
    def _resize_pool(self, new_size: int) -> None:
        """
        Resize the thread pool at runtime.
        
        Uses internal ThreadPoolExecutor attributes to adjust pool size.
        This is a well-known technique in Python systems programming.
        
        Args:
            new_size: New maximum worker count.
        """
        self._executor._max_workers = new_size
        self._executor._adjust_thread_count()
    
    def _log_experiment_data(
        self,
        timestamp: float,
        active_threads: int,
        blocking_ratio: float,
        cpu_percent: float,
        throughput: float,
    ) -> None:
        """Log data point for experiment visualization."""
        latencies = self._metrics_collector.get_latency_percentiles()
        
        with self._experiment_log_lock:
            self._experiment_log.append({
                "timestamp": timestamp,
                "active_threads": active_threads,
                "blocking_ratio": blocking_ratio,
                "cpu_percent": cpu_percent,
                "throughput": throughput,
                "p50_latency": latencies["p50"],
                "p99_latency": latencies["p99"],
            })
    
    def _wrap_task(self, fn: Callable[..., T], task_id: str) -> Callable[..., T]:
        """
        Wrap a task function to capture execution metrics.
        
        The wrapper measures both wall time (total elapsed) and CPU time
        (actual execution on CPU) to compute the blocking ratio.
        
        Args:
            fn: The callable to wrap.
            task_id: Unique identifier for this task.
            
        Returns:
            Wrapped callable with metrics instrumentation.
        """
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            wall_start = time.time()
            cpu_start = time.thread_time()
            
            success = True
            error_message = None
            result = None
            
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                cpu_end = time.thread_time()
                wall_end = time.time()
                
                wall_time = wall_end - wall_start
                cpu_time = cpu_end - cpu_start
                
                # Compute blocking ratio, handling edge cases
                if wall_time > 0:
                    blocking_ratio = 1.0 - (cpu_time / wall_time)
                else:
                    blocking_ratio = 0.0
                
                # Record metrics
                metrics = TaskMetrics(
                    task_id=task_id,
                    wall_time=wall_time,
                    cpu_time=cpu_time,
                    blocking_ratio=blocking_ratio,
                    timestamp=wall_end,
                    success=success,
                    error_message=error_message,
                )
                self._metrics_collector.record(metrics)
            
            return result
        
        return wrapper
    
    def _generate_task_id(self) -> str:
        """Generate a unique task identifier."""
        with self._task_counter_lock:
            self._task_counter += 1
            return f"task-{self._task_counter:08d}"
    
    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future:
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
        task_id = self._generate_task_id()
        wrapped_fn = self._wrap_task(fn, task_id)
        return self._executor.submit(wrapped_fn, *args, **kwargs)
    
    def map(
        self,
        fn: Callable[..., T],
        *iterables: Any,
        timeout: Optional[float] = None,
        chunksize: int = 1
    ) -> Iterator[T]:
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
    
    def get_current_thread_count(self) -> int:
        """
        Get the current active thread count.
        
        Returns:
            Current number of worker threads.
        """
        with self._controller_lock:
            return self._controller_state.current_threads
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics summary.
        
        Returns:
            Dictionary containing current executor state and metrics.
        """
        aggregated = self._metrics_collector.get_aggregated_metrics()
        
        return {
            "current_threads": self.get_current_thread_count(),
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "total_tasks": self._metrics_collector.total_tasks,
            "avg_blocking_ratio": aggregated.avg_blocking_ratio,
            "throughput": aggregated.throughput,
            "p50_latency": aggregated.p50_latency,
            "p99_latency": aggregated.p99_latency,
            "scale_up_count": self._controller_state.scale_up_count,
            "scale_down_count": self._controller_state.scale_down_count,
        }
    
    def get_experiment_log(self) -> List[Dict[str, Any]]:
        """
        Get the experiment data log for visualization.
        
        Returns:
            List of data points recorded during execution.
        """
        with self._experiment_log_lock:
            return list(self._experiment_log)
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of scaling decisions.
        
        Returns:
            List of all scaling decisions made by the controller.
        """
        return list(self._controller_state.decision_history)
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.
        
        Args:
            wait: If True, wait for pending tasks to complete.
        """
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        
        self._executor.shutdown(wait=wait)
    
    def __enter__(self) -> "AdaptiveThreadPoolExecutor":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit."""
        self.shutdown(wait=True)
        return False


class StaticThreadPoolExecutor:
    """
    Static thread pool executor for baseline comparison.
    
    Provides the same interface as AdaptiveThreadPoolExecutor but with
    fixed thread count. Useful for benchmarking against the adaptive solution.
    
    Example:
        with StaticThreadPoolExecutor(workers=16) as executor:
            futures = [executor.submit(task, arg) for arg in args]
            results = [f.result() for f in futures]
    """
    
    def __init__(self, workers: int = 4):
        """
        Initialize with fixed worker count.
        
        Args:
            workers: Number of worker threads (fixed throughout lifetime).
        """
        self.workers = workers
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._metrics_collector = MetricsCollector()
        
        self._task_counter = 0
        self._task_counter_lock = Lock()
        
        self._experiment_log: List[Dict[str, Any]] = []
        self._experiment_log_lock = Lock()
        self._start_time = time.time()
    
    def _wrap_task(self, fn: Callable[..., T], task_id: str) -> Callable[..., T]:
        """Wrap task for metrics collection."""
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            wall_start = time.time()
            cpu_start = time.thread_time()
            
            result = None
            success = True
            
            try:
                result = fn(*args, **kwargs)
            except Exception:
                success = False
                raise
            finally:
                cpu_end = time.thread_time()
                wall_end = time.time()
                
                wall_time = wall_end - wall_start
                cpu_time = cpu_end - cpu_start
                
                if wall_time > 0:
                    blocking_ratio = 1.0 - (cpu_time / wall_time)
                else:
                    blocking_ratio = 0.0
                
                metrics = TaskMetrics(
                    task_id=task_id,
                    wall_time=wall_time,
                    cpu_time=cpu_time,
                    blocking_ratio=blocking_ratio,
                    timestamp=wall_end,
                    success=success,
                )
                self._metrics_collector.record(metrics)
                
                # Log experiment data
                cpu_percent = psutil.cpu_percent(interval=None) if PSUTIL_AVAILABLE else 50.0
                with self._experiment_log_lock:
                    self._experiment_log.append({
                        "timestamp": wall_end,
                        "active_threads": self.workers,
                        "blocking_ratio": blocking_ratio,
                        "cpu_percent": cpu_percent,
                        "throughput": self._metrics_collector.get_throughput(),
                        "p50_latency": 0.0,
                        "p99_latency": 0.0,
                    })
            
            return result
        
        return wrapper
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        with self._task_counter_lock:
            self._task_counter += 1
            return f"task-{self._task_counter:08d}"
    
    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future:
        """
        Submit a task for execution.
        
        Args:
            fn: The callable to execute.
            *args: Positional arguments for the callable.
            **kwargs: Keyword arguments for the callable.
            
        Returns:
            A Future representing the pending execution.
        """
        task_id = self._generate_task_id()
        wrapped_fn = self._wrap_task(fn, task_id)
        return self._executor.submit(wrapped_fn, *args, **kwargs)
    
    def get_current_thread_count(self) -> int:
        """Get thread count (constant for static executor)."""
        return self.workers
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Dictionary containing executor state and metrics.
        """
        aggregated = self._metrics_collector.get_aggregated_metrics()
        return {
            "current_threads": self.workers,
            "total_tasks": self._metrics_collector.total_tasks,
            "avg_blocking_ratio": aggregated.avg_blocking_ratio,
            "throughput": aggregated.throughput,
            "p50_latency": aggregated.p50_latency,
            "p99_latency": aggregated.p99_latency,
        }
    
    def get_experiment_log(self) -> List[Dict[str, Any]]:
        """Get experiment log."""
        with self._experiment_log_lock:
            return list(self._experiment_log)
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.
        
        Args:
            wait: If True, wait for pending tasks to complete.
        """
        self._executor.shutdown(wait=wait)
    
    def __enter__(self) -> "StaticThreadPoolExecutor":
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.shutdown(wait=True)
        return False
