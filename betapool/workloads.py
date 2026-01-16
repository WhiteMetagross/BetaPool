"""
Workload Generators for Testing and Benchmarking

Provides configurable I/O-bound, CPU-bound, and mixed workloads
for controlled experimental evaluation of the adaptive executor.

Author: Mridankan Mandal
"""

import time
import random
import math
from typing import Callable, Dict, Any, Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class WorkloadGenerator:
    """
    Factory for generating different types of workloads.
    
    Provides methods to create I/O-bound, CPU-bound, and mixed
    workloads with configurable parameters for experimental control.
    
    Example:
        # Create an I/O-bound task
        io_task = WorkloadGenerator.io_task(duration_ms=50.0)
        io_task()  # Sleeps for 50ms
        
        # Create a CPU-bound task
        cpu_task = WorkloadGenerator.cpu_task_python(iterations=100000)
        result = cpu_task()  # Performs computation
    """
    
    @staticmethod
    def io_task(duration_ms: float = 50.0) -> Callable[[], bool]:
        """
        Create an I/O-bound task using sleep.
        
        This simulates network latency, database queries, or file I/O
        where the thread is blocked waiting for external resources.
        The GIL is released during sleep, allowing other threads to run.
        
        Args:
            duration_ms: Sleep duration in milliseconds.
            
        Returns:
            Callable that sleeps for the specified duration.
        """
        def task() -> bool:
            time.sleep(duration_ms / 1000.0)
            return True
        
        return task
    
    @staticmethod
    def cpu_task_python(iterations: int = 100000) -> Callable[[], float]:
        """
        Create a CPU-bound task using pure Python.
        
        This task holds the GIL throughout execution, making it useful
        for testing GIL contention scenarios. The computation cannot
        be parallelized across threads due to GIL serialization.
        
        Args:
            iterations: Number of iterations for computation.
            
        Returns:
            Callable that performs CPU-intensive computation.
        """
        def task() -> float:
            result = 0.0
            for i in range(iterations):
                result += math.sin(i) * math.cos(i)
            return result
        
        return task
    
    @staticmethod
    def cpu_task_numpy(matrix_size: int = 100) -> Callable[[], Any]:
        """
        Create a CPU-bound task using NumPy.
        
        NumPy operations release the GIL during computation, allowing
        true parallelism. This tests the executor's behavior with
        GIL-releasing workloads.
        
        Args:
            matrix_size: Size of square matrices for multiplication.
            
        Returns:
            Callable that performs matrix multiplication.
            
        Raises:
            ImportError: If NumPy is not available.
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for cpu_task_numpy")
        
        def task() -> Any:
            a = np.random.rand(matrix_size, matrix_size)
            b = np.random.rand(matrix_size, matrix_size)
            return np.dot(a, b)
        
        return task
    
    @staticmethod
    def mixed_task(
        io_duration_ms: float = 50.0,
        cpu_iterations: int = 10000
    ) -> Callable[[], float]:
        """
        Create a mixed I/O and CPU task.
        
        Simulates realistic workloads like RAG pipelines where there
        are alternating phases of waiting and computation.
        
        Args:
            io_duration_ms: I/O wait duration in milliseconds.
            cpu_iterations: Number of CPU computation iterations.
            
        Returns:
            Callable with mixed I/O and CPU work.
        """
        def task() -> float:
            # Simulate network receive
            time.sleep(io_duration_ms / 1000.0)
            
            # CPU computation
            result = 0.0
            for i in range(cpu_iterations):
                result += math.sin(i) * math.cos(i)
            
            return result
        
        return task
    
    @staticmethod
    def rag_pipeline_task(
        network_latency_ms: float = 10.0,
        vector_db_latency_ms: float = 200.0,
        llm_latency_ms: float = 500.0,
        tokenization_iterations: int = 10000,
        reranking_iterations: int = 20000,
    ) -> Callable[[], Dict[str, Any]]:
        """
        Simulate a complete RAG (Retrieval-Augmented Generation) pipeline.
        
        Stages:
        1. Network receive (I/O)
        2. Tokenization (CPU)
        3. Vector DB query (I/O)
        4. Re-ranking (CPU)
        5. LLM generation (I/O - waiting for API response)
        
        Args:
            network_latency_ms: Initial network latency.
            vector_db_latency_ms: Vector database query latency.
            llm_latency_ms: LLM API call latency.
            tokenization_iterations: CPU work for tokenization.
            reranking_iterations: CPU work for re-ranking.
            
        Returns:
            Callable simulating RAG pipeline.
        """
        def task() -> Dict[str, Any]:
            stages: Dict[str, float] = {}
            
            # Stage 1: Network receive
            stage_start = time.time()
            time.sleep(network_latency_ms / 1000.0)
            stages["network_receive"] = time.time() - stage_start
            
            # Stage 2: Tokenization (CPU-bound)
            stage_start = time.time()
            result = 0.0
            for i in range(tokenization_iterations):
                result += math.sin(i) * math.cos(i)
            stages["tokenization"] = time.time() - stage_start
            
            # Stage 3: Vector DB query (I/O-bound)
            stage_start = time.time()
            time.sleep(vector_db_latency_ms / 1000.0)
            stages["vector_db"] = time.time() - stage_start
            
            # Stage 4: Re-ranking (CPU-bound)
            stage_start = time.time()
            for i in range(reranking_iterations):
                result += math.sin(i) * math.cos(i)
            stages["reranking"] = time.time() - stage_start
            
            # Stage 5: LLM generation (I/O-bound)
            stage_start = time.time()
            time.sleep(llm_latency_ms / 1000.0)
            stages["llm_generation"] = time.time() - stage_start
            
            return {
                "result": result,
                "stages": stages,
            }
        
        return task
    
    @staticmethod
    def fibonacci_task(n: int = 30) -> Callable[[], int]:
        """
        Create a pure Python recursive Fibonacci task.
        
        This holds the GIL for 100% of execution time, making it ideal
        for testing GIL saturation behavior. The recursive implementation
        is intentionally inefficient to demonstrate GIL contention.
        
        Args:
            n: Fibonacci number to compute.
            
        Returns:
            Callable computing nth Fibonacci number.
        """
        def fib(x: int) -> int:
            if x <= 1:
                return x
            return fib(x - 1) + fib(x - 2)
        
        def task() -> int:
            return fib(n)
        
        return task
    
    @staticmethod
    def variable_latency_task(
        min_ms: float = 10.0,
        max_ms: float = 100.0,
        cpu_fraction: float = 0.2
    ) -> Callable[[], float]:
        """
        Create a task with variable latency.
        
        Useful for simulating real-world workloads where request
        processing time varies unpredictably.
        
        Args:
            min_ms: Minimum task duration.
            max_ms: Maximum task duration.
            cpu_fraction: Fraction of time spent on CPU (rest is I/O).
            
        Returns:
            Callable with variable execution time.
        """
        def task() -> float:
            total_ms = random.uniform(min_ms, max_ms)
            cpu_ms = total_ms * cpu_fraction
            io_ms = total_ms - cpu_ms
            
            # I/O portion
            time.sleep(io_ms / 1000.0)
            
            # CPU portion
            iterations = int(cpu_ms * 1000)  # Approximate scaling
            result = 0.0
            for i in range(iterations):
                result += math.sin(i)
            
            return result
        
        return task


class PoissonArrivalGenerator:
    """
    Generate task arrivals following a Poisson process.
    
    Useful for simulating realistic request arrival patterns in
    server workloads.
    
    Example:
        arrivals = PoissonArrivalGenerator(rate_per_second=100.0)
        
        while running:
            arrivals.wait_for_next_arrival()
            executor.submit(task)
    """
    
    def __init__(self, rate_per_second: float):
        """
        Initialize the arrival generator.
        
        Args:
            rate_per_second: Average number of arrivals per second (lambda).
        """
        self.rate_per_second = rate_per_second
        self.last_arrival_time = time.time()
    
    def get_next_interarrival_time(self) -> float:
        """
        Get the time until the next arrival.
        
        Returns:
            Time in seconds until next arrival (exponentially distributed).
        """
        return random.expovariate(self.rate_per_second)
    
    def wait_for_next_arrival(self) -> None:
        """Wait until the next arrival time."""
        wait_time = self.get_next_interarrival_time()
        time.sleep(wait_time)


class BurstArrivalGenerator:
    """
    Generate bursty traffic patterns.
    
    Alternates between high-rate bursts and quiet periods to
    test the executor's response to sudden load changes.
    
    Example:
        arrivals = BurstArrivalGenerator(
            burst_rate_per_second=100.0,
            quiet_rate_per_second=10.0,
            burst_duration_sec=5.0,
            quiet_duration_sec=10.0,
        )
        
        while running:
            wait_time = arrivals.get_next_interarrival_time()
            time.sleep(wait_time)
            executor.submit(task)
    """
    
    def __init__(
        self,
        burst_rate_per_second: float = 100.0,
        quiet_rate_per_second: float = 10.0,
        burst_duration_sec: float = 5.0,
        quiet_duration_sec: float = 10.0,
    ):
        """
        Initialize the burst generator.
        
        Args:
            burst_rate_per_second: Request rate during bursts.
            quiet_rate_per_second: Request rate during quiet periods.
            burst_duration_sec: Duration of each burst.
            quiet_duration_sec: Duration of each quiet period.
        """
        self.burst_rate = burst_rate_per_second
        self.quiet_rate = quiet_rate_per_second
        self.burst_duration = burst_duration_sec
        self.quiet_duration = quiet_duration_sec
        
        self.cycle_start = time.time()
        self.in_burst = True
    
    def get_current_rate(self) -> float:
        """
        Get the current arrival rate based on cycle position.
        
        Returns:
            Current requests per second.
        """
        elapsed = time.time() - self.cycle_start
        cycle_duration = self.burst_duration + self.quiet_duration
        
        position_in_cycle = elapsed % cycle_duration
        
        if position_in_cycle < self.burst_duration:
            return self.burst_rate
        else:
            return self.quiet_rate
    
    def get_next_interarrival_time(self) -> float:
        """
        Get time until next arrival based on current rate.
        
        Returns:
            Time in seconds until next arrival.
        """
        rate = self.get_current_rate()
        return random.expovariate(rate) if rate > 0 else 1.0
