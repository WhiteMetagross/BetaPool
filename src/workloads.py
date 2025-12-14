# Workload Generators for Experiments
# Provides configurable I/O-bound, CPU-bound, and mixed workloads
# for controlled experimental evaluation.

import time
import random
import math
from typing import Callable, Optional
import numpy as np


class WorkloadGenerator:
    # Factory for generating different types of workloads.
    # Provides methods to create I/O-bound, CPU-bound, and mixed
    # workloads with configurable parameters for experimental control.
    
    @staticmethod
    def ioTask(durationMs: float = 50.0) -> Callable[[], None]:
        # Create an I/O-bound task using sleep.
        # This simulates network latency, database queries, or file I/O
        # where the thread is blocked waiting for external resources.
        # durationMs: Sleep duration in milliseconds.
        # Returns: Callable that sleeps for the specified duration.
        def task():
            time.sleep(durationMs / 1000.0)
            return True
        
        return task
    
    @staticmethod
    def cpuTaskPython(iterations: int = 100000) -> Callable[[], float]:
        # Create a CPU-bound task using pure Python.
        # This task holds the GIL throughout execution, making it useful
        # for testing GIL contention scenarios.
        # iterations: Number of iterations for computation.
        # Returns: Callable that performs CPU-intensive computation.
        def task():
            result = 0.0
            for i in range(iterations):
                result += math.sin(i) * math.cos(i)
            return result
        
        return task
    
    @staticmethod
    def cpuTaskNumpy(matrixSize: int = 100) -> Callable[[], np.ndarray]:
        """
        Create a CPU-bound task using NumPy.
        
        NumPy operations release the GIL during computation, allowing
        true parallelism. This tests the executor's behavior with
        GIL-releasing workloads.
        
        Args:
            matrixSize: Size of square matrices for multiplication.
            
        Returns:
            Callable that performs matrix multiplication.
        """
        def task():
            a = np.random.rand(matrixSize, matrixSize)
            b = np.random.rand(matrixSize, matrixSize)
            return np.dot(a, b)
        
        return task
    
    @staticmethod
    def mixedTask(
        ioDurationMs: float = 50.0,
        cpuIterations: int = 10000
    ) -> Callable[[], float]:
        """
        Create a mixed I/O and CPU task.
        
        Simulates realistic workloads like RAG pipelines where there
        are alternating phases of waiting and computation.
        
        Args:
            ioDurationMs: I/O wait duration in milliseconds.
            cpuIterations: Number of CPU computation iterations.
            
        Returns:
            Callable with mixed I/O and CPU work.
        """
        def task():
            # Simulate network receive
            time.sleep(ioDurationMs / 1000.0)
            
            # CPU computation
            result = 0.0
            for i in range(cpuIterations):
                result += math.sin(i) * math.cos(i)
            
            return result
        
        return task
    
    @staticmethod
    def ragPipelineTask(
        networkLatencyMs: float = 10.0,
        vectorDbLatencyMs: float = 200.0,
        llmLatencyMs: float = 500.0,
        tokenizationIterations: int = 10000,
        rerankingIterations: int = 20000,
    ) -> Callable[[], dict]:
        """
        Simulate a complete RAG (Retrieval-Augmented Generation) pipeline.
        
        Stages:
        1. Network receive (I/O)
        2. Tokenization (CPU)
        3. Vector DB query (I/O)
        4. Re-ranking (CPU)
        5. LLM generation (I/O - waiting for API response)
        
        Args:
            networkLatencyMs: Initial network latency.
            vectorDbLatencyMs: Vector database query latency.
            llmLatencyMs: LLM API call latency.
            tokenizationIterations: CPU work for tokenization.
            rerankingIterations: CPU work for re-ranking.
            
        Returns:
            Callable simulating RAG pipeline.
        """
        def task():
            stages = {}
            
            # Stage 1: Network receive
            stageStart = time.time()
            time.sleep(networkLatencyMs / 1000.0)
            stages["networkReceive"] = time.time() - stageStart
            
            # Stage 2: Tokenization (CPU-bound)
            stageStart = time.time()
            result = 0.0
            for i in range(tokenizationIterations):
                result += math.sin(i) * math.cos(i)
            stages["tokenization"] = time.time() - stageStart
            
            # Stage 3: Vector DB query (I/O-bound)
            stageStart = time.time()
            time.sleep(vectorDbLatencyMs / 1000.0)
            stages["vectorDb"] = time.time() - stageStart
            
            # Stage 4: Re-ranking (CPU-bound)
            stageStart = time.time()
            for i in range(rerankingIterations):
                result += math.sin(i) * math.cos(i)
            stages["reranking"] = time.time() - stageStart
            
            # Stage 5: LLM generation (I/O-bound)
            stageStart = time.time()
            time.sleep(llmLatencyMs / 1000.0)
            stages["llmGeneration"] = time.time() - stageStart
            
            return {
                "result": result,
                "stages": stages,
            }
        
        return task
    
    @staticmethod
    def fibonacciTask(n: int = 30) -> Callable[[], int]:
        """
        Create a pure Python recursive Fibonacci task.
        
        This holds the GIL for 100% of execution time, making it ideal
        for testing GIL saturation behavior.
        
        Args:
            n: Fibonacci number to compute.
            
        Returns:
            Callable computing nth Fibonacci number.
        """
        def fib(x: int) -> int:
            if x <= 1:
                return x
            return fib(x - 1) + fib(x - 2)
        
        def task():
            return fib(n)
        
        return task
    
    @staticmethod
    def variableLatencyTask(
        minMs: float = 10.0,
        maxMs: float = 100.0,
        cpuFraction: float = 0.2
    ) -> Callable[[], float]:
        """
        Create a task with variable latency.
        
        Useful for simulating real-world workloads where request
        processing time varies.
        
        Args:
            minMs: Minimum task duration.
            maxMs: Maximum task duration.
            cpuFraction: Fraction of time spent on CPU (rest is I/O).
            
        Returns:
            Callable with variable execution time.
        """
        def task():
            totalMs = random.uniform(minMs, maxMs)
            cpuMs = totalMs * cpuFraction
            ioMs = totalMs - cpuMs
            
            # I/O portion
            time.sleep(ioMs / 1000.0)
            
            # CPU portion
            iterations = int(cpuMs * 1000)  # Approximate scaling
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
    """
    
    def __init__(self, ratePerSecond: float):
        """
        Initialize the arrival generator.
        
        Args:
            ratePerSecond: Average number of arrivals per second (lambda).
        """
        self.ratePerSecond = ratePerSecond
        self.lastArrivalTime = time.time()
    
    def getNextInterarrivalTime(self) -> float:
        """
        Get the time until the next arrival.
        
        Returns:
            Time in seconds until next arrival (exponentially distributed).
        """
        return random.expovariate(self.ratePerSecond)
    
    def waitForNextArrival(self) -> None:
        """Wait until the next arrival time."""
        waitTime = self.getNextInterarrivalTime()
        time.sleep(waitTime)


class BurstArrivalGenerator:
    """
    Generate bursty traffic patterns.
    
    Alternates between high-rate bursts and quiet periods to
    test the executor's response to sudden load changes.
    """
    
    def __init__(
        self,
        burstRatePerSecond: float = 100.0,
        quietRatePerSecond: float = 10.0,
        burstDurationSec: float = 5.0,
        quietDurationSec: float = 10.0,
    ):
        """
        Initialize the burst generator.
        
        Args:
            burstRatePerSecond: Request rate during bursts.
            quietRatePerSecond: Request rate during quiet periods.
            burstDurationSec: Duration of each burst.
            quietDurationSec: Duration of each quiet period.
        """
        self.burstRate = burstRatePerSecond
        self.quietRate = quietRatePerSecond
        self.burstDuration = burstDurationSec
        self.quietDuration = quietDurationSec
        
        self.cycleStart = time.time()
        self.inBurst = True
    
    def getCurrentRate(self) -> float:
        """Get the current arrival rate."""
        elapsed = time.time() - self.cycleStart
        cycleDuration = self.burstDuration + self.quietDuration
        
        positionInCycle = elapsed % cycleDuration
        
        if positionInCycle < self.burstDuration:
            return self.burstRate
        else:
            return self.quietRate
    
    def getNextInterarrivalTime(self) -> float:
        """Get time until next arrival based on current rate."""
        rate = self.getCurrentRate()
        return random.expovariate(rate) if rate > 0 else 1.0
