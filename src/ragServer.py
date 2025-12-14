# Experiment B: RAG Pipeline Simulation
# Simulates a Retrieval-Augmented Generation pipeline with mixed I/O and CPU phases
# to demonstrate P99 latency improvements under realistic workloads.

import time
import csv
import os
import random
import math
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import Future, as_completed
import statistics
import psutil

from src.adaptive_executor import AdaptiveThreadPoolExecutor, StaticThreadPoolExecutor, ControllerConfig


@dataclass
class RagConfig:
    """Configuration for the RAG pipeline simulation."""
    # Latency parameters (in milliseconds)
    networkLatencyMs: float = 10.0
    vectorDbLatencyMs: float = 200.0
    llmLatencyMs: float = 500.0
    
    # CPU work parameters (iteration counts)
    tokenizationIterations: int = 10000
    rerankingIterations: int = 20000
    
    # Experiment parameters
    warmupRequests: int = 50
    measurementRequests: int = 500
    concurrentClients: int = 20
    
    # Output
    outputDir: str = "results/experiment_b"


@dataclass
class RequestMetrics:
    """Metrics for a single RAG request."""
    requestId: int
    totalLatencyMs: float
    networkLatencyMs: float
    tokenizationMs: float
    vectorDbMs: float
    rerankingMs: float
    llmGenerationMs: float
    success: bool


@dataclass
class ExperimentResults:
    """Aggregated results from an experiment run."""
    executorName: str
    totalRequests: int
    successfulRequests: int
    avgLatencyMs: float
    p50LatencyMs: float
    p90LatencyMs: float
    p99LatencyMs: float
    throughputRps: float
    avgCpuUtilization: float


def ragPipelineRequest(
    requestId: int,
    config: RagConfig,
) -> RequestMetrics:
    """
    Execute a single RAG pipeline request.
    
    This simulates the complete flow of a RAG query:
    1. Network receive - waiting for request data
    2. Tokenization - CPU-bound text processing
    3. Vector DB query - I/O wait for similarity search
    4. Re-ranking - CPU-bound relevance scoring
    5. LLM generation - I/O wait for language model response
    
    Args:
        requestId: Unique identifier for this request.
        config: Configuration parameters.
        
    Returns:
        RequestMetrics containing timing breakdown.
    """
    stageTimes = {}
    totalStart = time.time()
    
    try:
        # Stage 1: Network receive
        stageStart = time.time()
        time.sleep(config.networkLatencyMs / 1000.0)
        stageTimes["network"] = (time.time() - stageStart) * 1000
        
        # Stage 2: Tokenization (CPU-bound)
        stageStart = time.time()
        result = 0.0
        for i in range(config.tokenizationIterations):
            result += math.sin(i) * math.cos(i)
        stageTimes["tokenization"] = (time.time() - stageStart) * 1000
        
        # Stage 3: Vector DB query (I/O-bound)
        stageStart = time.time()
        # Add slight jitter to simulate real DB latency variance
        jitter = random.uniform(0.9, 1.1)
        time.sleep((config.vectorDbLatencyMs * jitter) / 1000.0)
        stageTimes["vectorDb"] = (time.time() - stageStart) * 1000
        
        # Stage 4: Re-ranking (CPU-bound)
        stageStart = time.time()
        for i in range(config.rerankingIterations):
            result += math.sin(i) * math.cos(i)
        stageTimes["reranking"] = (time.time() - stageStart) * 1000
        
        # Stage 5: LLM generation (I/O-bound)
        stageStart = time.time()
        jitter = random.uniform(0.8, 1.2)
        time.sleep((config.llmLatencyMs * jitter) / 1000.0)
        stageTimes["llm"] = (time.time() - stageStart) * 1000
        
        totalLatency = (time.time() - totalStart) * 1000
        
        return RequestMetrics(
            requestId=requestId,
            totalLatencyMs=totalLatency,
            networkLatencyMs=stageTimes["network"],
            tokenizationMs=stageTimes["tokenization"],
            vectorDbMs=stageTimes["vectorDb"],
            rerankingMs=stageTimes["reranking"],
            llmGenerationMs=stageTimes["llm"],
            success=True,
        )
        
    except Exception as e:
        totalLatency = (time.time() - totalStart) * 1000
        return RequestMetrics(
            requestId=requestId,
            totalLatencyMs=totalLatency,
            networkLatencyMs=stageTimes.get("network", 0.0),
            tokenizationMs=stageTimes.get("tokenization", 0.0),
            vectorDbMs=stageTimes.get("vectorDb", 0.0),
            rerankingMs=stageTimes.get("reranking", 0.0),
            llmGenerationMs=stageTimes.get("llm", 0.0),
            success=False,
        )


class RagServerSimulator:
    """
    Simulates a RAG server handling concurrent requests.
    
    This class manages request submission and result collection,
    simulating realistic load patterns for the executor.
    """
    
    def __init__(self, config: Optional[RagConfig] = None):
        """
        Initialize the simulator.
        
        Args:
            config: RAG configuration. Uses defaults if None.
        """
        self.config = config or RagConfig()
        os.makedirs(self.config.outputDir, exist_ok=True)
    
    def runLoadTest(
        self,
        executor,
        executorName: str,
        loadLevel: str = "medium"
    ) -> Tuple[List[RequestMetrics], ExperimentResults]:
        """
        Run a load test with the given executor.
        
        Args:
            executor: Thread pool executor to test.
            executorName: Name for logging.
            loadLevel: "low", "medium", or "high" affecting request rate.
            
        Returns:
            Tuple of (individual request metrics, aggregated results).
        """
        # Adjust concurrent clients based on load level
        loadMultipliers = {"low": 0.5, "medium": 1.0, "high": 2.0}
        multiplier = loadMultipliers.get(loadLevel, 1.0)
        effectiveClients = int(self.config.concurrentClients * multiplier)
        
        print(f"[{executorName}] Running {loadLevel} load test "
              f"with {effectiveClients} concurrent clients")
        
        # Warmup phase
        print(f"[{executorName}] Warmup: {self.config.warmupRequests} requests...")
        warmupFutures = []
        for i in range(self.config.warmupRequests):
            future = executor.submit(ragPipelineRequest, i, self.config)
            warmupFutures.append(future)
        
        # Wait for warmup completion
        for future in warmupFutures:
            try:
                future.result(timeout=30.0)
            except Exception:
                pass
        
        print(f"[{executorName}] Warmup complete. Starting measurement...")
        
        # Measurement phase
        measurementStart = time.time()
        cpuSamples = []
        requestMetrics: List[RequestMetrics] = []
        futures: List[Future] = []
        
        # CPU monitoring thread
        stopMonitor = threading.Event()
        
        def monitorCpu():
            while not stopMonitor.is_set():
                cpuSamples.append(psutil.cpu_percent(interval=0.5))
        
        monitorThread = threading.Thread(target=monitorCpu)
        monitorThread.start()
        
        # Submit requests with controlled concurrency
        pendingFutures = []
        requestId = 0
        
        while requestId < self.config.measurementRequests:
            # Maintain concurrent request count
            while len(pendingFutures) < effectiveClients and requestId < self.config.measurementRequests:
                future = executor.submit(ragPipelineRequest, requestId, self.config)
                pendingFutures.append((requestId, future))
                requestId += 1
            
            # Check for completed futures
            newPending = []
            for rid, future in pendingFutures:
                if future.done():
                    try:
                        metrics = future.result()
                        requestMetrics.append(metrics)
                    except Exception as e:
                        print(f"[{executorName}] Request {rid} failed: {e}")
                else:
                    newPending.append((rid, future))
            
            pendingFutures = newPending
            
            # Brief sleep to avoid busy-waiting
            if pendingFutures:
                time.sleep(0.01)
        
        # Wait for remaining requests
        for rid, future in pendingFutures:
            try:
                metrics = future.result(timeout=30.0)
                requestMetrics.append(metrics)
            except Exception as e:
                print(f"[{executorName}] Request {rid} failed: {e}")
        
        measurementDuration = time.time() - measurementStart
        
        # Stop CPU monitoring
        stopMonitor.set()
        monitorThread.join()
        
        # Calculate statistics
        successfulRequests = [m for m in requestMetrics if m.success]
        latencies = sorted([m.totalLatencyMs for m in successfulRequests])
        
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = min(f + 1, len(data) - 1)
            return data[f] + (data[c] - data[f]) * (k - f)
        
        results = ExperimentResults(
            executorName=executorName,
            totalRequests=len(requestMetrics),
            successfulRequests=len(successfulRequests),
            avgLatencyMs=statistics.mean(latencies) if latencies else 0.0,
            p50LatencyMs=percentile(latencies, 50),
            p90LatencyMs=percentile(latencies, 90),
            p99LatencyMs=percentile(latencies, 99),
            throughputRps=len(successfulRequests) / measurementDuration if measurementDuration > 0 else 0.0,
            avgCpuUtilization=statistics.mean(cpuSamples) if cpuSamples else 0.0,
        )
        
        print(f"[{executorName}] Results: {results.successfulRequests}/{results.totalRequests} "
              f"successful, P99={results.p99LatencyMs:.1f}ms, "
              f"throughput={results.throughputRps:.1f} rps")
        
        return requestMetrics, results
    
    def runFullExperiment(
        self,
        minWorkers: int = 4,
        maxWorkers: int = 64,
        staticSmallPool: int = 4,
        staticLargePool: int = 50,
    ) -> Dict[str, Dict[str, ExperimentResults]]:
        """
        Run the complete RAG experiment across all configurations and load levels.
        
        Args:
            minWorkers: Minimum workers for adaptive executor.
            maxWorkers: Maximum workers for adaptive executor.
            staticSmallPool: Worker count for small static baseline.
            staticLargePool: Worker count for large static baseline.
            
        Returns:
            Nested dictionary: results[executor_name][load_level] = ExperimentResults
        """
        results: Dict[str, Dict[str, ExperimentResults]] = {
            "adaptive": {},
            "static_small": {},
            "static_large": {},
        }
        
        allMetrics: Dict[str, List[RequestMetrics]] = {}
        
        loadLevels = ["low", "medium", "high"]
        
        for loadLevel in loadLevels:
            print(f"\n{'='*60}")
            print(f"Load Level: {loadLevel.upper()}")
            print("="*60)
            
            # Adaptive executor
            print(f"\nTesting Adaptive Executor (min={minWorkers}, max={maxWorkers})")
            config = ControllerConfig(
                monitorIntervalSec=0.5,
                betaHighThreshold=0.7,
                betaLowThreshold=0.3,
            )
            with AdaptiveThreadPoolExecutor(
                minWorkers=minWorkers,
                maxWorkers=maxWorkers,
                config=config,
            ) as executor:
                metrics, result = self.runLoadTest(executor, "Adaptive", loadLevel)
                results["adaptive"][loadLevel] = result
                allMetrics[f"adaptive_{loadLevel}"] = metrics
            
            # Small static pool
            print(f"\nTesting Static Executor (workers={staticSmallPool})")
            with StaticThreadPoolExecutor(workers=staticSmallPool) as executor:
                metrics, result = self.runLoadTest(executor, f"Static-{staticSmallPool}", loadLevel)
                results["static_small"][loadLevel] = result
                allMetrics[f"static_small_{loadLevel}"] = metrics
            
            # Large static pool
            print(f"\nTesting Static Executor (workers={staticLargePool})")
            with StaticThreadPoolExecutor(workers=staticLargePool) as executor:
                metrics, result = self.runLoadTest(executor, f"Static-{staticLargePool}", loadLevel)
                results["static_large"][loadLevel] = result
                allMetrics[f"static_large_{loadLevel}"] = metrics
        
        # Save results
        self.saveResults(results, allMetrics)
        
        return results
    
    def saveResults(
        self,
        results: Dict[str, Dict[str, ExperimentResults]],
        allMetrics: Dict[str, List[RequestMetrics]]
    ) -> None:
        """
        Save experiment results to CSV files.
        
        Args:
            results: Aggregated results dictionary.
            allMetrics: Individual request metrics dictionary.
        """
        # Save summary results
        summaryFile = os.path.join(self.config.outputDir, "rag_summary.csv")
        with open(summaryFile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "executor", "load_level", "total_requests", "successful_requests",
                "avg_latency_ms", "p50_latency_ms", "p90_latency_ms", "p99_latency_ms",
                "throughput_rps", "avg_cpu_utilization"
            ])
            
            for executorName, loadResults in results.items():
                for loadLevel, result in loadResults.items():
                    writer.writerow([
                        executorName, loadLevel, result.totalRequests,
                        result.successfulRequests, result.avgLatencyMs,
                        result.p50LatencyMs, result.p90LatencyMs, result.p99LatencyMs,
                        result.throughputRps, result.avgCpuUtilization
                    ])
        
        print(f"\nSaved summary to {summaryFile}")
        
        # Save detailed request metrics
        for key, metrics in allMetrics.items():
            detailFile = os.path.join(self.config.outputDir, f"rag_detail_{key}.csv")
            with open(detailFile, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "request_id", "total_latency_ms", "network_ms", "tokenization_ms",
                    "vector_db_ms", "reranking_ms", "llm_generation_ms", "success"
                ])
                
                for m in metrics:
                    writer.writerow([
                        m.requestId, m.totalLatencyMs, m.networkLatencyMs,
                        m.tokenizationMs, m.vectorDbMs, m.rerankingMs,
                        m.llmGenerationMs, m.success
                    ])
            
            print(f"Saved details to {detailFile}")


def runExperimentB(
    outputDir: str = "results/experiment_b"
) -> Dict[str, Dict[str, ExperimentResults]]:
    """
    Convenience function to run Experiment B.
    
    Args:
        outputDir: Directory for output files.
        
    Returns:
        Dictionary of experiment results.
    """
    config = RagConfig(outputDir=outputDir)
    simulator = RagServerSimulator(config)
    return simulator.runFullExperiment()


if __name__ == "__main__":
    print("Starting Experiment B: RAG Pipeline Simulation")
    print("This experiment tests P99 latency under realistic RAG workloads.")
    print("")
    
    results = runExperimentB()
    
    print("\nExperiment B complete. Results saved to results/experiment_b/")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: P99 Latency (ms) by Load Level")
    print("="*80)
    print(f"{'Executor':<20} {'Low':<15} {'Medium':<15} {'High':<15}")
    print("-"*80)
    
    for executor in ["adaptive", "static_small", "static_large"]:
        row = f"{executor:<20}"
        for load in ["low", "medium", "high"]:
            if load in results[executor]:
                row += f"{results[executor][load].p99LatencyMs:<15.1f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
