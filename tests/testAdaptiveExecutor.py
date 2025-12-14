# Unit Tests for Adaptive Thread Pool Executor
# Validates correctness of core functionality and metrics collection.

import unittest
import time
import math
from concurrent.futures import Future
from typing import List
import threading

from src.adaptive_executor import (
    AdaptiveThreadPoolExecutor,
    StaticThreadPoolExecutor,
    ControllerConfig,
)
from src.metrics import TaskMetrics, MetricsCollector
from src.workloads import WorkloadGenerator


class TestTaskMetrics(unittest.TestCase):
    """Tests for TaskMetrics dataclass."""
    
    def testBlockingRatioClamp(self):
        """Blocking ratio should be clamped to [0, 1] range."""
        # Test value above 1
        metrics = TaskMetrics(
            taskId="test-1",
            wallTime=1.0,
            cpuTime=0.5,
            blockingRatio=1.5,  # Invalid, should clamp
            timestamp=time.time(),
        )
        self.assertEqual(metrics.blockingRatio, 1.0)
        
        # Test value below 0
        metrics = TaskMetrics(
            taskId="test-2",
            wallTime=1.0,
            cpuTime=0.5,
            blockingRatio=-0.5,  # Invalid, should clamp
            timestamp=time.time(),
        )
        self.assertEqual(metrics.blockingRatio, 0.0)
    
    def testValidMetrics(self):
        """Valid metrics should be stored correctly."""
        metrics = TaskMetrics(
            taskId="test-3",
            wallTime=1.0,
            cpuTime=0.3,
            blockingRatio=0.7,
            timestamp=12345.0,
            success=True,
        )
        
        self.assertEqual(metrics.taskId, "test-3")
        self.assertEqual(metrics.wallTime, 1.0)
        self.assertEqual(metrics.cpuTime, 0.3)
        self.assertEqual(metrics.blockingRatio, 0.7)
        self.assertTrue(metrics.success)


class TestMetricsCollector(unittest.TestCase):
    """Tests for MetricsCollector class."""
    
    def testRecordAndRetrieve(self):
        """Should correctly record and retrieve metrics."""
        collector = MetricsCollector(windowSize=10)
        
        # Record some metrics
        for i in range(5):
            metrics = TaskMetrics(
                taskId=f"task-{i}",
                wallTime=0.1,
                cpuTime=0.05,
                blockingRatio=0.5,
                timestamp=time.time(),
            )
            collector.record(metrics)
        
        self.assertEqual(collector.totalTasks, 5)
    
    def testRollingAverage(self):
        """Should compute correct rolling average."""
        collector = MetricsCollector(windowSize=10)
        
        # Record metrics with known blocking ratios
        ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
        for i, ratio in enumerate(ratios):
            metrics = TaskMetrics(
                taskId=f"task-{i}",
                wallTime=0.1,
                cpuTime=0.1 * (1 - ratio),
                blockingRatio=ratio,
                timestamp=time.time(),
            )
            collector.record(metrics)
        
        expectedAvg = sum(ratios) / len(ratios)
        actualAvg = collector.getRecentBlockingRatio()
        
        self.assertAlmostEqual(actualAvg, expectedAvg, places=5)
    
    def testEmptyCollector(self):
        """Empty collector should return neutral default."""
        collector = MetricsCollector()
        
        # Should return 0.5 (neutral) when no data
        self.assertEqual(collector.getRecentBlockingRatio(), 0.5)
    
    def testThreadSafety(self):
        """Should handle concurrent access safely."""
        collector = MetricsCollector()
        numThreads = 10
        tasksPerThread = 100
        
        def recordTasks():
            for i in range(tasksPerThread):
                metrics = TaskMetrics(
                    taskId=f"task-{threading.current_thread().name}-{i}",
                    wallTime=0.001,
                    cpuTime=0.0005,
                    blockingRatio=0.5,
                    timestamp=time.time(),
                )
                collector.record(metrics)
        
        threads = [threading.Thread(target=recordTasks) for _ in range(numThreads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(collector.totalTasks, numThreads * tasksPerThread)


class TestWorkloadGenerator(unittest.TestCase):
    """Tests for WorkloadGenerator class."""
    
    def testIoTask(self):
        """I/O task should sleep for specified duration."""
        durationMs = 50.0
        task = WorkloadGenerator.ioTask(durationMs)
        
        start = time.time()
        task()
        elapsed = (time.time() - start) * 1000
        
        # Allow 10ms tolerance
        self.assertGreater(elapsed, durationMs - 10)
        self.assertLess(elapsed, durationMs + 50)
    
    def testCpuTaskPython(self):
        """CPU task should execute and return a result."""
        task = WorkloadGenerator.cpuTaskPython(iterations=10000)
        result = task()
        
        # Should return a numeric result
        self.assertIsInstance(result, float)
    
    def testCpuTaskNumpy(self):
        """NumPy CPU task should execute and return array."""
        try:
            import numpy as np
            task = WorkloadGenerator.cpuTaskNumpy(matrixSize=50)
            result = task()
            
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (50, 50))
        except ImportError:
            self.skipTest("NumPy not available")
    
    def testFibonacciTask(self):
        """Fibonacci task should compute correct value."""
        task = WorkloadGenerator.fibonacciTask(n=10)
        result = task()
        
        self.assertEqual(result, 55)  # fib(10) = 55


class TestStaticThreadPoolExecutor(unittest.TestCase):
    """Tests for StaticThreadPoolExecutor class."""
    
    def testBasicExecution(self):
        """Should execute tasks and return results."""
        with StaticThreadPoolExecutor(workers=4) as executor:
            future = executor.submit(lambda x: x * 2, 21)
            result = future.result(timeout=5.0)
            
            self.assertEqual(result, 42)
    
    def testMultipleTasks(self):
        """Should handle multiple concurrent tasks."""
        with StaticThreadPoolExecutor(workers=4) as executor:
            futures = [executor.submit(lambda x: x ** 2, i) for i in range(10)]
            results = [f.result(timeout=5.0) for f in futures]
            
            expected = [i ** 2 for i in range(10)]
            self.assertEqual(results, expected)
    
    def testMetricsCollection(self):
        """Should collect metrics for executed tasks."""
        with StaticThreadPoolExecutor(workers=4) as executor:
            futures = [executor.submit(time.sleep, 0.01) for _ in range(5)]
            for f in futures:
                f.result(timeout=5.0)
            
            metrics = executor.getMetrics()
            self.assertEqual(metrics["totalTasks"], 5)
            self.assertEqual(metrics["currentThreads"], 4)


class TestAdaptiveThreadPoolExecutor(unittest.TestCase):
    """Tests for AdaptiveThreadPoolExecutor class."""
    
    def testBasicExecution(self):
        """Should execute tasks correctly."""
        with AdaptiveThreadPoolExecutor(minWorkers=2, maxWorkers=8) as executor:
            future = executor.submit(lambda x: x + 1, 41)
            result = future.result(timeout=5.0)
            
            self.assertEqual(result, 42)
    
    def testContextManager(self):
        """Should work correctly as context manager."""
        executor = AdaptiveThreadPoolExecutor(minWorkers=2, maxWorkers=8)
        
        with executor:
            future = executor.submit(lambda: 42)
            self.assertEqual(future.result(timeout=5.0), 42)
        
        # Executor should be shut down after context exit
        # Submitting should raise or fail gracefully
    
    def testMinMaxWorkerValidation(self):
        """Should validate min/max worker parameters."""
        # minWorkers must be >= 1
        with self.assertRaises(ValueError):
            AdaptiveThreadPoolExecutor(minWorkers=0, maxWorkers=8)
        
        # maxWorkers must be >= minWorkers
        with self.assertRaises(ValueError):
            AdaptiveThreadPoolExecutor(minWorkers=10, maxWorkers=5)
    
    def testMetricsTracking(self):
        """Should track execution metrics."""
        config = ControllerConfig(
            monitorIntervalSec=0.1,
            warmupTaskCount=3,
        )
        
        with AdaptiveThreadPoolExecutor(
            minWorkers=2, maxWorkers=8, config=config
        ) as executor:
            # Submit some tasks
            futures = [executor.submit(time.sleep, 0.01) for _ in range(5)]
            for f in futures:
                f.result(timeout=5.0)
            
            metrics = executor.getMetrics()
            
            self.assertEqual(metrics["totalTasks"], 5)
            self.assertGreaterEqual(metrics["currentThreads"], 2)
            self.assertLessEqual(metrics["currentThreads"], 8)
    
    def testBlockingRatioComputation(self):
        """Should compute blocking ratio correctly for I/O tasks."""
        with AdaptiveThreadPoolExecutor(minWorkers=2, maxWorkers=8) as executor:
            # Submit I/O-bound tasks (high blocking ratio expected)
            futures = [executor.submit(time.sleep, 0.05) for _ in range(10)]
            for f in futures:
                f.result(timeout=5.0)
            
            time.sleep(0.5)  # Allow metrics to aggregate
            
            metrics = executor.getMetrics()
            
            # I/O tasks should have high blocking ratio (> 0.5)
            self.assertGreater(metrics["avgBlockingRatio"], 0.5)
    
    def testCpuBoundBlockingRatio(self):
        """Should compute lower blocking ratio for CPU tasks than I/O tasks."""
        # Run CPU-bound tasks
        with AdaptiveThreadPoolExecutor(minWorkers=2, maxWorkers=8) as cpuExecutor:
            def cpuTask():
                result = 0.0
                for i in range(100000):
                    result += math.sin(i)
                return result
            
            futures = [cpuExecutor.submit(cpuTask) for _ in range(20)]
            for f in futures:
                f.result(timeout=30.0)
            
            time.sleep(0.5)
            cpuMetrics = cpuExecutor.getMetrics()
        
        # Run I/O-bound tasks
        with AdaptiveThreadPoolExecutor(minWorkers=2, maxWorkers=8) as ioExecutor:
            futures = [ioExecutor.submit(time.sleep, 0.05) for _ in range(20)]
            for f in futures:
                f.result(timeout=30.0)
            
            time.sleep(0.5)
            ioMetrics = ioExecutor.getMetrics()
        
        # CPU tasks should have lower blocking ratio than I/O tasks
        # The absolute value depends on system load, but relative comparison is stable
        self.assertLess(
            cpuMetrics["avgBlockingRatio"],
            ioMetrics["avgBlockingRatio"],
            f"CPU blocking ratio ({cpuMetrics['avgBlockingRatio']:.2f}) should be "
            f"less than I/O blocking ratio ({ioMetrics['avgBlockingRatio']:.2f})"
        )
    
    def testMapFunction(self):
        """Should correctly implement map functionality."""
        with AdaptiveThreadPoolExecutor(minWorkers=2, maxWorkers=8) as executor:
            results = list(executor.map(lambda x: x ** 2, range(5)))
            
            self.assertEqual(results, [0, 1, 4, 9, 16])
    
    def testExceptionHandling(self):
        """Should handle task exceptions gracefully."""
        with AdaptiveThreadPoolExecutor(minWorkers=2, maxWorkers=8) as executor:
            def failingTask():
                raise ValueError("Test exception")
            
            future = executor.submit(failingTask)
            
            with self.assertRaises(ValueError):
                future.result(timeout=5.0)


class TestControllerConfig(unittest.TestCase):
    """Tests for ControllerConfig class."""
    
    def testDefaultValues(self):
        """Should have sensible default values."""
        config = ControllerConfig()
        
        self.assertEqual(config.monitorIntervalSec, 0.5)
        self.assertEqual(config.betaHighThreshold, 0.7)
        self.assertEqual(config.betaLowThreshold, 0.3)
        self.assertGreater(config.scaleUpStep, 0)
        self.assertGreater(config.scaleDownStep, 0)
    
    def testCustomValues(self):
        """Should accept custom configuration values."""
        config = ControllerConfig(
            monitorIntervalSec=1.0,
            betaHighThreshold=0.8,
            betaLowThreshold=0.2,
            scaleUpStep=4,
            scaleDownStep=2,
        )
        
        self.assertEqual(config.monitorIntervalSec, 1.0)
        self.assertEqual(config.betaHighThreshold, 0.8)
        self.assertEqual(config.betaLowThreshold, 0.2)
        self.assertEqual(config.scaleUpStep, 4)
        self.assertEqual(config.scaleDownStep, 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def testMixedWorkload(self):
        """Should handle mixed I/O and CPU workload."""
        config = ControllerConfig(
            monitorIntervalSec=0.2,
            warmupTaskCount=5,
        )
        
        with AdaptiveThreadPoolExecutor(
            minWorkers=2, maxWorkers=16, config=config
        ) as executor:
            # Submit mix of I/O and CPU tasks
            futures: List[Future] = []
            
            for i in range(20):
                if i % 2 == 0:
                    # I/O task
                    futures.append(executor.submit(time.sleep, 0.02))
                else:
                    # CPU task
                    def cpuWork():
                        result = 0.0
                        for j in range(10000):
                            result += math.sin(j)
                        return result
                    futures.append(executor.submit(cpuWork))
            
            # All tasks should complete
            for f in futures:
                f.result(timeout=10.0)
            
            metrics = executor.getMetrics()
            
            self.assertEqual(metrics["totalTasks"], 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
