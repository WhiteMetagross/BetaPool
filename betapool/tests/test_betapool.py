"""
Unit Tests for BetaPool Library

Validates correctness of core functionality and metrics collection.

Author: Mridankan Mandal
"""

import unittest
import time
import math
import threading
from typing import List
from concurrent.futures import Future

from betapool import (
    AdaptiveThreadPoolExecutor,
    StaticThreadPoolExecutor,
    ControllerConfig,
    TaskMetrics,
    MetricsCollector,
    WorkloadGenerator,
)


class TestTaskMetrics(unittest.TestCase):
    """Tests for TaskMetrics dataclass."""
    
    def test_blocking_ratio_clamp_high(self):
        """Blocking ratio above 1.0 should be clamped to 1.0."""
        metrics = TaskMetrics(
            task_id="test-1",
            wall_time=1.0,
            cpu_time=0.5,
            blocking_ratio=1.5,  # Invalid, should clamp
            timestamp=time.time(),
        )
        self.assertEqual(metrics.blocking_ratio, 1.0)
    
    def test_blocking_ratio_clamp_low(self):
        """Blocking ratio below 0.0 should be clamped to 0.0."""
        metrics = TaskMetrics(
            task_id="test-2",
            wall_time=1.0,
            cpu_time=0.5,
            blocking_ratio=-0.5,  # Invalid, should clamp
            timestamp=time.time(),
        )
        self.assertEqual(metrics.blocking_ratio, 0.0)
    
    def test_valid_metrics(self):
        """Valid metrics should be stored correctly."""
        metrics = TaskMetrics(
            task_id="test-3",
            wall_time=1.0,
            cpu_time=0.3,
            blocking_ratio=0.7,
            timestamp=12345.0,
            success=True,
        )
        
        self.assertEqual(metrics.task_id, "test-3")
        self.assertEqual(metrics.wall_time, 1.0)
        self.assertEqual(metrics.cpu_time, 0.3)
        self.assertEqual(metrics.blocking_ratio, 0.7)
        self.assertTrue(metrics.success)
    
    def test_to_dict(self):
        """to_dict should return correct dictionary representation."""
        metrics = TaskMetrics(
            task_id="test-4",
            wall_time=1.0,
            cpu_time=0.3,
            blocking_ratio=0.7,
            timestamp=12345.0,
            success=True,
        )
        
        d = metrics.to_dict()
        self.assertEqual(d["task_id"], "test-4")
        self.assertEqual(d["wall_time"], 1.0)
        self.assertEqual(d["blocking_ratio"], 0.7)


class TestMetricsCollector(unittest.TestCase):
    """Tests for MetricsCollector class."""
    
    def test_record_and_retrieve(self):
        """Should correctly record and retrieve metrics."""
        collector = MetricsCollector(window_size=10)
        
        for i in range(5):
            metrics = TaskMetrics(
                task_id=f"task-{i}",
                wall_time=0.1,
                cpu_time=0.05,
                blocking_ratio=0.5,
                timestamp=time.time(),
            )
            collector.record(metrics)
        
        self.assertEqual(collector.total_tasks, 5)
    
    def test_rolling_average(self):
        """Should compute correct rolling average."""
        collector = MetricsCollector(window_size=10)
        
        ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
        for i, ratio in enumerate(ratios):
            metrics = TaskMetrics(
                task_id=f"task-{i}",
                wall_time=0.1,
                cpu_time=0.1 * (1 - ratio),
                blocking_ratio=ratio,
                timestamp=time.time(),
            )
            collector.record(metrics)
        
        expected_avg = sum(ratios) / len(ratios)
        actual_avg = collector.get_recent_blocking_ratio()
        
        self.assertAlmostEqual(actual_avg, expected_avg, places=5)
    
    def test_empty_collector(self):
        """Empty collector should return neutral default."""
        collector = MetricsCollector()
        
        # Should return 0.5 (neutral) when no data
        self.assertEqual(collector.get_recent_blocking_ratio(), 0.5)
    
    def test_thread_safety(self):
        """Should handle concurrent access safely."""
        collector = MetricsCollector()
        num_threads = 10
        tasks_per_thread = 100
        
        def record_tasks():
            for i in range(tasks_per_thread):
                metrics = TaskMetrics(
                    task_id=f"task-{threading.current_thread().name}-{i}",
                    wall_time=0.001,
                    cpu_time=0.0005,
                    blocking_ratio=0.5,
                    timestamp=time.time(),
                )
                collector.record(metrics)
        
        threads = [threading.Thread(target=record_tasks) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(collector.total_tasks, num_threads * tasks_per_thread)
    
    def test_export_to_list(self):
        """Should export metrics as list of dictionaries."""
        collector = MetricsCollector()
        
        for i in range(3):
            metrics = TaskMetrics(
                task_id=f"task-{i}",
                wall_time=0.1,
                cpu_time=0.05,
                blocking_ratio=0.5,
                timestamp=time.time(),
            )
            collector.record(metrics)
        
        exported = collector.export_to_list()
        self.assertEqual(len(exported), 3)
        self.assertIsInstance(exported[0], dict)
        self.assertIn("task_id", exported[0])


class TestWorkloadGenerator(unittest.TestCase):
    """Tests for WorkloadGenerator class."""
    
    def test_io_task(self):
        """I/O task should sleep for specified duration."""
        duration_ms = 50.0
        task = WorkloadGenerator.io_task(duration_ms)
        
        start = time.time()
        task()
        elapsed = (time.time() - start) * 1000
        
        # Allow tolerance for timing variations
        self.assertGreater(elapsed, duration_ms - 10)
        self.assertLess(elapsed, duration_ms + 50)
    
    def test_cpu_task_python(self):
        """CPU task should execute and return a result."""
        task = WorkloadGenerator.cpu_task_python(iterations=10000)
        result = task()
        
        self.assertIsInstance(result, float)
    
    def test_fibonacci_task(self):
        """Fibonacci task should compute correct value."""
        task = WorkloadGenerator.fibonacci_task(n=10)
        result = task()
        
        self.assertEqual(result, 55)  # fib(10) = 55
    
    def test_mixed_task(self):
        """Mixed task should complete and return result."""
        task = WorkloadGenerator.mixed_task(
            io_duration_ms=20.0,
            cpu_iterations=1000
        )
        
        start = time.time()
        result = task()
        elapsed = (time.time() - start) * 1000
        
        self.assertIsInstance(result, float)
        self.assertGreater(elapsed, 15)  # Should take at least the I/O time


class TestStaticThreadPoolExecutor(unittest.TestCase):
    """Tests for StaticThreadPoolExecutor class."""
    
    def test_basic_execution(self):
        """Should execute tasks and return results."""
        with StaticThreadPoolExecutor(workers=4) as executor:
            future = executor.submit(lambda x: x * 2, 21)
            result = future.result(timeout=5.0)
            
            self.assertEqual(result, 42)
    
    def test_multiple_tasks(self):
        """Should handle multiple concurrent tasks."""
        with StaticThreadPoolExecutor(workers=4) as executor:
            futures = [executor.submit(lambda x: x ** 2, i) for i in range(10)]
            results = [f.result(timeout=5.0) for f in futures]
            
            expected = [i ** 2 for i in range(10)]
            self.assertEqual(results, expected)
    
    def test_metrics_collection(self):
        """Should collect metrics for executed tasks."""
        with StaticThreadPoolExecutor(workers=4) as executor:
            futures = [executor.submit(time.sleep, 0.01) for _ in range(5)]
            for f in futures:
                f.result(timeout=5.0)
            
            metrics = executor.get_metrics()
            self.assertEqual(metrics["total_tasks"], 5)
            self.assertEqual(metrics["current_threads"], 4)


class TestAdaptiveThreadPoolExecutor(unittest.TestCase):
    """Tests for AdaptiveThreadPoolExecutor class."""
    
    def test_basic_execution(self):
        """Should execute tasks correctly."""
        with AdaptiveThreadPoolExecutor(min_workers=2, max_workers=8) as executor:
            future = executor.submit(lambda x: x + 1, 41)
            result = future.result(timeout=5.0)
            
            self.assertEqual(result, 42)
    
    def test_context_manager(self):
        """Should work correctly as context manager."""
        executor = AdaptiveThreadPoolExecutor(min_workers=2, max_workers=8)
        
        with executor:
            future = executor.submit(lambda: 42)
            self.assertEqual(future.result(timeout=5.0), 42)
    
    def test_min_max_worker_validation(self):
        """Should validate min/max worker parameters."""
        # min_workers must be >= 1
        with self.assertRaises(ValueError):
            AdaptiveThreadPoolExecutor(min_workers=0, max_workers=8)
        
        # max_workers must be >= min_workers
        with self.assertRaises(ValueError):
            AdaptiveThreadPoolExecutor(min_workers=10, max_workers=5)
    
    def test_metrics_tracking(self):
        """Should track execution metrics."""
        config = ControllerConfig(
            monitor_interval_sec=0.1,
            warmup_task_count=3,
        )
        
        with AdaptiveThreadPoolExecutor(
            min_workers=2, max_workers=8, config=config
        ) as executor:
            futures = [executor.submit(time.sleep, 0.01) for _ in range(5)]
            for f in futures:
                f.result(timeout=5.0)
            
            metrics = executor.get_metrics()
            
            self.assertEqual(metrics["total_tasks"], 5)
            self.assertGreaterEqual(metrics["current_threads"], 2)
            self.assertLessEqual(metrics["current_threads"], 8)
    
    def test_blocking_ratio_computation_io(self):
        """Should compute high blocking ratio for I/O tasks."""
        with AdaptiveThreadPoolExecutor(min_workers=2, max_workers=8) as executor:
            # Submit I/O-bound tasks (high blocking ratio expected)
            futures = [executor.submit(time.sleep, 0.05) for _ in range(10)]
            for f in futures:
                f.result(timeout=5.0)
            
            time.sleep(0.5)  # Allow metrics to aggregate
            
            metrics = executor.get_metrics()
            
            # I/O tasks should have high blocking ratio (> 0.5)
            self.assertGreater(metrics["avg_blocking_ratio"], 0.5)
    
    def test_cpu_bound_blocking_ratio(self):
        """Should compute lower blocking ratio for CPU tasks than I/O tasks."""
        # Run CPU-bound tasks
        with AdaptiveThreadPoolExecutor(min_workers=2, max_workers=8) as cpu_executor:
            def cpu_task():
                result = 0.0
                for i in range(100000):
                    result += math.sin(i)
                return result
            
            futures = [cpu_executor.submit(cpu_task) for _ in range(20)]
            for f in futures:
                f.result(timeout=30.0)
            
            time.sleep(0.5)
            cpu_metrics = cpu_executor.get_metrics()
        
        # Run I/O-bound tasks
        with AdaptiveThreadPoolExecutor(min_workers=2, max_workers=8) as io_executor:
            futures = [io_executor.submit(time.sleep, 0.05) for _ in range(20)]
            for f in futures:
                f.result(timeout=30.0)
            
            time.sleep(0.5)
            io_metrics = io_executor.get_metrics()
        
        # CPU tasks should have lower blocking ratio than I/O tasks
        self.assertLess(
            cpu_metrics["avg_blocking_ratio"],
            io_metrics["avg_blocking_ratio"],
            f"CPU blocking ratio ({cpu_metrics['avg_blocking_ratio']:.2f}) should be "
            f"less than I/O blocking ratio ({io_metrics['avg_blocking_ratio']:.2f})"
        )
    
    def test_map_function(self):
        """Should correctly implement map functionality."""
        with AdaptiveThreadPoolExecutor(min_workers=2, max_workers=8) as executor:
            results = list(executor.map(lambda x: x ** 2, range(5)))
            
            self.assertEqual(results, [0, 1, 4, 9, 16])
    
    def test_exception_handling(self):
        """Should handle task exceptions gracefully."""
        with AdaptiveThreadPoolExecutor(min_workers=2, max_workers=8) as executor:
            def failing_task():
                raise ValueError("Test exception")
            
            future = executor.submit(failing_task)
            
            with self.assertRaises(ValueError):
                future.result(timeout=5.0)
    
    def test_get_decision_history(self):
        """Should record decision history."""
        config = ControllerConfig(
            monitor_interval_sec=0.1,
            warmup_task_count=3,
            stabilization_window_sec=0.1,
        )
        
        with AdaptiveThreadPoolExecutor(
            min_workers=2, max_workers=16, config=config
        ) as executor:
            # Submit enough tasks to trigger scaling decisions
            futures = [executor.submit(time.sleep, 0.02) for _ in range(20)]
            for f in futures:
                f.result(timeout=10.0)
            
            # Wait for controller to make decisions
            time.sleep(1.0)
            
            history = executor.get_decision_history()
            # History should have at least some entries
            self.assertIsInstance(history, list)


class TestControllerConfig(unittest.TestCase):
    """Tests for ControllerConfig class."""
    
    def test_default_values(self):
        """Should have sensible default values."""
        config = ControllerConfig()
        
        self.assertEqual(config.monitor_interval_sec, 0.5)
        self.assertEqual(config.beta_high_threshold, 0.7)
        self.assertEqual(config.beta_low_threshold, 0.3)
        self.assertGreater(config.scale_up_step, 0)
        self.assertGreater(config.scale_down_step, 0)
    
    def test_custom_values(self):
        """Should accept custom configuration values."""
        config = ControllerConfig(
            monitor_interval_sec=1.0,
            beta_high_threshold=0.8,
            beta_low_threshold=0.2,
            scale_up_step=4,
            scale_down_step=2,
        )
        
        self.assertEqual(config.monitor_interval_sec, 1.0)
        self.assertEqual(config.beta_high_threshold, 0.8)
        self.assertEqual(config.beta_low_threshold, 0.2)
        self.assertEqual(config.scale_up_step, 4)
        self.assertEqual(config.scale_down_step, 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_mixed_workload(self):
        """Should handle mixed I/O and CPU workload."""
        config = ControllerConfig(
            monitor_interval_sec=0.2,
            warmup_task_count=5,
        )
        
        with AdaptiveThreadPoolExecutor(
            min_workers=2, max_workers=16, config=config
        ) as executor:
            futures: List[Future] = []
            
            for i in range(20):
                if i % 2 == 0:
                    # I/O task
                    futures.append(executor.submit(time.sleep, 0.02))
                else:
                    # CPU task
                    def cpu_work():
                        result = 0.0
                        for j in range(10000):
                            result += math.sin(j)
                        return result
                    futures.append(executor.submit(cpu_work))
            
            # All tasks should complete
            for f in futures:
                f.result(timeout=10.0)
            
            metrics = executor.get_metrics()
            
            self.assertEqual(metrics["total_tasks"], 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
