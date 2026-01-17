# CodeBase Index

A comprehensive guide to the BetaPool library codebase structure and navigation.

**Author:** Mridankan Mandal

## Table of Contents:

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Module Reference](#module-reference)
4. [Class Reference](#class-reference)
5. [Architecture](#architecture)

## Overview:

BetaPool is a Python library implementing the adaptive thread pool algorithm described in the paper "GIL Bottleneck Mitigation for Pythonic Microservices."

Key features:

- Adaptive scaling based on blocking ratio (beta).
- GIL Safety Veto mechanism to prevent scaling during CPU-bound workloads.
- Compatible API with concurrent.futures.ThreadPoolExecutor.
- Built-in metrics collection and experiment logging.

Research paper: https://arxiv.org/abs/2601.10582

## Directory Structure:

```
BetaPool/
    betapool/                  # Main library package.
        __init__.py            # Public API exports.
        executor.py            # Core executor implementation.
        metrics.py             # Metrics collection system.
        workloads.py           # Workload generators for testing.
        py.typed               # PEP 561 marker for type hints.
        README.md              # Library-specific documentation.
        tests/                 # Unit tests.
            __init__.py
            test_betapool.py   # All unit tests.
    LICENSE                    # MIT License.
    README.md                  # Main repository documentation.
    Usage.md                   # Comprehensive usage guide.
    CodeBaseIndex.md           # This file.
    pyproject.toml             # Package configuration and dependencies.
    requirements.txt           # Minimal dependencies.
```

## Module Reference:

### betapool/__init__.py:

Purpose: Defines the public API of the library.

Exports:

- `AdaptiveThreadPoolExecutor`: Main adaptive executor class.
- `StaticThreadPoolExecutor`: Baseline static executor for comparison.
- `ControllerConfig`: Configuration dataclass for the adaptive controller.
- `TaskMetrics`: Dataclass representing metrics for a single task.
- `MetricsCollector`: Class for collecting and aggregating task metrics.
- `AggregatedMetrics`: Dataclass for aggregated window metrics.
- `WorkloadGenerator`: Factory class for creating test workloads.
- `PoissonArrivalGenerator`: Generator for steady-state arrivals.
- `BurstArrivalGenerator`: Generator for bursty traffic patterns.
- `__version__`: Library version string ("1.0.0").

### betapool/executor.py:

Purpose: Core implementation of the adaptive thread pool executor.

Key components:

- `ControllerConfig`: Configuration parameters for the adaptive controller.
- `ControllerState`: Internal state tracking for the controller thread.
- `AdaptiveThreadPoolExecutor`: Main class with adaptive scaling logic.
- `StaticThreadPoolExecutor`: Baseline executor without adaptive scaling.

Implementation details:

- Uses a background controller thread to monitor metrics.
- Calculates blocking ratio as `beta = 1 - (cpu_time / wall_time)`.
- Implements GIL Safety Veto when `beta < beta_low_threshold`.
- Scales up when `beta > beta_high_threshold` and `CPU < cpu_upper_threshold`.
- Wraps tasks with instrumentation to measure cpu_time and wall_time.

### betapool/metrics.py:

Purpose: Task metrics collection and aggregation system.

Key components:

- `TaskMetrics`: Dataclass storing individual task measurements.
- `MetricsCollector`: Rolling window collector for real-time analysis.
- `AggregatedMetrics`: Dataclass for window-based statistics.

Implementation details:

- Uses thread-safe collections with locks.
- Maintains rolling window of recent metrics (default 100).
- Calculates throughput, latency percentiles (P50, P95, P99).
- Tracks blocking ratio statistics (mean and standard deviation).

### betapool/workloads.py:

Purpose: Test workload generators for benchmarking and experiments.

Key components:

- `WorkloadGenerator`: Factory class with static methods for workloads.
- `PoissonArrivalGenerator`: Generates inter-arrival times from Poisson process.
- `BurstArrivalGenerator`: Alternates between high and low arrival rates.

Workload types:

- `io_task()`: Pure I/O simulation (sleep-based).
- `cpu_task_python()`: Pure CPU using math operations.
- `cpu_task_numpy()`: CPU using NumPy (releases GIL).
- `mixed_task()`: Combination of I/O and CPU.
- `rag_pipeline_task()`: Simulates RAG inference pipeline.
- `fibonacci_task()`: Recursive CPU task (extreme GIL holding).
- `variable_latency_task()`: Randomized latency with CPU fraction.

### betapool/tests/test_betapool.py:

Purpose: Unit tests for all library components.

Test coverage:

- Executor creation and configuration.
- Task submission and result retrieval.
- Metrics collection accuracy.
- GIL Safety Veto behavior.
- Scaling decisions.
- Workload generator functionality.

## Class Reference:

### ControllerConfig:

Dataclass for configuring the adaptive controller.

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `monitor_interval_sec` | float | 0.5 | Interval between metric evaluations. |
| `beta_high_threshold` | float | 0.7 | Blocking ratio threshold for scale-up. |
| `beta_low_threshold` | float | 0.3 | Blocking ratio threshold for GIL Safety Veto. |
| `scale_up_step` | int | 2 | Threads to add when scaling up. |
| `scale_down_step` | int | 1 | Threads to remove when scaling down. |
| `cpu_upper_threshold` | float | 85.0 | CPU percent to prevent scale-up. |
| `cpu_lower_threshold` | float | 50.0 | CPU percent for potential scale-down. |
| `stabilization_window_sec` | float | 2.0 | Minimum time between scaling decisions. |
| `warmup_task_count` | int | 10 | Tasks to complete before scaling. |

### AdaptiveThreadPoolExecutor:

Main executor class implementing adaptive scaling.

Constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_workers` | int | required | Minimum thread count. |
| `max_workers` | int | required | Maximum thread count. |
| `config` | ControllerConfig | None | Controller configuration. |
| `enable_logging` | bool | False | Enable debug logging. |

Methods:

| Method | Return Type | Description |
|--------|-------------|-------------|
| `submit(fn, *args, **kwargs)` | Future | Submit a task for execution. |
| `map(fn, *iterables, timeout, chunksize)` | Iterator | Apply function to iterables. |
| `get_current_thread_count()` | int | Get current number of threads. |
| `get_metrics()` | dict | Get current executor metrics. |
| `get_experiment_log()` | list[dict] | Get detailed experiment data. |
| `get_decision_history()` | list[dict] | Get scaling decision history. |
| `shutdown(wait)` | None | Shut down the executor. |

### StaticThreadPoolExecutor:

Baseline executor without adaptive scaling.

Constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workers` | int | required | Fixed number of threads. |

Methods: Same as AdaptiveThreadPoolExecutor.

### TaskMetrics:

Dataclass for individual task metrics.

Fields:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | int | Unique task identifier. |
| `wall_time` | float | Total elapsed time (seconds). |
| `cpu_time` | float | CPU time consumed (seconds). |
| `blocking_ratio` | float | Calculated beta value. |
| `timestamp` | float | Unix timestamp of completion. |
| `success` | bool | Whether task completed successfully. |
| `error_message` | str or None | Error message if failed. |

### MetricsCollector:

Class for collecting and aggregating metrics.

Constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | int | 100 | Number of recent metrics to keep. |
| `max_history` | int | 10000 | Maximum history size. |

Methods:

| Method | Return Type | Description |
|--------|-------------|-------------|
| `record(metrics)` | None | Record a TaskMetrics instance. |
| `get_recent_blocking_ratio()` | float | Get average beta over window. |
| `get_blocking_ratio_std()` | float | Get standard deviation of beta. |
| `get_throughput()` | float | Get tasks per second. |
| `get_latency_percentiles()` | dict | Get P50, P95, P99 latencies. |
| `get_aggregated_metrics()` | AggregatedMetrics | Get all aggregated stats. |
| `reset()` | None | Clear all collected metrics. |
| `export_to_list()` | list[dict] | Export all metrics as dicts. |

### WorkloadGenerator:

Factory class with static methods for workloads.

Static methods:

| Method | Parameters | Description |
|--------|------------|-------------|
| `io_task` | duration_ms | Returns callable for I/O simulation. |
| `cpu_task_python` | iterations | Returns callable for Python CPU work. |
| `cpu_task_numpy` | matrix_size | Returns callable for NumPy CPU work. |
| `mixed_task` | io_duration_ms, cpu_iterations | Returns callable for mixed workload. |
| `rag_pipeline_task` | network_latency_ms, vector_db_latency_ms, llm_latency_ms, tokenization_iterations, reranking_iterations | Returns callable for RAG simulation. |
| `fibonacci_task` | n | Returns callable for Fibonacci calculation. |
| `variable_latency_task` | min_ms, max_ms, cpu_fraction | Returns callable with random latency. |

### PoissonArrivalGenerator:

Generator for steady-state arrivals.

Constructor parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `rate_per_second` | float | Average arrival rate. |

Methods:

| Method | Return Type | Description |
|--------|-------------|-------------|
| `wait_for_next_arrival()` | None | Sleep until next arrival time. |
| `get_inter_arrival_time()` | float | Get time until next arrival. |

### BurstArrivalGenerator:

Generator for bursty traffic patterns.

Constructor parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `burst_rate_per_second` | float | Arrival rate during burst. |
| `quiet_rate_per_second` | float | Arrival rate during quiet period. |
| `burst_duration_sec` | float | Length of burst period. |
| `quiet_duration_sec` | float | Length of quiet period. |

Methods:

| Method | Return Type | Description |
|--------|-------------|-------------|
| `wait_for_next_arrival()` | None | Sleep until next arrival time. |
| `is_burst_phase()` | bool | Check if currently in burst phase. |

## Architecture:

High-level architecture:

```
                    User Code
                        |
                        v
        +-------------------------------+
        |   AdaptiveThreadPoolExecutor  |
        |                               |
        |  +-----------------------+    |
        |  |   Controller Thread   |    |
        |  |   - Monitor metrics   |    |
        |  |   - Make decisions    |    |
        |  |   - Adjust pool size  |    |
        |  +-----------------------+    |
        |             |                 |
        |             v                 |
        |  +-----------------------+    |
        |  |   MetricsCollector    |    |
        |  |   - Record tasks      |    |
        |  |   - Calculate beta    |    |
        |  |   - Track throughput  |    |
        |  +-----------------------+    |
        |             |                 |
        |             v                 |
        |  +-----------------------+    |
        |  |   ThreadPoolExecutor  |    |
        |  |   (stdlib base)       |    |
        |  |   - Worker threads    |    |
        |  |   - Task queue        |    |
        |  +-----------------------+    |
        +-------------------------------+
```

Scaling decision flow:

```
    Metrics Window
         |
         v
    Calculate beta = 1 - (cpu_time / wall_time)
         |
         v
    beta < 0.3? ----YES----> GIL Safety Veto (scale down)
         |
         NO
         |
         v
    beta > 0.7? ----NO-----> Hold (no action)
         |
         YES
         |
         v
    CPU < 85%? ----NO-----> Hold (CPU saturated)
         |
         YES
         |
         v
    Scale Up (+scale_up_step threads)
```

Thread safety:

All public methods are thread-safe:

- Metrics recording uses locks.
- Pool size changes are atomic.
- Decision history uses thread-safe collections.

Performance considerations:

- Controller thread overhead is minimal (runs every 0.5s by default).
- Task instrumentation adds approximately 0.1ms overhead per task.
- Memory usage scales with history size (configurable).
