# Usage Guide

Comprehensive usage instructions for the BetaPool library.

**Author:** Mridankan Mandal

## Table of Contents:

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Configuration](#configuration)
4. [Monitoring and Metrics](#monitoring-and-metrics)
5. [Workload Generators](#workload-generators)
6. [Advanced Usage](#advanced-usage)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Installation:

Prerequisites:

- Python 3.11 or higher.
- pip package manager.

Install from source:

```bash
git clone https://github.com/WhiteMetagross/BetaPool.git
cd BetaPool
pip install -e .
```

Install with optional dependencies:

```bash
pip install -e ".[numpy]"    # NumPy workload generators.
pip install -e ".[dev]"      # Development tools (pytest, mypy, black).
pip install -e ".[all]"      # All dependencies.
```

Verify installation:

```python
import betapool
print(betapool.__version__)  # Should print "1.0.0".
```

## Basic Usage:

Simple task execution:

```python
from betapool import AdaptiveThreadPoolExecutor

def process_item(item):
    # Your processing logic here.
    return item * 2

# Use as a context manager (recommended).
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32) as executor:
    future = executor.submit(process_item, 42)
    result = future.result()
    print(f"Result: {result}")  # Output: Result: 84.
```

Batch processing with map():

```python
from betapool import AdaptiveThreadPoolExecutor

def process_data(x):
    import time
    time.sleep(0.01)  # Simulate I/O.
    return x ** 2

data = list(range(100))

with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32) as executor:
    results = list(executor.map(process_data, data))
    print(f"Processed {len(results)} items.")
```

Mixed workload example:

```python
from betapool import AdaptiveThreadPoolExecutor
import time
import math

def io_task():
    """Simulates an I/O-bound operation."""
    time.sleep(0.05)
    return "io_done"

def cpu_task():
    """Simulates a CPU-bound operation."""
    result = 0.0
    for i in range(100000):
        result += math.sin(i)
    return result

with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    futures = []
    
    for i in range(50):
        if i % 2 == 0:
            futures.append(executor.submit(io_task))
        else:
            futures.append(executor.submit(cpu_task))
    
    results = [f.result() for f in futures]
    
    metrics = executor.get_metrics()
    print(f"Blocking ratio: {metrics['avg_blocking_ratio']:.2f}")
```

## Configuration:

ControllerConfig parameters:

```python
from betapool import AdaptiveThreadPoolExecutor, ControllerConfig

config = ControllerConfig(
    # How often the controller evaluates metrics (seconds).
    monitor_interval_sec=0.5,
    
    # Blocking ratio threshold for scaling up (I/O-bound detection).
    # Values above this indicate threads are waiting on I/O.
    beta_high_threshold=0.7,
    
    # Blocking ratio threshold for scaling down (CPU-bound detection).
    # Values below this indicate threads are CPU-bound, triggering GIL Safety Veto.
    beta_low_threshold=0.3,
    
    # Number of threads to add when scaling up.
    scale_up_step=2,
    
    # Number of threads to remove when scaling down.
    scale_down_step=1,
    
    # CPU utilization threshold to prevent scaling up.
    cpu_upper_threshold=85.0,
    
    # CPU utilization threshold that may trigger scaling down.
    cpu_lower_threshold=50.0,
    
    # Minimum time between scaling decisions (seconds).
    stabilization_window_sec=2.0,
    
    # Minimum tasks to complete before making scaling decisions.
    warmup_task_count=10,
)

with AdaptiveThreadPoolExecutor(
    min_workers=4,
    max_workers=64,
    config=config
) as executor:
    pass
```

Configuration presets:

I/O-heavy workload:

```python
io_heavy_config = ControllerConfig(
    beta_high_threshold=0.8,   # More aggressive scaling for high I/O.
    beta_low_threshold=0.4,
    scale_up_step=4,           # Scale up faster.
    scale_down_step=1,
)
```

CPU-heavy workload:

```python
cpu_heavy_config = ControllerConfig(
    beta_high_threshold=0.6,   # More conservative.
    beta_low_threshold=0.2,
    scale_up_step=1,
    scale_down_step=2,         # Scale down faster when CPU-bound.
)
```

Latency-sensitive workload:

```python
latency_sensitive_config = ControllerConfig(
    monitor_interval_sec=0.2,          # More frequent checks.
    stabilization_window_sec=1.0,      # Faster response.
    warmup_task_count=5,               # Shorter warmup.
)
```

## Monitoring and Metrics:

Getting current metrics:

```python
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    futures = [executor.submit(task, arg) for arg in args]
    
    metrics = executor.get_metrics()
    
    print(f"Current threads: {metrics['current_threads']}")
    print(f"Min workers: {metrics['min_workers']}")
    print(f"Max workers: {metrics['max_workers']}")
    print(f"Total tasks: {metrics['total_tasks']}")
    print(f"Avg blocking ratio: {metrics['avg_blocking_ratio']:.3f}")
    print(f"Throughput: {metrics['throughput']:.1f} tasks/sec")
    print(f"P50 latency: {metrics['p50_latency']*1000:.1f} ms")
    print(f"P99 latency: {metrics['p99_latency']*1000:.1f} ms")
    print(f"Scale up count: {metrics['scale_up_count']}")
    print(f"Scale down count: {metrics['scale_down_count']}")
```

Tracking decision history:

```python
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    for _ in range(100):
        executor.submit(task)
    
    history = executor.get_decision_history()
    
    for decision in history:
        print(f"Time: {decision['timestamp']:.2f}")
        print(f"  Threads: {decision['threads_before']} -> {decision['threads_after']}")
        print(f"  Blocking ratio: {decision['blocking_ratio']:.3f}")
        print(f"  CPU: {decision['cpu_percent']:.1f}%")
        print(f"  Decision: {decision['decision']}")
```

Exporting experiment data:

```python
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    futures = [executor.submit(task, arg) for arg in args]
    for f in futures:
        f.result()
    
    log = executor.get_experiment_log()
    
    import csv
    with open("experiment_log.csv", "w", newline="") as f:
        if log:
            writer = csv.DictWriter(f, fieldnames=log[0].keys())
            writer.writeheader()
            writer.writerows(log)
```

## Workload Generators:

Built-in workload types:

```python
from betapool import WorkloadGenerator

# Pure I/O task (releases GIL during sleep).
io_task = WorkloadGenerator.io_task(duration_ms=50.0)

# Pure CPU task (holds GIL).
cpu_task = WorkloadGenerator.cpu_task_python(iterations=100000)

# NumPy CPU task (releases GIL during computation).
numpy_task = WorkloadGenerator.cpu_task_numpy(matrix_size=100)

# Mixed I/O and CPU task.
mixed_task = WorkloadGenerator.mixed_task(
    io_duration_ms=50.0,
    cpu_iterations=10000
)

# Fibonacci task (extreme GIL holding).
fib_task = WorkloadGenerator.fibonacci_task(n=30)

# Variable latency task.
var_task = WorkloadGenerator.variable_latency_task(
    min_ms=10.0,
    max_ms=100.0,
    cpu_fraction=0.2
)
```

RAG pipeline simulation:

```python
from betapool import WorkloadGenerator

rag_task = WorkloadGenerator.rag_pipeline_task(
    network_latency_ms=10.0,      # Initial request receive.
    vector_db_latency_ms=200.0,   # Vector database query.
    llm_latency_ms=500.0,         # LLM API call.
    tokenization_iterations=10000, # CPU work for tokenization.
    reranking_iterations=20000,    # CPU work for reranking.
)

with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32) as executor:
    futures = [executor.submit(rag_task) for _ in range(10)]
    results = [f.result() for f in futures]
    
    for result in results:
        print(f"Stages: {result['stages']}")
```

Arrival patterns:

```python
from betapool import (
    AdaptiveThreadPoolExecutor,
    PoissonArrivalGenerator,
    BurstArrivalGenerator,
    WorkloadGenerator,
)

# Poisson arrivals (steady stream).
poisson = PoissonArrivalGenerator(rate_per_second=50.0)

# Bursty arrivals.
bursty = BurstArrivalGenerator(
    burst_rate_per_second=100.0,
    quiet_rate_per_second=10.0,
    burst_duration_sec=5.0,
    quiet_duration_sec=10.0,
)

task = WorkloadGenerator.io_task(duration_ms=20.0)

with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32) as executor:
    import time
    end_time = time.time() + 30.0
    
    while time.time() < end_time:
        poisson.wait_for_next_arrival()
        executor.submit(task)
```

## Advanced Usage:

Comparison with static thread pool:

```python
from betapool import AdaptiveThreadPoolExecutor, StaticThreadPoolExecutor
import time

def benchmark(executor_class, **kwargs):
    start = time.time()
    
    with executor_class(**kwargs) as executor:
        futures = [executor.submit(task, arg) for arg in args]
        results = [f.result() for f in futures]
        metrics = executor.get_metrics()
    
    elapsed = time.time() - start
    return elapsed, metrics

adaptive_time, adaptive_metrics = benchmark(
    AdaptiveThreadPoolExecutor,
    min_workers=4,
    max_workers=64
)

static_time, static_metrics = benchmark(
    StaticThreadPoolExecutor,
    workers=32
)

print(f"Adaptive: {adaptive_time:.2f}s, throughput={adaptive_metrics['throughput']:.1f}")
print(f"Static: {static_time:.2f}s, throughput={static_metrics['throughput']:.1f}")
```

Manual shutdown:

```python
from betapool import AdaptiveThreadPoolExecutor

executor = AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32)

try:
    futures = [executor.submit(task, arg) for arg in args]
    results = [f.result() for f in futures]
finally:
    executor.shutdown(wait=True)  # Wait for pending tasks.
```

Integration with Flask:

```python
from flask import Flask, request, jsonify
from betapool import AdaptiveThreadPoolExecutor

app = Flask(__name__)
executor = AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32)

def heavy_task(data):
    return {"processed": data}

@app.route("/process", methods=["POST"])
def process():
    data = request.json
    future = executor.submit(heavy_task, data)
    result = future.result(timeout=30.0)
    return jsonify(result)

@app.route("/metrics")
def metrics():
    return jsonify(executor.get_metrics())

@app.teardown_appcontext
def shutdown_executor(exception=None):
    executor.shutdown(wait=False)
```

## Best Practices:

1. Choose appropriate worker limits:

```python
import os

# min_workers: Usually equal to CPU cores for compute tasks.
min_workers = os.cpu_count() or 4

# max_workers: Depends on workload type.
# I/O-heavy: Can go higher (50-200).
# CPU-heavy: Keep close to core count.
max_workers = min_workers * 4

executor = AdaptiveThreadPoolExecutor(
    min_workers=min_workers,
    max_workers=max_workers
)
```

2. Allow warmup period:

```python
config = ControllerConfig(
    warmup_task_count=20,  # Wait for 20 tasks before scaling.
)
```

3. Monitor in production:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

executor = AdaptiveThreadPoolExecutor(
    min_workers=4,
    max_workers=32,
    enable_logging=True
)
```

4. Handle exceptions:

```python
from concurrent.futures import Future

with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32) as executor:
    futures = [executor.submit(task, arg) for arg in args]
    
    for future in futures:
        try:
            result = future.result(timeout=30.0)
        except TimeoutError:
            print("Task timed out.")
        except Exception as e:
            print(f"Task failed: {e}")
```

## Troubleshooting:

High memory usage:

If memory usage is high, reduce max_workers:

```python
executor = AdaptiveThreadPoolExecutor(
    min_workers=2,
    max_workers=16  # Reduced from default.
)
```

Slow scaling response:

If the executor is slow to respond to workload changes:

```python
config = ControllerConfig(
    monitor_interval_sec=0.2,      # More frequent checks.
    stabilization_window_sec=0.5,  # Shorter stabilization.
    warmup_task_count=5,           # Shorter warmup.
)
```

Oscillating thread count:

If thread count oscillates frequently:

```python
config = ControllerConfig(
    stabilization_window_sec=5.0,  # Longer stabilization.
    scale_up_step=1,               # Smaller steps.
    scale_down_step=1,
)
```

psutil not available:

If psutil is not installed:

```bash
pip install psutil
```

The executor will work without psutil, but CPU-based scaling will be disabled.
