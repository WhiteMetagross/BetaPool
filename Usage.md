# 📘 Usage Guide

Comprehensive usage instructions for the BetaPool library.

**Author:** Mridankan Mandal

## 📑 Table of Contents

1. [Installation](#-installation)
2. [Basic Usage](#-basic-usage)
3. [Configuration](#-configuration)
4. [Monitoring and Metrics](#-monitoring-and-metrics)
5. [Workload Generators](#-workload-generators)
6. [Advanced Usage](#-advanced-usage)
7. [Best Practices](#-best-practices)
8. [Troubleshooting](#-troubleshooting)

## 📦 Installation

**Prerequisites:**

- Python 3.11 or higher.
- pip package manager.

**Install from Source:**

```bash
# Clone the repository.
git clone https://github.com/WhiteMetagross/BetaPool.git
cd BetaPool

# Install in development mode.
pip install -e .

# Or install with all optional dependencies.
pip install -e ".[all]"
```

**Install Optional Dependencies:**

```bash
# NumPy support for workload generators.
pip install betapool[numpy]

# Visualization support.
pip install betapool[visualization]

# Development tools.
pip install betapool[dev]
```

**Verify Installation:**

```python
import betapool
print(betapool.__version__)  # Should print "1.0.0".
```

## 🚀 Basic Usage

**Simple Task Execution:**

```python
from betapool import AdaptiveThreadPoolExecutor

def process_item(item):
    # Your processing logic here.
    return item * 2

# Use as a context manager (recommended).
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32) as executor:
    # Submit individual tasks.
    future = executor.submit(process_item, 42)
    result = future.result()
    print(f"Result: {result}")  # Output: Result: 84.
```

**Batch Processing with map():**

```python
from betapool import AdaptiveThreadPoolExecutor

def process_data(x):
    # Simulate some work.
    import time
    time.sleep(0.01)
    return x ** 2

data = list(range(100))

with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32) as executor:
    results = list(executor.map(process_data, data))
    print(f"Processed {len(results)} items")
```

**Mixed Workload Example:**

```python
from betapool import AdaptiveThreadPoolExecutor
import time
import math

def io_task():
    """Simulates an I/O-bound operation."""
    time.sleep(0.05)  # Network call, file I/O, etc.
    return "io_done"

def cpu_task():
    """Simulates a CPU-bound operation."""
    result = 0.0
    for i in range(100000):
        result += math.sin(i)
    return result

with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    futures = []
    
    # Submit mixed workload.
    for i in range(50):
        if i % 2 == 0:
            futures.append(executor.submit(io_task))
        else:
            futures.append(executor.submit(cpu_task))
    
    # Wait for all results.
    results = [f.result() for f in futures]
    
    # Check metrics.
    metrics = executor.get_metrics()
    print(f"Blocking ratio: {metrics['avg_blocking_ratio']:.2f}")
```

## ⚙️ Configuration

**ControllerConfig Parameters:**

```python
from betapool import AdaptiveThreadPoolExecutor, ControllerConfig

config = ControllerConfig(
    # How often the controller evaluates metrics (seconds).
    monitor_interval_sec=0.5,
    
    # Blocking ratio threshold for scaling up (I/O-bound detection).
    # Higher values mean more waiting on I/O.
    beta_high_threshold=0.7,
    
    # Blocking ratio threshold for scaling down (CPU-bound detection).
    # Lower values mean more CPU work.
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
    # Your workload.
    pass
```

**Configuration Presets:**

**I/O-Heavy Workload:**

```python
io_heavy_config = ControllerConfig(
    beta_high_threshold=0.8,   # More aggressive scaling for high I/O.
    beta_low_threshold=0.4,
    scale_up_step=4,           # Scale up faster.
    scale_down_step=1,
)
```

**CPU-Heavy Workload:**

```python
cpu_heavy_config = ControllerConfig(
    beta_high_threshold=0.6,   # More conservative.
    beta_low_threshold=0.2,
    scale_up_step=1,
    scale_down_step=2,         # Scale down faster when CPU-bound.
)
```

**Latency-Sensitive Workload:**

```python
latency_sensitive_config = ControllerConfig(
    monitor_interval_sec=0.2,          # More frequent checks.
    stabilization_window_sec=1.0,      # Faster response.
    warmup_task_count=5,               # Shorter warmup.
)
```

## 📊 Monitoring and Metrics

**Getting Current Metrics:**

```python
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    # Submit tasks.
    futures = [executor.submit(task, arg) for arg in args]
    
    # Get metrics at any time.
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

**Tracking Decision History:**

```python
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    # Run workload.
    for _ in range(100):
        executor.submit(task)
    
    # Get scaling decision history.
    history = executor.get_decision_history()
    
    for decision in history:
        print(f"Time: {decision['timestamp']:.2f}")
        print(f"  Threads: {decision['threads_before']} -> {decision['threads_after']}")
        print(f"  Blocking ratio: {decision['blocking_ratio']:.3f}")
        print(f"  CPU: {decision['cpu_percent']:.1f}%")
        print(f"  Decision: {decision['decision']}")
```

**Experiment Logging:**

```python
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    # Run workload.
    futures = [executor.submit(task, arg) for arg in args]
    for f in futures:
        f.result()
    
    # Get experiment log for analysis.
    log = executor.get_experiment_log()
    
    # Export to CSV for analysis.
    import csv
    with open("experiment_log.csv", "w", newline="") as f:
        if log:
            writer = csv.DictWriter(f, fieldnames=log[0].keys())
            writer.writeheader()
            writer.writerows(log)
```

## 🔧 Workload Generators

**Built-in Workload Types:**

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

**RAG Pipeline Simulation:**

```python
from betapool import WorkloadGenerator

# Simulates a complete RAG pipeline.
rag_task = WorkloadGenerator.rag_pipeline_task(
    network_latency_ms=10.0,      # Initial request receive.
    vector_db_latency_ms=200.0,   # Vector database query.
    llm_latency_ms=500.0,         # LLM API call.
    tokenization_iterations=10000, # CPU work for tokenization.
    reranking_iterations=20000,    # CPU work for reranking.
)

with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32) as executor:
    # Simulate 10 concurrent RAG requests.
    futures = [executor.submit(rag_task) for _ in range(10)]
    results = [f.result() for f in futures]
    
    for result in results:
        print(f"Stages: {result['stages']}")
```

**Arrival Patterns:**

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
    # Run for 30 seconds with Poisson arrivals.
    import time
    end_time = time.time() + 30.0
    
    while time.time() < end_time:
        poisson.wait_for_next_arrival()
        executor.submit(task)
```

## 🔬 Advanced Usage

**Comparison with Static Thread Pool:**

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

# Compare adaptive vs static.
adaptive_time, adaptive_metrics = benchmark(
    AdaptiveThreadPoolExecutor,
    min_workers=4,
    max_workers=64
)

static_time, static_metrics = benchmark(
    StaticThreadPoolExecutor,
    workers=32
)

print(f"Adaptive: {adaptive_time:.2f}s, "
      f"throughput={adaptive_metrics['throughput']:.1f}")
print(f"Static: {static_time:.2f}s, "
      f"throughput={static_metrics['throughput']:.1f}")
```

**Manual Shutdown:**

```python
from betapool import AdaptiveThreadPoolExecutor

executor = AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32)

try:
    futures = [executor.submit(task, arg) for arg in args]
    results = [f.result() for f in futures]
finally:
    executor.shutdown(wait=True)  # Wait for pending tasks.
    # or
    # executor.shutdown(wait=False)  # Don't wait.
```

**Integration with Flask:**

```python
from flask import Flask, request, jsonify
from betapool import AdaptiveThreadPoolExecutor

app = Flask(__name__)

# Create a global executor.
executor = AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32)

def heavy_task(data):
    # Process data.
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

if __name__ == "__main__":
    app.run()
```

## ✅ Best Practices

**1. Choose Appropriate Worker Limits:**

```python
import os

# min_workers: Usually equal to CPU cores for compute tasks.
# or slightly higher for I/O tasks.
min_workers = os.cpu_count() or 4

# max_workers: Depends on workload type.
# I/O-heavy: Can go much higher (50-200).
# CPU-heavy: Keep close to core count.
max_workers = min_workers * 4  # Adjust based on workload.

executor = AdaptiveThreadPoolExecutor(
    min_workers=min_workers,
    max_workers=max_workers
)
```

**2. Allow Warmup Period:**

```python
# The controller needs some tasks to measure before making decisions.
config = ControllerConfig(
    warmup_task_count=20,  # Wait for 20 tasks before scaling.
)
```

**3. Monitor in Production:**

```python
import logging

logging.basicConfig(level=logging.DEBUG)

executor = AdaptiveThreadPoolExecutor(
    min_workers=4,
    max_workers=32,
    enable_logging=True  # Enable debug logging.
)
```

**4. Handle Exceptions:**

```python
from concurrent.futures import Future

with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32) as executor:
    futures = [executor.submit(task, arg) for arg in args]
    
    for future in futures:
        try:
            result = future.result(timeout=30.0)
        except TimeoutError:
            print("Task timed out")
        except Exception as e:
            print(f"Task failed: {e}")
```

## 🔧 Troubleshooting

**High Memory Usage:**

If memory usage is high, reduce max_workers:

```python
# Use fewer threads.
executor = AdaptiveThreadPoolExecutor(
    min_workers=2,
    max_workers=16  # Reduced from default.
)
```

**Slow Scaling Response:**

If the executor is slow to respond to workload changes:

```python
config = ControllerConfig(
    monitor_interval_sec=0.2,      # More frequent checks.
    stabilization_window_sec=0.5,  # Shorter stabilization.
    warmup_task_count=5,           # Shorter warmup.
)
```

**Oscillating Thread Count:**

If thread count oscillates frequently:

```python
config = ControllerConfig(
    stabilization_window_sec=5.0,  # Longer stabilization.
    scale_up_step=1,               # Smaller steps.
    scale_down_step=1,
)
```

**psutil Not Available:**

If psutil is not installed:

```bash
pip install psutil
```

Or the executor will work without CPU monitoring:

```python
# Works without psutil, but CPU-based scaling disabled.
from betapool import AdaptiveThreadPoolExecutor
executor = AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32)
```
