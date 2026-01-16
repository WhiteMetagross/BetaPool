# Codebase Index

Navigation guide for the BetaPool repository.

**Author:** Mridankan Mandal

## Overview

This document provides a comprehensive guide to navigating the codebase, understanding the architecture, and locating specific functionality.

## Directory Structure

```
MandalSchedulingResearchAlgorithm/
├── README.md                 # Project overview and quick start
├── Usage.md                  # Detailed usage instructions
├── CodeBaseIndex.md          # This file
├── pyproject.toml            # Package configuration (pip install)
├── requirements.txt          # Python dependencies
│
├── betapool/                 # THE LIBRARY (pip-installable)
│   ├── __init__.py           # Public API exports
│   ├── executor.py           # Core executor implementation
│   ├── metrics.py            # Metrics and instrumentation
│   ├── workloads.py          # Workload generators
│   ├── py.typed              # Type hint marker
│   ├── README.md             # Library documentation
│   └── tests/                # Unit tests
│       ├── __init__.py
│       └── test_betapool.py
│
├── src/                      # Original research implementation
│   ├── __init__.py
│   ├── adaptiveExecutor.py   # Research version of executor
│   ├── gilSaturation.py      # GIL saturation experiments
│   ├── metrics.py            # Research metrics module
│   ├── visualization.py      # Figure generation
│   ├── workloads.py          # Research workload generators
│   └── workloadSwitching.py  # Workload switching experiments
│
├── experiments/              # Experiment scripts
│   ├── singleCoreBenchmark.py
│   ├── quadCoreBenchmark.py
│   ├── baselineComparison.py
│   ├── generateFigures.py
│   └── ...
│
├── docs/                     # Research paper
│   ├── paper.md              # Paper in Markdown
│   └── paper.tex             # Paper in LaTeX
│
├── results/                  # Experiment results (CSV, JSON)
├── figures/                  # Generated figures
├── docker/                   # Docker configurations
└── tests/                    # Original test suite
```

## Library Components

### betapool/__init__.py

**Purpose:** Package entry point and public API exports.

**Exports:**
- `AdaptiveThreadPoolExecutor` - Main adaptive executor class
- `StaticThreadPoolExecutor` - Baseline static executor
- `ControllerConfig` - Controller configuration
- `ControllerState` - Runtime state tracking
- `TaskMetrics` - Individual task metrics
- `MetricsCollector` - Metrics aggregation
- `AggregatedMetrics` - Aggregated statistics
- `WorkloadGenerator` - Workload factory
- `PoissonArrivalGenerator` - Poisson arrivals
- `BurstArrivalGenerator` - Bursty traffic patterns

### betapool/executor.py

**Purpose:** Core implementation of adaptive thread pool.

**Classes:**

| Class | Description |
|-------|-------------|
| `ControllerConfig` | Configuration parameters for the adaptive controller |
| `ControllerState` | Runtime state and decision history |
| `AdaptiveThreadPoolExecutor` | Main adaptive executor with GIL Safety Veto |
| `StaticThreadPoolExecutor` | Fixed-size thread pool for comparison |

**Key Methods (AdaptiveThreadPoolExecutor):**

| Method | Description |
|--------|-------------|
| `submit(fn, *args, **kwargs)` | Submit a task for execution |
| `map(fn, *iterables)` | Map function over iterables |
| `get_current_thread_count()` | Get current thread count |
| `get_metrics()` | Get current metrics summary |
| `get_decision_history()` | Get scaling decision history |
| `get_experiment_log()` | Get detailed experiment log |
| `shutdown(wait=True)` | Shutdown the executor |

**Internal Methods:**

| Method | Description |
|--------|-------------|
| `_monitor_loop()` | Background thread for scaling decisions |
| `_make_scaling_decision()` | Evaluate metrics and scale |
| `_resize_pool(new_size)` | Resize thread pool at runtime |
| `_wrap_task(fn, task_id)` | Wrap task for metrics capture |

### betapool/metrics.py

**Purpose:** Task metrics collection and aggregation.

**Classes:**

| Class | Description |
|-------|-------------|
| `TaskMetrics` | Container for individual task execution metrics |
| `AggregatedMetrics` | Aggregated metrics over time window |
| `MetricsCollector` | Thread-safe metrics collection |

**Key Methods (MetricsCollector):**

| Method | Description |
|--------|-------------|
| `record(metrics)` | Record a completed task's metrics |
| `get_recent_blocking_ratio(n)` | Average blocking ratio over n tasks |
| `get_throughput()` | Current throughput (tasks/sec) |
| `get_latency_percentiles(n)` | P50, P90, P99 latencies |
| `get_aggregated_metrics(n)` | Complete aggregated statistics |
| `reset()` | Clear all collected metrics |
| `export_to_list()` | Export metrics as list of dicts |

### betapool/workloads.py

**Purpose:** Workload generators for testing and benchmarking.

**Classes:**

| Class | Description |
|-------|-------------|
| `WorkloadGenerator` | Factory for creating workloads |
| `PoissonArrivalGenerator` | Poisson process arrivals |
| `BurstArrivalGenerator` | Bursty traffic patterns |

**WorkloadGenerator Methods:**

| Method | Description |
|--------|-------------|
| `io_task(duration_ms)` | I/O-bound task (releases GIL) |
| `cpu_task_python(iterations)` | CPU-bound task (holds GIL) |
| `cpu_task_numpy(matrix_size)` | NumPy task (releases GIL) |
| `mixed_task(io_ms, cpu_iter)` | Mixed I/O and CPU task |
| `rag_pipeline_task(...)` | RAG pipeline simulation |
| `fibonacci_task(n)` | Pure Python Fibonacci |
| `variable_latency_task(...)` | Variable latency task |

## Research Implementation (src/)

### src/adaptiveExecutor.py

Original research implementation with camelCase naming.

**Key Differences from Library:**
- Uses camelCase method names (e.g., `monitorIntervalSec`)
- Includes additional experiment logging
- Less optimized for production use

### src/gilSaturation.py

GIL saturation experiment implementation.

**Classes:**
- `GilExperimentConfig` - Experiment configuration
- `GilDataPoint` - Single data point
- `GilSaturationExperiment` - Complete experiment

**Functions:**
- `fibonacciRecursive(n)` - GIL-holding computation
- `cpuIntensivePythonTask(iterations)` - Pure Python CPU task
- `runExperimentC(...)` - Run GIL saturation experiment

### src/visualization.py

Publication figure generation.

**Classes:**
- `ExperimentAVisualizer` - Square wave stress test figures
- `ExperimentBVisualizer` - RAG pipeline figures
- `ExperimentCVisualizer` - GIL saturation figures

**Functions:**
- `generateAllFigures()` - Generate all publication figures

## Experiment Scripts

### experiments/singleCoreBenchmark.py

Single-core saturation cliff characterization.

**What it does:**
- Limits execution to single CPU core
- Tests thread counts from 1 to 2048
- Measures throughput and latency

### experiments/quadCoreBenchmark.py

Quad-core edge device benchmark.

**What it does:**
- Simulates Raspberry Pi 4 / Jetson Nano
- Tests OS-GIL Paradox on multi-core
- Compares GIL vs free-threading

### experiments/baselineComparison.py

Comparison of scheduling strategies.

**Strategies tested:**
- Static-Naive (256 threads)
- Static-Optimal (32/64 threads)
- Adaptive (our solution)

### experiments/generateFigures.py

Publication figure generation.

**Outputs:**
- `figures/fig1_saturation_cliff.png`
- `figures/fig2_latency_analysis.png`
- `figures/fig3_efficiency.png`
- `figures/fig4_solution_comparison.png`

## Configuration Files

### pyproject.toml

Package configuration for pip installation.

**Key Sections:**
- `[project]` - Package metadata
- `[project.dependencies]` - Required dependencies
- `[project.optional-dependencies]` - Optional deps (numpy, dev, etc.)
- `[tool.setuptools]` - Package discovery
- `[tool.pytest]` - Test configuration
- `[tool.mypy]` - Type checking
- `[tool.black]` - Code formatting

### requirements.txt

Development dependencies.

```
psutil>=5.9.0
numpy>=1.24.0
matplotlib>=3.7.0
pytest>=7.3.0
pytest-cov>=4.0.0
mypy>=1.0.0
black>=23.0.0
isort>=5.12.0
```

## Test Suite

### betapool/tests/test_betapool.py

Comprehensive unit tests for the library.

**Test Classes:**

| Class | Description |
|-------|-------------|
| `TestTaskMetrics` | TaskMetrics dataclass tests |
| `TestMetricsCollector` | MetricsCollector tests |
| `TestWorkloadGenerator` | Workload generator tests |
| `TestStaticThreadPoolExecutor` | Static executor tests |
| `TestAdaptiveThreadPoolExecutor` | Adaptive executor tests |
| `TestControllerConfig` | Configuration tests |
| `TestIntegration` | Integration tests |

**Running Tests:**

```bash
# Run all tests
pytest betapool/tests/ -v

# Run with coverage
pytest betapool/tests/ --cov=betapool

# Run specific test class
pytest betapool/tests/test_betapool.py::TestAdaptiveThreadPoolExecutor -v
```

## Algorithm Overview

### The Blocking Ratio (Beta)

```
beta = 1 - (cpu_time / wall_time)
```

- **beta near 1.0:** Thread is waiting (I/O-bound)
- **beta near 0.0:** Thread is computing (CPU-bound)

### GIL Safety Veto Logic

```python
def control_loop(queue_length, blocking_ratio):
    GIL_DANGER_ZONE = 0.3
    
    if queue_length > 0:
        if blocking_ratio > GIL_DANGER_ZONE:
            return "scale_up"    # Safe: threads are doing I/O
        else:
            return "hold"        # VETO: threads are fighting for GIL
    else:
        return "scale_down"      # No work, reduce threads
```

### Controller Decision Flow

```
┌─────────────────┐
│ Collect Metrics │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check Warmup    │──No──► Hold
└────────┬────────┘
         │ Yes
         ▼
┌─────────────────┐
│ Check Stabilize │──No──► Hold
└────────┬────────┘
         │ Yes
         ▼
┌─────────────────┐
│ beta > 0.7?     │──Yes──► Scale Up (if CPU < 85%)
└────────┬────────┘
         │ No
         ▼
┌─────────────────┐
│ beta < 0.3?     │──Yes──► Scale Down (GIL Veto)
└────────┬────────┘
         │ No
         ▼
┌─────────────────┐
│ CPU > 85%?      │──Yes──► Scale Down
└────────┬────────┘
         │ No
         ▼
       Hold
```

## Performance Characteristics

### Memory Overhead

| Component | Memory |
|-----------|--------|
| Per thread | ~8 KB stack |
| MetricsCollector | ~1 MB (10K history) |
| Controller state | ~100 KB |
| Total (64 threads) | ~2 MB |

### Latency Overhead

| Operation | Latency |
|-----------|---------|
| Task instrumentation | ~10 us |
| Metrics recording | ~1 us |
| Scaling decision | ~100 us |

### Throughput

| Configuration | Expected TPS |
|---------------|--------------|
| I/O-heavy (50ms sleep) | 500-2000 |
| Mixed workload | 1000-5000 |
| CPU-heavy | Limited by cores |

## Extension Points

### Custom Scaling Logic

Override `_make_scaling_decision()` in a subclass:

```python
class CustomAdaptiveExecutor(AdaptiveThreadPoolExecutor):
    def _make_scaling_decision(self):
        # Your custom logic here
        pass
```

### Custom Metrics

Extend `MetricsCollector`:

```python
class CustomMetricsCollector(MetricsCollector):
    def custom_metric(self):
        # Your custom metric
        pass
```

### Custom Workloads

Add to `WorkloadGenerator`:

```python
@staticmethod
def my_workload(param):
    def task():
        # Custom workload logic
        pass
    return task
```
