# Codebase Index:

This document provides a comprehensive overview of the project structure, file purposes, and key components for navigating the codebase.

---

## Project Overview:

This research project investigates GIL-induced concurrency thrashing in Python-based edge AI systems. The codebase includes:

- Benchmark scripts for characterizing the saturation cliff.
- An adaptive thread pool implementation that avoids the cliff.
- Platform-specific scripts for Raspberry Pi and Jetson Nano.
- Docker configurations for reproducible experiments.
- Publication-quality figure generation.

---

## Directory Structure:

```
edge-gil/
    docs/                       # Documentation and research paper.
        paper.md                # Full research paper.
        README.md               # Project overview and quick start.
        reproducibility.md      # Detailed reproducibility instructions.
    docker/                     # Docker configurations.
        Dockerfile              # Base image for experiments.
        Dockerfile.singleCore   # Single-core edge simulation.
        Dockerfile.quadCore     # Quad-core edge simulation.
        docker-compose.yml      # Orchestration for all experiments.
    experiments/                # Experiment scripts.
        singleCoreBenchmark.py  # Single-core saturation cliff benchmark.
        quadCoreBenchmark.py    # Quad-core saturation cliff benchmark.
        generateFigures.py      # Publication figure generator.
    figures/                    # Generated figures (PDF and PNG).
    platforms/                  # Platform-specific implementations.
        raspberryPiBenchmark.py # Raspberry Pi 4 native benchmark.
        jetsonNanoBenchmark.py  # NVIDIA Jetson Nano native benchmark.
    results/                    # Experiment output data (CSV files).
    src/                        # Core library modules.
        __init__.py             # Package initialization.
        adaptiveExecutor.py     # Adaptive thread pool implementation.
        metrics.py              # Task metrics and instrumentation.
        workloads.py            # Workload generators for experiments.
        visualization.py        # Visualization utilities.
        gilSaturation.py        # GIL saturation analysis utilities.
        ragServer.py            # Example RAG server implementation.
        workloadSwitching.py    # Workload phase switching experiments.
    tests/                      # Unit tests.
        __init__.py             # Test package initialization.
        testAdaptiveExecutor.py # Tests for adaptive executor.
    Usage.md                    # This usage guide.
    CodeBaseIndex.md            # This codebase index.
    requirements.txt            # Python dependencies.
```

---

## Core Modules:

### src/adaptiveExecutor.py:

The main contribution of this research. Implements the Metric-Driven Adaptive Thread Pool.

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `ControllerConfig` | Configuration parameters for the adaptive controller. |
| `ControllerState` | Runtime state tracking for scaling decisions. |
| `AdaptiveThreadPoolExecutor` | Main executor with adaptive sizing. |
| `StaticThreadPoolExecutor` | Baseline executor for comparison. |

**Key Methods:**

- `submit(fn, *args, **kwargs)` - Submit a task for execution.
- `_monitorLoop()` - Background thread for workload monitoring.
- `_makeScalingDecision()` - Implements the veto-based scaling logic.

**Usage Example:**

```python
from src.adaptiveExecutor import AdaptiveThreadPoolExecutor

with AdaptiveThreadPoolExecutor(minWorkers=4, maxWorkers=64) as executor:
    futures = [executor.submit(task, arg) for arg in args]
    results = [f.result() for f in futures]
```

### src/metrics.py:

Task instrumentation and metrics collection.

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `TaskMetrics` | Container for individual task execution data. |
| `AggregatedMetrics` | Aggregated metrics over time windows. |
| `MetricsCollector` | Thread-safe collector for rolling averages. |

**Key Metrics:**

- `wallTime` - Total elapsed time for task execution.
- `cpuTime` - CPU time consumed (via `time.thread_time()`).
- `blockingRatio` - Calculated as `1 - (cpuTime / wallTime)`.

### src/workloads.py:

Workload generators for controlled experiments.

**Key Methods:**

| Method | Description |
|--------|-------------|
| `ioTask(durationMs)` | Pure I/O workload using sleep. |
| `cpuTaskPython(iterations)` | CPU-bound task holding the GIL. |
| `cpuTaskNumpy(matrixSize)` | CPU-bound task releasing the GIL. |
| `mixedTask(ioDurationMs, cpuIterations)` | Combined CPU and I/O workload. |

---

## Experiment Scripts:

### experiments/singleCoreBenchmark.py:

Characterizes the saturation cliff on single-core edge devices.

**What it does:**

1. Pins execution to a single CPU core.
2. Runs mixed workload across thread counts from 1 to 2048.
3. Measures throughput, latency, and efficiency.
4. Saves results to CSV for analysis.

**Expected Output:**

- Peak throughput around 32 threads.
- 30-40% degradation at 2048 threads.

### experiments/quadCoreBenchmark.py:

Validates that the cliff persists on multi-core hardware.

**What it does:**

1. Pins execution to four CPU cores.
2. Runs the same benchmark as single-core.
3. Demonstrates the OS-GIL Paradox.

**Expected Output:**

- Peak throughput around 64 threads.
- 15-30% degradation at high thread counts.

### experiments/generateFigures.py:

Generates publication-quality figures from experiment data.

**Generated Figures:**

| Figure | Content |
|--------|---------|
| `fig1_saturation_cliff.pdf` | Main throughput vs thread count plot. |
| `fig2_latency_analysis.pdf` | P99 latency degradation. |
| `fig3_efficiency.pdf` | Per-thread efficiency curve. |
| `fig4_solution_comparison.pdf` | Adaptive vs naive strategies. |

---

## Platform Scripts:

### platforms/raspberryPiBenchmark.py:

Optimized for native execution on Raspberry Pi 4.

**Features:**

- Automatic hardware detection (CPU model, memory, architecture).
- ARM-optimized configuration.
- CSV output compatible with figure generator.

### platforms/jetsonNanoBenchmark.py:

Optimized for NVIDIA Jetson Nano.

**Features:**

- GPU availability detection.
- Thermal monitoring during benchmarks.
- L4T version detection for compatibility.

---

## Docker Configuration:

### docker/Dockerfile:

Base image with all dependencies installed.

### docker/Dockerfile.singleCore:

Configured for single-core experiments. Use with:

```bash
docker run --cpus="1.0" edge-gil:singlecore
```

### docker/Dockerfile.quadCore:

Configured for quad-core experiments. Use with:

```bash
docker run --cpus="4.0" edge-gil:quadcore
```

### docker/docker-compose.yml:

Orchestrates all experiments. Services:

- `singlecore` - Single-core benchmark.
- `quadcore` - Quad-core benchmark.
- `figures` - Figure generation.
- `all` - Run everything in sequence.

---

## Test Suite:

### tests/testAdaptiveExecutor.py:

Unit tests for the adaptive executor.

**Test Classes:**

| Class | Tests |
|-------|-------|
| `TestTaskMetrics` | Metric calculation and bounds checking. |
| `TestMetricsCollector` | Thread-safe collection and aggregation. |
| `TestAdaptiveExecutor` | Executor behavior and scaling decisions. |

**Running Tests:**

```bash
python -m pytest tests/ -v
```

---

## Data Files:

### results/ Directory:

Contains CSV files from experiment runs:

| File | Content |
|------|---------|
| `mixed_workload.csv` | Single-core mixed workload results. |
| `io_baseline.csv` | Single-core pure I/O baseline. |
| `quadcore_mixed_workload.csv` | Quad-core mixed workload results. |
| `quadcore_io_baseline.csv` | Quad-core pure I/O baseline. |
| `solution_comparison.csv` | Adaptive vs static strategy comparison. |

### CSV Schema:

Common columns across result files:

| Column | Type | Description |
|--------|------|-------------|
| threads | int | Number of worker threads. |
| run | int | Run number for statistical significance. |
| tps | float | Throughput in tasks per second. |
| avgLat | float | Average latency in milliseconds. |
| p99Lat | float | 99th percentile latency. |

---

## Key Algorithms:

### Blocking Ratio Calculation:

```python
beta = 1.0 - (cpuTime / wallTime)
```

- `beta > 0.7` indicates I/O-bound workload (safe to scale up).
- `beta < 0.3` indicates CPU-bound workload (risk of GIL contention).

### Scaling Decision Logic:

```python
if queueLength > 0 and avgBeta > GIL_DANGER_ZONE:
    # I/O-bound with backlog: scale up.
    return currentThreads + scaleUpStep
elif queueLength > 0 and avgBeta <= GIL_DANGER_ZONE:
    # CPU-bound with backlog: VETO scale-up.
    return currentThreads
elif queueLength == 0:
    # No backlog: scale down to save resources.
    return max(currentThreads - scaleDownStep, minThreads)
```

---

## Dependencies:

### Required:

- `psutil>=5.9.0` - System monitoring.
- `matplotlib>=3.7.0` - Figure generation.

### Optional:

- `numpy>=1.24.0` - Extended workload types.
- `pytest>=7.0.0` - Running test suite.

---

## Contributing:

### Code Style:

- Use camelCase for function and variable names.
- Use single-line comments with `#` ending in periods.
- Follow PEP 8 with the above naming convention exception.

### Adding New Experiments:

1. Create script in `experiments/` directory.
2. Use consistent CSV output format.
3. Update `generateFigures.py` to include new visualizations.

### Adding Platform Support:

1. Create script in `platforms/` directory.
2. Include hardware detection for the target platform.
3. Match CSV output schema for compatibility.
