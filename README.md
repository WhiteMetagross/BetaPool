# 🧵 Mitigating GIL Bottlenecks in Edge AI Systems

A Metric-Driven Adaptive Thread Pool for Python Concurrency Control.

**Author:** Mridankan Mandal  
**License:** MIT

## 📖 Abstract

Deploying Python-based AI agents on resource-constrained edge devices presents a fundamental concurrency paradox. While high thread counts are necessary to mask the latency of I/O-bound operations (sensor reads, API calls, model serving), Python's Global Interpreter Lock (GIL) imposes a hard ceiling on compute scalability. Standard thread pool heuristics, which rely on queue depth, CPU saturation, or response time, fail to detect GIL-specific contention, leading to **concurrency thrashing**: a pathological state where increasing thread count paradoxically degrades throughput.

This repository contains the research implementation and the **betapool** Python library that implements the Metric-Driven Adaptive Thread Pool with GIL Safety Veto mechanism.

## 📊 Key Results

| Configuration | Peak TPS | Degradation at High Threads |
|---------------|----------|----------------------------|
| Single-core devices | 37,437 at 32 threads | 32.2% loss at 2048 threads |
| Quad-core devices | 68,742 at 64 threads | 33.3% loss at 2048 threads |
| Adaptive solution | 36,142 TPS | 96.5% of optimal (automatic) |

## 📦 Installation

**Install the Library:**

```bash
# From the repository root.
pip install -e .

# Or install with all dependencies.
pip install -e ".[all]"
```

**Requirements:**

- Python 3.11 or higher.
- psutil >= 5.9.0.

## 🚀 Quick Start

```python
from betapool import AdaptiveThreadPoolExecutor

# Drop-in replacement for ThreadPoolExecutor.
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    futures = [executor.submit(my_task, arg) for arg in args]
    results = [f.result() for f in futures]

    # Monitor the adaptive behavior.
    metrics = executor.get_metrics()
    print(f"Current threads: {metrics['current_threads']}")
    print(f"Blocking ratio: {metrics['avg_blocking_ratio']:.2f}")
```

## 📊 The Blocking Ratio (Beta)

The core innovation is the **Blocking Ratio** metric:

```
beta = 1 - (cpu_time / wall_time)
```

- **beta near 1.0:** Thread is mostly waiting (I/O-bound) - safe to add more threads.
- **beta near 0.0:** Thread is mostly computing (CPU-bound) - adding threads causes GIL contention.

## 📁 Project Structure

```
MandalSchedulingResearchAlgorithm/
├── README.md                    # This file.
├── Usage.md                     # Detailed usage instructions.
├── CodeBaseIndex.md             # Codebase navigation guide.
├── pyproject.toml               # Package configuration for pip install.
├── requirements.txt             # Python dependencies.
├── betapool/                    # The pip-installable library.
│   ├── __init__.py              # Public API exports.
│   ├── executor.py              # AdaptiveThreadPoolExecutor implementation.
│   ├── metrics.py               # TaskMetrics and MetricsCollector.
│   ├── workloads.py             # Workload generators for testing.
│   ├── README.md                # Library documentation.
│   └── tests/                   # Unit tests.
├── experiments/                 # Research experiment scripts.
│   ├── singleCoreBenchmark.py   # Single-core cliff characterization.
│   ├── quadCoreBenchmark.py     # Quad-core edge device benchmark.
│   ├── baselineComparison.py    # Alternative strategy comparison.
│   └── generateFigures.py       # Publication figure generation.
├── src/                         # Original research implementation.
├── docs/                        # Research paper and documentation.
├── results/                     # Experiment results.
├── figures/                     # Generated figures.
└── docker/                      # Docker configurations for reproducibility.
```

## 📚 Library API

**AdaptiveThreadPoolExecutor:**

```python
from betapool import AdaptiveThreadPoolExecutor, ControllerConfig

config = ControllerConfig(
    monitor_interval_sec=0.5,      # Check metrics every 500ms.
    beta_high_threshold=0.7,       # Scale up when beta > 0.7.
    beta_low_threshold=0.3,        # Scale down when beta < 0.3.
    scale_up_step=2,               # Add 2 threads at a time.
    scale_down_step=1,             # Remove 1 thread at a time.
)

with AdaptiveThreadPoolExecutor(
    min_workers=4,
    max_workers=64,
    config=config
) as executor:
    # Submit tasks.
    future = executor.submit(task_function, arg1, arg2)
    result = future.result()
    
    # Or use map.
    results = list(executor.map(task_function, iterable))
    
    # Get metrics.
    metrics = executor.get_metrics()
```

**Workload Generators:**

```python
from betapool import WorkloadGenerator

# Create test workloads.
io_task = WorkloadGenerator.io_task(duration_ms=50.0)
cpu_task = WorkloadGenerator.cpu_task_python(iterations=100000)
mixed_task = WorkloadGenerator.mixed_task(io_duration_ms=50.0, cpu_iterations=10000)
rag_task = WorkloadGenerator.rag_pipeline_task()
```

## 🔬 Running Experiments

**Single-Core Benchmark:**

```bash
python experiments/singleCoreBenchmark.py
```

**Quad-Core Benchmark:**

```bash
python experiments/quadCoreBenchmark.py
```

**Generate Figures:**

```bash
python experiments/generateFigures.py
```

## 🧪 Testing

```bash
# Run library tests.
pytest betapool/tests/ -v

# Run with coverage.
pytest betapool/tests/ --cov=betapool --cov-report=html
```

## 📖 Documentation

- [Usage.md](Usage.md) - Detailed usage instructions.
- [CodeBaseIndex.md](CodeBaseIndex.md) - Codebase navigation guide.
- [betapool/README.md](betapool/README.md) - Library documentation.
- [docs/paper.md](docs/paper.md) - Full research paper.

## 📄 Research Paper

📄 **Read the full paper:** [Mitigating GIL Bottlenecks in Edge AI Systems (arXiv:2601.10582)](https://arxiv.org/pdf/2601.10582)

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{mandal2026gilscheduler,
  title={Mitigating GIL Bottlenecks in Edge AI Systems: A Metric-Driven Adaptive Thread Pool},
  author={Mandal, Mridankan},
  journal={arXiv preprint arXiv:2601.10582},
  year={2026},
  url={https://arxiv.org/abs/2601.10582}
}
```

## ⚖️ License

MIT License - see LICENSE file for details.
