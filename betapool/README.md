# BetaPool

A Python library implementing the Metric-Driven Adaptive Thread Pool for mitigating GIL bottlenecks in mixed I/O and CPU workloads.

**Author:** Mridankan Mandal  
**License:** MIT  
**Python:** 3.11+

## Overview:

Standard thread pool implementations fail to detect GIL-specific contention in Python, leading to **concurrency thrashing** where increasing thread count paradoxically degrades throughput. BetaPool solves this by implementing a **GIL Safety Veto** mechanism using the **Blocking Ratio (beta)** metric to automatically maintain optimal concurrency levels.

## Why BetaPool:

Research demonstrates that naive thread scaling causes significant performance degradation:

| Configuration | Peak Throughput | Degradation at High Threads |
|---------------|-----------------|----------------------------|
| Single-core | 37,437 TPS at 32 threads | 32.2% loss at 2048 threads |
| Quad-core | 68,742 TPS at 64 threads | 33.3% loss at 2048 threads |
| **BetaPool** | 36,142 TPS | **96.5% of optimal** (automatic) |

BetaPool achieves near-optimal performance without manual tuning by detecting when thread scaling would cause GIL contention.

## Key Features:

- **Adaptive Thread Pool:** Automatically adjusts thread count based on workload characteristics.
- **GIL Safety Veto:** Prevents thread pool expansion when CPU-bound work would cause GIL contention.
- **Blocking Ratio Metric:** Distinguishes I/O-bound from CPU-bound workloads without GIL instrumentation.
- **Drop-in Replacement:** Compatible with `concurrent.futures.ThreadPoolExecutor` interface.
- **Memory Efficient:** Suitable for edge devices with limited RAM (unlike multiprocessing).

## Installation:

From source:

```bash
git clone https://github.com/WhiteMetagross/BetaPool.git
cd BetaPool
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[numpy]"    # NumPy workload generators.
pip install -e ".[dev]"      # Development tools.
pip install -e ".[all]"      # All dependencies.
```

## Quick Start:

Basic usage:

```python
from betapool import AdaptiveThreadPoolExecutor

# Drop-in replacement for ThreadPoolExecutor.
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    futures = [executor.submit(my_task, arg) for arg in args]
    results = [f.result() for f in futures]
```

With custom configuration:

```python
from betapool import AdaptiveThreadPoolExecutor, ControllerConfig

config = ControllerConfig(
    monitor_interval_sec=0.5,    # How often to check metrics.
    beta_high_threshold=0.7,     # Scale up when blocking ratio > 0.7.
    beta_low_threshold=0.3,      # GIL danger zone threshold.
    scale_up_step=2,             # Add 2 threads when scaling up.
    scale_down_step=1,           # Remove 1 thread when scaling down.
    warmup_task_count=10,        # Wait for 10 tasks before scaling.
)

with AdaptiveThreadPoolExecutor(
    min_workers=4,
    max_workers=64,
    config=config
) as executor:
    # Your workload here.
    pass
```

Monitoring metrics:

```python
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    futures = [executor.submit(task, arg) for arg in args]
    
    metrics = executor.get_metrics()
    print(f"Current threads: {metrics['current_threads']}")
    print(f"Blocking ratio: {metrics['avg_blocking_ratio']:.2f}")
    print(f"Throughput: {metrics['throughput']:.1f} tasks/sec")
```

## The Blocking Ratio:

The core algorithm uses the **Blocking Ratio** metric:

```
beta = 1 - (cpu_time / wall_time)
```

- **beta near 1.0:** Thread is mostly waiting (I/O-bound). Safe to add threads.
- **beta near 0.0:** Thread is mostly computing (CPU-bound). Adding threads causes GIL contention.

This metric detects when the system is approaching the "saturation cliff" without requiring GIL-level instrumentation.

## The GIL Safety Veto:

When beta falls below the danger threshold (default 0.3), the controller vetoes thread pool expansion:

```
if beta > threshold:    # I/O-bound work.
    scale_up()          # Safe to add threads.
else:
    hold()              # VETO: Prevents GIL thrashing.
```

This mechanism prevents the 32% throughput loss observed with naive thread scaling.

## API Reference:

AdaptiveThreadPoolExecutor:

```python
AdaptiveThreadPoolExecutor(
    min_workers: int = 4,
    max_workers: int = 64,
    config: Optional[ControllerConfig] = None,
    enable_logging: bool = False
)
```

Methods:

- `submit(fn, *args, **kwargs) -> Future`: Submit a task for execution.
- `map(fn, *iterables, timeout=None) -> Iterator`: Map function over iterables.
- `get_current_thread_count() -> int`: Get current active thread count.
- `get_metrics() -> Dict`: Get current metrics summary.
- `get_decision_history() -> List[Dict]`: Get scaling decision history.
- `shutdown(wait=True)`: Shutdown the executor.

ControllerConfig:

```python
ControllerConfig(
    monitor_interval_sec: float = 0.5,
    beta_high_threshold: float = 0.7,
    beta_low_threshold: float = 0.3,
    scale_up_step: int = 2,
    scale_down_step: int = 1,
    cpu_upper_threshold: float = 85.0,
    cpu_lower_threshold: float = 50.0,
    stabilization_window_sec: float = 2.0,
    warmup_task_count: int = 10,
)
```

TaskMetrics:

```python
TaskMetrics(
    task_id: str,
    wall_time: float,
    cpu_time: float,
    blocking_ratio: float,
    timestamp: float,
    success: bool = True,
    error_message: Optional[str] = None,
)
```

## Workload Generators:

The library includes utilities for testing:

```python
from betapool import WorkloadGenerator

# I/O-bound task (releases GIL during sleep).
io_task = WorkloadGenerator.io_task(duration_ms=50.0)

# CPU-bound task (holds GIL).
cpu_task = WorkloadGenerator.cpu_task_python(iterations=100000)

# Mixed workload.
mixed_task = WorkloadGenerator.mixed_task(
    io_duration_ms=50.0,
    cpu_iterations=10000
)
```

## Requirements:

- Python 3.11 or higher.
- psutil >= 5.9.0.

Optional dependencies:

- numpy >= 1.24.0 (for NumPy workload generators).

## Testing:

```bash
pytest betapool/tests/ -v
```

## Research Paper:

**Read the full paper:** [Mitigating GIL Bottlenecks in Edge AI Systems (arXiv:2601.10582)](https://arxiv.org/pdf/2601.10582)

## Citation:

```bibtex
@article{mandal2026gilscheduler,
  title={Mitigating GIL Bottlenecks in Edge AI Systems: A Metric-Driven Adaptive Thread Pool},
  author={Mandal, Mridankan},
  journal={arXiv preprint arXiv:2601.10582},
  year={2026},
  url={https://arxiv.org/abs/2601.10582}
}
```

## License:

MIT License - see LICENSE file for details.
