# BetaPool:

A Python library implementing the Metric-Driven Adaptive Thread Pool for mitigating GIL bottlenecks in mixed I/O and CPU workloads.

**Author:** Mridankan Mandal  
**License:** MIT  
**Python:** 3.8+

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

## Installation:

```bash
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[dev]"      # Development tools.
pip install -e ".[numpy]"    # NumPy workload generators.
pip install -e ".[all]"      # All dependencies.
```

## Quick Start:

```python
from betapool import AdaptiveThreadPoolExecutor

# Drop-in replacement for ThreadPoolExecutor.
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    futures = [executor.submit(my_task, arg) for arg in args]
    results = [f.result() for f in futures]

    # Monitor adaptive behavior.
    metrics = executor.get_metrics()
    print(f"Current threads: {metrics['current_threads']}")
    print(f"Blocking ratio: {metrics['avg_blocking_ratio']:.2f}")
```

## The Blocking Ratio:

The core algorithm uses the **Blocking Ratio** metric:

```
beta = 1 - (cpu_time / wall_time)
```

- **beta near 1.0:** Thread is mostly waiting (I/O-bound). Safe to add threads.
- **beta near 0.0:** Thread is mostly computing (CPU-bound). Adding threads causes GIL contention.

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

**AdaptiveThreadPoolExecutor:**

```python
from betapool import AdaptiveThreadPoolExecutor, ControllerConfig

config = ControllerConfig(
    monitor_interval_sec=0.5,    # Metric check interval.
    beta_high_threshold=0.7,     # Scale up threshold.
    beta_low_threshold=0.3,      # GIL danger zone.
    scale_up_step=2,             # Threads to add.
    scale_down_step=1,           # Threads to remove.
)

with AdaptiveThreadPoolExecutor(
    min_workers=4,
    max_workers=64,
    config=config
) as executor:
    future = executor.submit(task_function, arg1, arg2)
    result = future.result()
```

**Methods:**

- `submit(fn, *args, **kwargs) -> Future`: Submit a task for execution.
- `map(fn, *iterables, timeout=None) -> Iterator`: Map function over iterables.
- `get_current_thread_count() -> int`: Get current active thread count.
- `get_metrics() -> Dict`: Get current metrics summary.
- `shutdown(wait=True)`: Shutdown the executor.

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
