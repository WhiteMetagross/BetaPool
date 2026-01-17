# BetaPool

A GIL-aware adaptive thread pool for Python that prevents concurrency thrashing through the Blocking Ratio metric and GIL Safety Veto mechanism.

**Author:** Mridankan Mandal  
**License:** MIT  
**Python Version:** 3.11+  
**Paper:** [arXiv:2601.10582](https://arxiv.org/abs/2601.10582)

## The Problem:

Standard thread pool heuristics (queue depth, CPU saturation, response time) fail to detect GIL-specific contention. This leads to **concurrency thrashing**: increasing thread count paradoxically degrades throughput.

Research findings:
- Single-core: 32.2% throughput loss at 2048 threads vs optimal 32 threads.
- Quad-core: 33.3% throughput loss at 2048 threads vs optimal 64 threads.

## The Solution:

BetaPool uses the **Blocking Ratio (beta)** metric to distinguish I/O-bound from CPU-bound work:

```
beta = 1 - (cpu_time / wall_time)
```

- **beta near 1.0:** Thread is mostly waiting (I/O-bound) - safe to add threads.
- **beta near 0.0:** Thread is mostly computing (CPU-bound) - adding threads causes GIL contention.

The **GIL Safety Veto** prevents thread scaling when beta < 0.3, avoiding the saturation cliff.

## Performance:

| Metric | Static (256 threads) | Static Optimal (32) | BetaPool |
|--------|---------------------|---------------------|----------|
| Throughput | 31,087 TPS | 37,437 TPS | 36,142 TPS |
| P99 Latency | 38.2 ms | 10.1 ms | 11.8 ms |

BetaPool achieves 96.5% of optimal performance without manual tuning.

## Installation:

```bash
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[numpy]"        # NumPy workload generators.
pip install -e ".[dev]"          # Development tools.
pip install -e ".[all]"          # All dependencies.
```

## Quick Start:

```python
from betapool import AdaptiveThreadPoolExecutor

with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    futures = [executor.submit(my_task, arg) for arg in args]
    results = [f.result() for f in futures]
```

## Configuration:

```python
from betapool import AdaptiveThreadPoolExecutor, ControllerConfig

config = ControllerConfig(
    monitor_interval_sec=0.5,      # Metrics check interval.
    beta_high_threshold=0.7,       # Scale up when beta > 0.7.
    beta_low_threshold=0.3,        # GIL Safety Veto threshold.
    scale_up_step=2,               # Threads to add per scale-up.
    scale_down_step=1,             # Threads to remove per scale-down.
    cpu_upper_threshold=85.0,      # CPU limit for scaling.
    warmup_task_count=10,          # Tasks before scaling decisions.
)

with AdaptiveThreadPoolExecutor(
    min_workers=4,
    max_workers=64,
    config=config,
    enable_logging=True
) as executor:
    pass
```

## Monitoring:

```python
metrics = executor.get_metrics()
print(f"Threads: {metrics['current_threads']}")
print(f"Beta: {metrics['avg_blocking_ratio']:.2f}")
print(f"Throughput: {metrics['throughput']:.1f} TPS")
print(f"P99 Latency: {metrics['p99_latency']*1000:.1f} ms")
```

## API Reference:

**AdaptiveThreadPoolExecutor:**

Constructor:

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

**ControllerConfig:**

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

**TaskMetrics:**

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

For testing and benchmarking:

```python
from betapool import WorkloadGenerator

io_task = WorkloadGenerator.io_task(duration_ms=50.0)
cpu_task = WorkloadGenerator.cpu_task_python(iterations=100000)
mixed_task = WorkloadGenerator.mixed_task(io_duration_ms=50.0, cpu_iterations=10000)
rag_task = WorkloadGenerator.rag_pipeline_task()
```

## Requirements:

- Python 3.11 or higher.
- psutil >= 5.9.0.

## Testing:

```bash
pytest betapool/tests/ -v
```

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
