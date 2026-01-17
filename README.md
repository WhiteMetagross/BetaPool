# BetaPool

A GIL-aware adaptive thread pool for Python that prevents concurrency thrashing.

**Author:** Mridankan Mandal  
**License:** MIT  
**Paper:** [arXiv:2601.10582](https://arxiv.org/abs/2601.10582)

## Overview:

Standard thread pool heuristics fail to detect GIL-specific contention, leading to **concurrency thrashing**: a pathological state where increasing thread count paradoxically degrades throughput. Research shows up to 32.2% throughput loss when naively scaling threads on CPU-bound Python workloads.

BetaPool solves this with the **Blocking Ratio (beta)** metric and a **GIL Safety Veto** mechanism that automatically prevents harmful thread scaling.

## Why BetaPool:

| Metric | Static (256 threads) | Static Optimal (32) | BetaPool Adaptive |
|--------|---------------------|---------------------|-------------------|
| Throughput | 31,087 TPS | 37,437 TPS | 36,142 TPS |
| P99 Latency | 38.2 ms | 10.1 ms | 11.8 ms |
| vs Naive | Baseline | +20% | +16% |

BetaPool achieves **96.5% of optimal performance** without requiring manual tuning.

## Installation:

```bash
pip install -e .
```

Requirements:
- Python 3.11 or higher.
- psutil >= 5.9.0.

## Quick Start:

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

## The Blocking Ratio (Beta):

The core innovation is the **Blocking Ratio** metric:

```
beta = 1 - (cpu_time / wall_time)
```

- **beta near 1.0:** Thread is mostly waiting (I/O-bound) - safe to add more threads.
- **beta near 0.0:** Thread is mostly computing (CPU-bound) - adding threads causes GIL contention.

## The GIL Safety Veto:

When the blocking ratio indicates CPU-bound work (beta < 0.3), the controller implements a **veto** on thread pool expansion:

```
if queue_length > 0:           # There is work waiting.
    if beta > GIL_DANGER_ZONE: # Workers are doing I/O.
        scale_up()             # Safe to add threads.
    else:
        hold()                 # VETO: Adding threads would cause thrashing.
```

This prevents the 32% throughput degradation observed when scaling threads on CPU-bound workloads.

## API Reference:

See [betapool/README.md](betapool/README.md) for complete API documentation.

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
