# 🧵 BetaPool

A Python library implementing the Metric-Driven Adaptive Thread Pool for mitigating GIL bottlenecks in mixed I/O and CPU workloads. Designed for edge AI systems and resource-constrained environments.

**Author:** Mridankan Mandal  
**License:** MIT  
**Python Version:** 3.11+

## 📖 Overview

Deploying Python-based AI agents on resource-constrained edge devices presents a fundamental concurrency paradox. While high thread counts are necessary to mask I/O latency (sensor reads, API calls, model serving), Python's Global Interpreter Lock (GIL) imposes a hard ceiling on compute scalability. Standard thread pool heuristics fail to detect GIL-specific contention, leading to **concurrency thrashing**: a state where increasing thread count paradoxically degrades throughput.

BetaPool provides a GIL Safety Veto mechanism using the **Blocking Ratio (beta)** metric to automatically maintain optimal concurrency levels.

## ✨ Key Features

- **Adaptive Thread Pool:** Automatically adjusts thread count based on workload characteristics.
- **GIL Safety Veto:** Prevents thread pool expansion when CPU-bound work would cause GIL contention.
- **Blocking Ratio Metric:** Distinguishes I/O-bound from CPU-bound workloads without GIL instrumentation.
- **Drop-in Replacement:** Compatible with `concurrent.futures.ThreadPoolExecutor` interface.
- **Memory Efficient:** Suitable for edge devices with limited RAM (unlike multiprocessing).
- **Python 3.11+ Support:** Optimized for modern Python versions.

## 📦 Installation

**From PyPI (when published):**

```bash
pip install betapool
```

**From Source:**

```bash
git clone https://github.com/WhiteMetagross/BetaPool.git
cd BetaPool
pip install -e .
```

**With Optional Dependencies:**

```bash
# With NumPy support for workload generators.
pip install betapool[numpy]

# With visualization support.
pip install betapool[visualization]

# With development dependencies.
pip install betapool[dev]

# All dependencies.
pip install betapool[all]
```

## 🚀 Quick Start

**Basic Usage:**

```python
from betapool import AdaptiveThreadPoolExecutor

# Create an adaptive executor.
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    # Submit tasks (same API as ThreadPoolExecutor).
    futures = [executor.submit(my_task, arg) for arg in args]
    results = [f.result() for f in futures]
```

**With Custom Configuration:**

```python
from betapool import AdaptiveThreadPoolExecutor, ControllerConfig

# Configure the adaptive controller.
config = ControllerConfig(
    monitor_interval_sec=0.5,      # How often to check metrics.
    beta_high_threshold=0.7,       # Scale up when blocking ratio > 0.7.
    beta_low_threshold=0.3,        # Scale down when blocking ratio < 0.3.
    scale_up_step=2,               # Add 2 threads when scaling up.
    scale_down_step=1,             # Remove 1 thread when scaling down.
    warmup_task_count=10,          # Wait for 10 tasks before scaling.
)

with AdaptiveThreadPoolExecutor(
    min_workers=4,
    max_workers=64,
    config=config,
    enable_logging=True
) as executor:
    # Your workload here.
    pass
```

**Monitoring Metrics:**

```python
with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
    # Submit your tasks.
    futures = [executor.submit(task, arg) for arg in args]
    
    # Get current metrics.
    metrics = executor.get_metrics()
    print(f"Current threads: {metrics['current_threads']}")
    print(f"Blocking ratio: {metrics['avg_blocking_ratio']:.2f}")
    print(f"Throughput: {metrics['throughput']:.1f} tasks/sec")
    print(f"P99 latency: {metrics['p99_latency']*1000:.1f} ms")
```

## 📊 The Blocking Ratio (Beta)

The core innovation is the **Blocking Ratio** metric:

```
beta = 1 - (cpu_time / wall_time)
```

- **beta near 1.0:** Thread is mostly waiting (I/O-bound) - safe to add more threads.
- **beta near 0.0:** Thread is mostly computing (CPU-bound) - adding threads causes GIL contention.

This metric allows the controller to detect when the system is approaching the "saturation cliff" without requiring GIL-level instrumentation.

## 🛡️ The GIL Safety Veto

When the blocking ratio indicates CPU-bound work (beta < threshold), the controller implements a **veto** on thread pool expansion:

```
if queue_length > 0:           # There is work waiting.
    if beta > GIL_DANGER_ZONE: # Workers are doing I/O.
        scale_up()             # Safe to add threads.
    else:
        hold()                 # VETO: Adding threads would cause thrashing.
```

## 📈 Comparison with Static Thread Pools

| Metric | Static (256 threads) | Static Optimal (32) | Adaptive |
|--------|---------------------|---------------------|----------|
| Throughput | 31,087 TPS | 37,437 TPS | 36,142 TPS |
| P99 Latency | 38.2 ms | 10.1 ms | 11.8 ms |
| vs Naive | Baseline | +20% | +16% |

The adaptive solution achieves 96.5% of optimal performance without requiring manual tuning.

## 🔧 Workload Generators

The library includes utilities for testing and benchmarking:

```python
from betapool import WorkloadGenerator

# I/O-bound task (releases GIL during sleep).
io_task = WorkloadGenerator.io_task(duration_ms=50.0)

# CPU-bound task (holds GIL).
cpu_task = WorkloadGenerator.cpu_task_python(iterations=100000)

# Mixed workload (simulates real applications).
mixed_task = WorkloadGenerator.mixed_task(
    io_duration_ms=50.0,
    cpu_iterations=10000
)

# RAG pipeline simulation.
rag_task = WorkloadGenerator.rag_pipeline_task(
    network_latency_ms=10.0,
    vector_db_latency_ms=200.0,
    llm_latency_ms=500.0,
)
```

## 📚 API Reference

**AdaptiveThreadPoolExecutor:**

The main class providing adaptive thread pool functionality.

**Constructor:**

```python
AdaptiveThreadPoolExecutor(
    min_workers: int = 4,
    max_workers: int = 64,
    config: Optional[ControllerConfig] = None,
    enable_logging: bool = False
)
```

**Methods:**

- `submit(fn, *args, **kwargs) -> Future`: Submit a task for execution.
- `map(fn, *iterables, timeout=None) -> Iterator`: Map function over iterables.
- `get_current_thread_count() -> int`: Get current active thread count.
- `get_metrics() -> Dict`: Get current metrics summary.
- `get_decision_history() -> List[Dict]`: Get scaling decision history.
- `shutdown(wait=True)`: Shutdown the executor.

**ControllerConfig:**

Configuration for the adaptive controller.

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

Container for individual task execution metrics.

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

## 💡 Use Cases

**Edge AI Inference:**

```python
from betapool import AdaptiveThreadPoolExecutor
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")

def inference_task(input_data):
    # Preprocess (CPU).
    processed = preprocess(input_data)
    # Inference (releases GIL in ONNX Runtime).
    result = session.run(None, {"input": processed})
    # Postprocess (CPU).
    return postprocess(result)

with AdaptiveThreadPoolExecutor(min_workers=2, max_workers=16) as executor:
    results = list(executor.map(inference_task, batch_data))
```

**RAG Pipeline:**

```python
from betapool import AdaptiveThreadPoolExecutor

def rag_request(query):
    # Embedding (CPU + API call).
    embedding = embed_query(query)
    # Vector search (I/O).
    docs = vector_db.search(embedding)
    # Reranking (CPU).
    ranked = rerank(docs)
    # LLM call (I/O).
    response = llm.generate(query, ranked)
    return response

with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=32) as executor:
    responses = list(executor.map(rag_request, queries))
```

**Web Server Handler:**

```python
from betapool import AdaptiveThreadPoolExecutor
from flask import Flask, request

app = Flask(__name__)
executor = AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64)

@app.route("/process", methods=["POST"])
def process():
    future = executor.submit(heavy_processing, request.json)
    return {"result": future.result(timeout=30.0)}
```

## 📋 Requirements

- Python 3.11 or higher.
- psutil >= 5.9.0.

**Optional Dependencies:**

- numpy >= 1.24.0 (for NumPy workload generators).
- matplotlib >= 3.7.0 (for visualization).

## 🧪 Testing

```bash
# Run tests.
pytest betapool/tests/ -v

# Run with coverage.
pytest betapool/tests/ --cov=betapool
```

## 📄 Research Paper

📄 **Read the full paper:** [Mitigating GIL Bottlenecks in Edge AI Systems (arXiv:2601.10582)](https://arxiv.org/pdf/2601.10582)

## 📝 Citation

If you use this library in your research, please cite:

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

## 🤝 Contributing

Contributions are welcome. Please ensure:

1. Code follows the existing style (use `black` and `isort`).
2. All tests pass (`pytest`).
3. Type hints are included (`mypy`).
4. Documentation is updated.

## 🙏 Acknowledgments

This work builds upon the foundational GIL analysis by David Beazley and the Python concurrency community.
