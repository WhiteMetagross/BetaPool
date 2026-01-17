"""
BetaPool: Adaptive Thread Pool for GIL-Aware Concurrency Control

A Python library implementing the Metric-Driven Adaptive Thread Pool for mitigating
GIL bottlenecks in mixed I/O and CPU workloads, particularly suited for edge AI systems.

BetaPool uses the Blocking Ratio (beta) metric to distinguish between I/O-bound and
CPU-bound workloads, implementing a GIL Safety Veto mechanism to prevent concurrency
thrashing - the pathological state where increasing thread count degrades throughput.

Key Features:
- AdaptiveThreadPoolExecutor: Dynamically adjusts thread count based on workload
- Blocking Ratio (beta) metric for workload classification
- GIL Safety Veto mechanism to prevent the "saturation cliff"
- Support for edge devices with memory constraints (512MB-2GB RAM)

Research Background:
- Demonstrates 32.2% throughput loss at 2048 threads on single-core (peak at 32 threads)
- Demonstrates 33.3% throughput loss at 2048 threads on quad-core (peak at 64 threads)
- Adaptive solution achieves 96.5% of optimal performance without manual tuning

Example:
    from betapool import AdaptiveThreadPoolExecutor

    with AdaptiveThreadPoolExecutor(min_workers=4, max_workers=64) as executor:
        futures = [executor.submit(task, arg) for arg in args]
        results = [f.result() for f in futures]

Author: Mridankan Mandal
License: MIT
"""

from __future__ import annotations

from betapool.executor import (
    AdaptiveThreadPoolExecutor,
    StaticThreadPoolExecutor,
    ControllerConfig,
    ControllerState,
)
from betapool.metrics import (
    TaskMetrics,
    MetricsCollector,
    AggregatedMetrics,
)
from betapool.workloads import (
    WorkloadGenerator,
    PoissonArrivalGenerator,
    BurstArrivalGenerator,
)

__version__ = "1.0.0"
__author__ = "Mridankan Mandal"
__license__ = "MIT"

__all__ = [
    # Core executor classes
    "AdaptiveThreadPoolExecutor",
    "StaticThreadPoolExecutor",
    "ControllerConfig",
    "ControllerState",
    # Metrics classes
    "TaskMetrics",
    "MetricsCollector",
    "AggregatedMetrics",
    # Workload utilities
    "WorkloadGenerator",
    "PoissonArrivalGenerator",
    "BurstArrivalGenerator",
]
