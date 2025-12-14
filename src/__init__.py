# Adaptive Thread Pool Executor for Mixed Workload Optimization
# Research Implementation for Systems Paper Submission
#
# This package provides an adaptive thread pool executor that dynamically
# adjusts thread count based on workload characteristics using control theory.

from src.adaptive_executor import AdaptiveThreadPoolExecutor
from src.metrics import TaskMetrics, MetricsCollector

__version__ = "0.1.0"
__author__ = "Research Team"

__all__ = [
    "AdaptiveThreadPoolExecutor",
    "TaskMetrics",
    "MetricsCollector",
]
