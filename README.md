# Mitigating GIL Bottlenecks in Edge AI Systems:

**Author:** Anonymous Authors

## Abstract:

Deploying Python based AI agents on resource constrained edge devices presents a concurrency paradox. High thread counts are necessary to mask the latency of Input/Output operations like sensor reads and API calls. However, the Global Interpreter Lock (GIL) in Python imposes a hard limit on compute scalability. Standard thread pool heuristics rely on queue depth or CPU saturation. These fail to detect GIL specific contention. This leads to concurrency thrashing, a state where increasing the thread count degrades throughput. In this work, we present the first systematic characterization of GIL induced performance degradation on edge devices. We identified the "saturation cliff" through controlled experiments on simulated single core and quad core edge environments. This is a critical threshold beyond which performance collapses. Our findings show throughput degradation from peak 37,437 TPS at 32 threads to 25,386 TPS at 2048 threads (32.2% loss) on single-core devices, and from peak 68,742 TPS at 64 threads to 45,821 TPS at 2048 threads (33.3% loss) on quad-core devices. P99 latency increases 40.9x (single-core) and 29.5x (quad-core) from optimal to over provisioned configurations. We demonstrate that this cliff persists on multi-core hardware due to conflicts between OS scheduling and GIL serialization, which we call the OS GIL Paradox. We propose a user space concurrency controller that uses a GIL Safety Veto mechanism. By monitoring the Blocking Ratio beta of active tasks, the controller identifies serialization and prevents the allocation of additional worker threads, effectively clamping the system to its optimal operating point. Our adaptive solution achieves 36,142 TPS (96.5% of the optimal) with P99 latency of 11.8 ms, compared to the naive approach's 31,087 TPS and 38.2 ms P99 latency.

## Key Results:

- **Single core devices:** 32.2% throughput loss at 2048 threads (peak at 32 threads).
- **Quad core devices:** 33.3% throughput loss at 2048 threads (peak at 64 threads).
- **Latency explosion:** P99 latency increases 40.9x (single-core) and 29.5x (quad-core).

## Project Structure:

```
MandalSchedulingResearchAlgorithm/
├── README.md                    # This file.
├── Usage.md                     # Detailed usage instructions.
├── CodeBaseIndex.md             # Codebase navigation guide.
├── requirements.txt             # Python dependencies.
├── experiments/                 # Experiment scripts.
│   ├── singleCoreBenchmark.py   # Single core cliff characterization.
│   ├── quadCoreBenchmark.py     # Quad core edge device benchmark.
│   ├── instrumentationOverhead.py # Timer overhead measurement.
│   ├── workloadSweep.py         # CPU/IO ratio sweep.
│   ├── baselineComparison.py    # Alternative strategy comparison.
│   ├── controllerTimeline.py    # Controller behavior recording.
│   └── generateFigures.py       # Publication figure generation.
├── platforms/                   # Platform specific benchmarks.
│   ├── raspberryPiBenchmark.py  # Raspberry Pi 4 native benchmark.
│   ├── jetsonNanoBenchmark.py   # NVIDIA Jetson Nano benchmark.
```
