# Edge GIL: Mitigating GIL Induced Concurrency Thrashing in Edge AI Systems:

A research project characterizing and mitigating the GIL Saturation Cliff phenomenon in Python based edge AI systems.

## Overview:

This repository provides experimental evidence of the **GIL Saturation Cliff**, a phenomenon where Python thread pool throughput crashes by 30 to 40% at high thread counts due to Global Interpreter Lock contention.

**Author:** Mridankan Mandal  
**Affiliation:** Indian Institute of Information Technology, Allahabad  
**Target Venue:** EdgeSys 2026 (EuroSys Workshop).

## The Problem:

Python uses a **Global Interpreter Lock (GIL)** that allows only one thread to execute Python bytecode at a time. When thread counts increase beyond optimal levels:

1. **Low thread counts (1-32):** Threads coordinate efficiently, and throughput scales.
2. **High thread counts (128+):** Threads compete aggressively for the lock, spending more time waiting than working. **Throughput crashes.**

This is the **Saturation Cliff**.

## Key Results:

| Threads | Throughput (TPS) | Drop from Peak |
|---------|------------------|----------------|
| 32      | 39,658           | Peak (0%)      |
| 128     | 36,965           | -6.8%          |
| 512     | 35,862           | -9.6%          |
| 1024    | 28,701           | -27.6%         |
| 2048    | 24,043           | -39.4%         |

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
│   └── generateFigures.py       # Publication figure generation.
├── platforms/                   # Platform specific benchmarks.
│   ├── raspberryPiBenchmark.py  # Raspberry Pi 4 native benchmark.
│   └── jetsonNanoBenchmark.py   # NVIDIA Jetson Nano benchmark.
├── docker/                      # Docker configurations.
│   ├── Dockerfile               # Base image.
│   ├── Dockerfile.singleCore    # Single core simulation.
│   ├── Dockerfile.quadCore      # Quad core simulation.
│   └── docker-compose.yml       # Container orchestration.
├── docs/                        # Documentation.
│   ├── paper.md                 # Research paper draft.
│   └── reproducibility.md       # Reproducibility guide.
├── src/                         # Core implementation.
│   ├── adaptiveExecutor.py      # Adaptive thread pool.
│   ├── metrics.py               # Blocking ratio metrics.
│   └── workloads.py             # Workload generators.
├── tests/                       # Unit tests.
│   └── testAdaptiveExecutor.py  # Executor tests.
├── results/                     # Experimental data (CSV).
└── figures/                     # Generated figures (PDF/PNG).
```

## Quick Start:

### Prerequisites:

```bash
pip install -r requirements.txt
```

### Run Single Core Experiment:

```bash
python experiments/singleCoreBenchmark.py
```

### Run Quad Core Experiment:

```bash
python experiments/quadCoreBenchmark.py
```

### Generate Publication Figures:

```bash
python experiments/generateFigures.py
```

### Docker Experiments:

```bash
cd docker
docker-compose up --build
```

## The Solution:

The **Blocking Ratio Metric** enables runtime detection of workload characteristics:

```
beta = 1 - (cpu_time / wall_time)
```

- **beta near 0:** CPU bound workload. Keep threads low to avoid GIL contention.
- **beta near 1:** I/O bound workload. Scale threads up for parallel waiting.

The `AdaptiveThreadPoolExecutor` uses Hill Climbing optimization to find the optimal thread count dynamically.

## Target Platforms:

- **Raspberry Pi 4:** ARM Cortex A72, 4 cores.
- **NVIDIA Jetson Nano:** ARM Cortex A57, 4 cores with GPU.
- **Docker containers:** CPU constrained simulation on any platform.

## Documentation:

See [Usage.md](Usage.md) for detailed instructions and [CodeBaseIndex.md](CodeBaseIndex.md) for codebase navigation.

## License:

Research code for academic purposes.
