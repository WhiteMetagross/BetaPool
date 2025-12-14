# Usage Guide:

This document provides comprehensive instructions for running the GIL Saturation Cliff experiments and using the Adaptive Thread Pool in your own applications.

---

## Table of Contents:

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Running Experiments](#running-experiments)
4. [Docker Reproducibility](#docker-reproducibility)
5. [Platform-Specific Instructions](#platform-specific-instructions)
6. [Using the Adaptive Executor](#using-the-adaptive-executor)
7. [Generating Figures](#generating-figures)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites:

### System Requirements:

- Python 3.8 or higher.
- Linux environment recommended (WSL2 on Windows works well).
- Minimum 2GB RAM for benchmarks.
- Docker (optional, for reproducible experiments).

### Python Dependencies:

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

The requirements include:

- `psutil` - System monitoring.
- `matplotlib` - Figure generation.
- `numpy` - Numerical operations (optional, for extended workloads).

---

## Quick Start:

### Running a Basic Experiment:

```bash
# Clone the repository.
git clone https://github.com/yourusername/edge-gil.git
cd edge-gil

# Install dependencies.
pip install -r requirements.txt

# Run the single-core experiment (recommended for strongest signal).
python experiments/singleCoreBenchmark.py

# Generate publication figures.
python experiments/generateFigures.py
```

### Expected Output:

The experiment will output a table showing throughput and latency at different thread counts. You should observe:

- Peak throughput at 16-32 threads.
- Declining throughput beyond the peak (the saturation cliff).
- Results saved to CSV files in the `results/` directory.

---

## Running Experiments:

### Single-Core Benchmark:

Simulates a single-core edge device (Raspberry Pi Zero, containerized workload):

```bash
python experiments/singleCoreBenchmark.py
```

This benchmark uses `os.sched_setaffinity()` to pin execution to one CPU core.

### Quad-Core Benchmark:

Simulates a quad-core edge device (Raspberry Pi 4, Jetson Nano):

```bash
python experiments/quadCoreBenchmark.py
```

This demonstrates that the saturation cliff persists even on multi-core hardware.

### Understanding the Output:

The benchmark displays a table with these columns:

| Column | Description |
|--------|-------------|
| Threads | Number of worker threads in the pool. |
| TPS | Throughput in tasks per second. |
| Avg Lat | Average task latency in milliseconds. |
| P99 Lat | 99th percentile latency. |
| Status | OK, DECLINE, DROP, or CLIFF based on degradation. |

---

## Docker Reproducibility:

### Building Docker Images:

```bash
cd docker

# Build the base image.
docker build -t edge-gil:latest -f Dockerfile ..

# Build single-core simulation.
docker build -t edge-gil:singlecore -f Dockerfile.singleCore ..

# Build quad-core simulation.
docker build -t edge-gil:quadcore -f Dockerfile.quadCore ..
```

### Running with Docker:

Single-core simulation (1 CPU, 512MB RAM):

```bash
docker run --cpus="1.0" --memory="512m" -v $(pwd)/results:/app/results edge-gil:singlecore
```

Quad-core simulation (4 CPUs, 2GB RAM):

```bash
docker run --cpus="4.0" --memory="2g" -v $(pwd)/results:/app/results edge-gil:quadcore
```

### Using Docker Compose:

Run all experiments in sequence:

```bash
cd docker
docker-compose up all
```

Run experiments separately:

```bash
docker-compose up singlecore
docker-compose up quadcore
docker-compose up figures
```

Results will be saved to the `results/` and `figures/` directories on your host machine.

---

## Platform-Specific Instructions:

### Raspberry Pi 4:

1. Ensure you are running Raspberry Pi OS (64-bit recommended).
2. Install Python dependencies:

```bash
sudo apt update
sudo apt install python3-pip
pip3 install psutil matplotlib
```

3. Run the dedicated benchmark:

```bash
python3 platforms/raspberryPiBenchmark.py
```

4. Expected cliff severity: 30-50% degradation at high thread counts.

### NVIDIA Jetson Nano:

1. Ensure JetPack SDK is installed.
2. Install dependencies:

```bash
pip3 install psutil matplotlib
```

3. Run the dedicated benchmark:

```bash
python3 platforms/jetsonNanoBenchmark.py
```

4. The benchmark includes thermal monitoring to detect throttling.

### Windows (WSL2):

1. Enable WSL2 and install Ubuntu.
2. Run experiments inside WSL2 for accurate CPU affinity:

```bash
wsl python3 experiments/singleCoreBenchmark.py
```

3. Native Windows may show weaker cliff due to different scheduler behavior.

---

## Using the Adaptive Executor:

### Basic Usage:

```python
from src.adaptiveExecutor import AdaptiveThreadPoolExecutor

# Create an adaptive executor with bounds.
with AdaptiveThreadPoolExecutor(minWorkers=4, maxWorkers=64) as executor:
    # Submit tasks as you would with ThreadPoolExecutor.
    futures = [executor.submit(myTask, arg) for arg in myArgs]
    
    # Collect results.
    results = [f.result() for f in futures]
```

### Configuration:

```python
from src.adaptiveExecutor import AdaptiveThreadPoolExecutor, ControllerConfig

# Custom configuration for specific workloads.
config = ControllerConfig(
    monitorIntervalSec=0.5,      # How often to check metrics.
    betaHighThreshold=0.7,       # Scale up if blocking ratio exceeds this.
    betaLowThreshold=0.3,        # Scale down if blocking ratio falls below this.
    scaleUpStep=2,               # Threads to add when scaling up.
    scaleDownStep=1,             # Threads to remove when scaling down.
)

executor = AdaptiveThreadPoolExecutor(
    minWorkers=4,
    maxWorkers=128,
    config=config,
    enableLogging=True,
)
```

### Monitoring:

```python
# Get current state during execution.
state = executor.controllerState
print(f"Current threads: {state.currentThreads}")
print(f"Scale-up count: {state.scaleUpCount}")
print(f"Scale-down count: {state.scaleDownCount}")
```

---

## Generating Figures:

### Publication Figures:

```bash
python experiments/generateFigures.py
```

This generates figures in the `figures/` directory:

- `fig1_saturation_cliff.pdf` - Main throughput cliff visualization.
- `fig2_latency_analysis.pdf` - Latency degradation at high thread counts.
- `fig3_efficiency.pdf` - Per-thread efficiency analysis.
- `fig4_solution_comparison.pdf` - Adaptive vs static strategies.

### Custom Visualization:

The figure generator uses matplotlib with publication-quality settings. Modify `experiments/generateFigures.py` to customize:

- Color schemes.
- Figure dimensions.
- Axis labels and titles.
- Font sizes.

---

## Troubleshooting:

### No Cliff Detected:

If you observe less than 10% degradation:

1. Ensure you are running on a constrained environment (single or few cores).
2. On powerful hardware, use Docker with CPU limits.
3. Try increasing CPU_ITERATIONS to amplify the CPU-bound phase.

### Permission Errors on Linux:

If `sched_setaffinity` fails:

1. Run with elevated privileges: `sudo python3 ...`
2. Or use Docker which handles CPU constraints at the container level.

### Import Errors:

Ensure the PYTHONPATH includes the project root:

```bash
export PYTHONPATH=/path/to/edge-gil:$PYTHONPATH
```

Or run from the project root directory.

### Docker Build Failures:

Ensure Docker has access to the context:

```bash
cd /path/to/edge-gil
docker build -t edge-gil -f docker/Dockerfile .
```

---

## Running Tests:

Unit tests validate the core components:

```bash
python -m pytest tests/
```

Or run individual test files:

```bash
python -m pytest tests/testAdaptiveExecutor.py -v
```

---

## Contact and Support:

For issues or questions:

1. Open an issue on the GitHub repository.
2. Check the paper in `docs/paper.md` for theoretical background.
3. Review `docs/reproducibility.md` for detailed experimental methodology.
