# Reproducibility Guide:

This document provides detailed instructions for reproducing the experimental results presented in the Edge-GIL research paper.

## Prerequisites:

### Software Requirements:

- Python 3.8 or higher.
- Linux environment (WSL2 on Windows is acceptable).
- Docker (optional, for containerized experiments).

### Hardware Requirements:

For authentic edge device results:
- Raspberry Pi 4B (4-core ARM Cortex-A72).
- NVIDIA Jetson Nano (4-core ARM Cortex-A57).

For simulated edge device results:
- Any x86 or ARM64 machine with Docker support.

## Installation:

### Clone and Setup:

```bash
git clone https://github.com/[username]/edge-gil.git
cd edge-gil

python -m venv venv
source venv/bin/activate  # Linux/macOS/WSL.
# venv\Scripts\activate    # Windows.:

pip install -r requirements.txt
```

## Running Experiments:

### Single-Core Characterization:

This experiment characterizes the saturation cliff on a single-core edge simulation:

```bash
python experiments/singleCoreBenchmark.py
```

**Output files:**
- `results/mixed_workload.csv`: Mixed CPU+I/O workload data.
- `results/io_baseline.csv`: Pure I/O baseline data.

### Quad-Core Characterization:

This experiment validates the cliff persists on quad-core edge devices:

```bash
python experiments/quadCoreBenchmark.py
```

**Output files:**
- `results/quadcore_mixed_workload.csv`: Quad-core mixed workload data.
- `results/quadcore_io_baseline.csv`: Quad-core I/O baseline data.

### Figure Generation:

Generate publication-quality figures from experimental data:

```bash
python experiments/generateFigures.py
```

**Output files:**
- `figures/fig1_saturation_cliff.pdf`: Main cliff result.
- `figures/fig2_latency_analysis.pdf`: Latency breakdown.
- `figures/fig3_efficiency.pdf`: Per-thread efficiency.
- `figures/fig4_solution_comparison.pdf`: Strategy comparison.
- `figures/fig5_combined_panel.pdf`: Combined 2x2 panel.

## Docker-Based Reproduction:

For reproducible results on any platform, use Docker:

```bash
cd docker

# Build all images.:
docker-compose build

# Run single-core experiment.:
docker-compose up single-core

# Run quad-core experiment.:
docker-compose up quad-core

# Run both experiments.:
docker-compose up
```

### CPU Constraint Explanation:

The Docker containers use CPU constraints to simulate edge devices:
- Single-core: `--cpus=1` limits to one CPU.
- Quad-core: `--cpus=4` limits to four CPUs.
- Memory: `--memory=512m` simulates edge memory constraints.

## Platform-Specific Instructions:

### Raspberry Pi 4:

```bash
# Transfer code to Pi.:
scp -r edge-gil/ pi@raspberrypi:~/

# SSH and run.:
ssh pi@raspberrypi
cd edge-gil
pip3 install -r requirements.txt
python3 platforms/raspberryPiBenchmark.py
```

### NVIDIA Jetson Nano:

```bash
# Transfer code to Jetson.:
scp -r edge-gil/ jetson@jetson-nano:~/

# SSH and run.:
ssh jetson@jetson-nano
cd edge-gil
pip3 install -r requirements.txt
python3 platforms/jetsonNanoBenchmark.py
```

## Expected Results:

### Single-Core Cliff:

| Threads | Expected TPS | Expected Drop |
|---------|--------------|---------------|
| 32      | ~37,000      | Peak (0%)     |
| 128     | ~34,000      | -8%           |
| 512     | ~30,000      | -19%          |
| 2048    | ~25,000      | -32%          |

### Quad-Core Cliff:

| Threads | Expected TPS | Expected Drop |
|---------|--------------|---------------|
| 32      | ~70,000      | Peak (0%)     |
| 128     | ~60,000      | -14%          |
| 512     | ~50,000      | -28%          |
| 2048    | ~45,000      | -35%          |

## Troubleshooting:

### Permission Denied for sched_setaffinity:

On some systems, CPU pinning requires elevated privileges:

```bash
sudo python experiments/singleCoreBenchmark.py
```

Or run inside Docker where the container has the necessary capabilities.

### Results Vary Significantly:

- Ensure no other CPU-intensive processes are running.
- Run experiments multiple times and average results.
- Use Docker for more consistent environments.

### Module Not Found Errors:

Ensure the virtual environment is activated:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Data Format:

All CSV files follow this format:

```
threads,run,tps,avg_lat,p99_lat
1,0,1234.56,0.81,2.34
2,0,2345.67,0.65,1.89
...
```

- `threads`: Thread count for this data point.
- `run`: Run number (0-indexed) for statistical robustness.
- `tps`: Throughput in tasks per second.
- `avg_lat`: Average latency in milliseconds.
- `p99_lat`: 99th percentile latency in milliseconds.

## Citation:

If you use this code in your research, please cite:

```bibtex
@inproceedings{edge-gil-2026,
  title={Edge-GIL: Mitigating GIL-Induced Concurrency Thrashing in Edge AI Systems},
  author={[Authors]},
  year={2026}
}
```
