# Visualization Module for Research Paper Graphs
# Produces publication-quality figures using matplotlib with
# consistent styling suitable for systems research papers.

import os
import csv
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np

# Configure matplotlib for publication-quality output
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

# Color palette suitable for colorblind readers and print
COLORS = {
    "adaptive": "#1f77b4",      # Blue
    "static_small": "#ff7f0e",  # Orange
    "static_large": "#2ca02c",  # Green
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "tertiary": "#2ca02c",
    "highlight": "#d62728",     # Red
    "neutral": "#7f7f7f",       # Gray
}

LINE_STYLES = {
    "adaptive": "-",
    "static_small": "--",
    "static_large": ":",
}

MARKERS = {
    "adaptive": "o",
    "static_small": "s",
    "static_large": "^",
}


class ExperimentAVisualizer:
    """
    Visualizer for Experiment A: Square Wave Stress Test.
    
    Produces dual-axis time series plots showing thread count adaptation
    in response to workload phase changes.
    """
    
    def __init__(self, resultsDir: str = "results/experiment_a"):
        """
        Initialize the visualizer.
        
        Args:
            resultsDir: Directory containing experiment results.
        """
        self.resultsDir = resultsDir
        self.outputDir = os.path.join(resultsDir, "figures")
        os.makedirs(self.outputDir, exist_ok=True)
    
    def loadData(self, filename: str) -> Dict[str, List]:
        """
        Load experiment data from CSV file.
        
        Args:
            filename: Name of the CSV file to load.
            
        Returns:
            Dictionary with column names as keys and data lists as values.
        """
        filepath = os.path.join(self.resultsDir, filename)
        data = {
            "elapsed_sec": [],
            "phase": [],
            "active_threads": [],
            "throughput_rps": [],
            "cpu_utilization": [],
            "avg_blocking_ratio": [],
        }
        
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["elapsed_sec"].append(float(row["elapsed_sec"]))
                data["phase"].append(row["phase"])
                data["active_threads"].append(int(row["active_threads"]))
                data["throughput_rps"].append(float(row["throughput_rps"]))
                data["cpu_utilization"].append(float(row["cpu_utilization"]))
                data["avg_blocking_ratio"].append(float(row["avg_blocking_ratio"]))
        
        return data
    
    def plotThreadCountVsThroughput(self) -> str:
        """
        Create dual-axis plot showing thread count and throughput over time.
        
        This is the primary figure for Experiment A, demonstrating how the
        adaptive controller responds to workload phase changes.
        
        Returns:
            Path to the saved figure.
        """
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Try to load all three configurations
        configs = [
            ("square_wave_adaptive.csv", "Adaptive", COLORS["adaptive"], "-"),
            ("square_wave_static_small.csv", "Static (4)", COLORS["static_small"], "--"),
            ("square_wave_static_large.csv", "Static (50)", COLORS["static_large"], ":"),
        ]
        
        ax2 = ax1.twinx()
        
        for filename, label, color, linestyle in configs:
            try:
                data = self.loadData(filename)
                
                # Plot thread count on left axis
                ax1.plot(
                    data["elapsed_sec"],
                    data["active_threads"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    label=f"{label} (threads)",
                    alpha=0.9,
                )
                
                # Plot throughput on right axis (lighter, for reference)
                ax2.plot(
                    data["elapsed_sec"],
                    data["throughput_rps"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1,
                    alpha=0.4,
                )
                
            except FileNotFoundError:
                print(f"Warning: {filename} not found, skipping")
                continue
        
        # Get phase boundaries from adaptive data if available
        try:
            adaptiveData = self.loadData("square_wave_adaptive.csv")
            phases = adaptiveData["phase"]
            times = adaptiveData["elapsed_sec"]
            
            # Find phase transition points
            transitionTimes = []
            for i in range(1, len(phases)):
                if phases[i] != phases[i-1]:
                    transitionTimes.append(times[i])
            
            # Draw vertical lines at phase transitions
            for t in transitionTimes:
                ax1.axvline(x=t, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            
            # Add phase labels
            if len(transitionTimes) >= 2:
                ax1.text(transitionTimes[0]/2, ax1.get_ylim()[1]*0.95, "I/O Phase",
                        ha="center", fontsize=9, style="italic")
                ax1.text((transitionTimes[0]+transitionTimes[1])/2, ax1.get_ylim()[1]*0.95,
                        "CPU Phase", ha="center", fontsize=9, style="italic")
                ax1.text((transitionTimes[1]+times[-1])/2, ax1.get_ylim()[1]*0.95,
                        "Recovery", ha="center", fontsize=9, style="italic")
        except Exception:
            pass
        
        # Configure axes
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Thread Count", color=COLORS["primary"])
        ax1.tick_params(axis="y", labelcolor=COLORS["primary"])
        ax1.set_ylim(0, None)
        
        ax2.set_ylabel("Throughput (req/sec)", color=COLORS["neutral"])
        ax2.tick_params(axis="y", labelcolor=COLORS["neutral"])
        ax2.set_ylim(0, None)
        
        # Legend
        ax1.legend(loc="upper left", framealpha=0.9)
        
        plt.title("Experiment A: Thread Count Adaptation to Workload Phase Changes")
        plt.tight_layout()
        
        # Save figure
        outputPath = os.path.join(self.outputDir, "exp_a_thread_adaptation.pdf")
        plt.savefig(outputPath, format="pdf")
        plt.savefig(outputPath.replace(".pdf", ".png"), format="png")
        plt.close()
        
        print(f"Saved: {outputPath}")
        return outputPath
    
    def plotBlockingRatioTimeSeries(self) -> str:
        """
        Create time series plot of blocking ratio showing workload classification.
        
        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        
        try:
            data = self.loadData("square_wave_adaptive.csv")
            
            ax.plot(
                data["elapsed_sec"],
                data["avg_blocking_ratio"],
                color=COLORS["primary"],
                linewidth=2,
                label="Blocking Ratio (beta)",
            )
            
            # Add threshold lines
            ax.axhline(y=0.7, color=COLORS["highlight"], linestyle="--",
                      linewidth=1, label="High threshold (I/O-bound)")
            ax.axhline(y=0.3, color=COLORS["secondary"], linestyle="--",
                      linewidth=1, label="Low threshold (CPU-bound)")
            
            # Find and mark phase transitions
            phases = data["phase"]
            times = data["elapsed_sec"]
            
            for i in range(1, len(phases)):
                if phases[i] != phases[i-1]:
                    ax.axvline(x=times[i], color="gray", linestyle=":",
                              linewidth=1, alpha=0.7)
            
        except FileNotFoundError:
            print("Warning: Adaptive data not found")
        
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Blocking Ratio (beta)")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", framealpha=0.9)
        
        plt.title("Blocking Ratio as Workload Classification Signal")
        plt.tight_layout()
        
        outputPath = os.path.join(self.outputDir, "exp_a_blocking_ratio.pdf")
        plt.savefig(outputPath, format="pdf")
        plt.savefig(outputPath.replace(".pdf", ".png"), format="png")
        plt.close()
        
        print(f"Saved: {outputPath}")
        return outputPath


class ExperimentBVisualizer:
    """
    Visualizer for Experiment B: RAG Pipeline Simulation.
    
    Produces grouped bar charts comparing P99 latency and throughput
    across different executor configurations and load levels.
    """
    
    def __init__(self, resultsDir: str = "results/experiment_b"):
        """
        Initialize the visualizer.
        
        Args:
            resultsDir: Directory containing experiment results.
        """
        self.resultsDir = resultsDir
        self.outputDir = os.path.join(resultsDir, "figures")
        os.makedirs(self.outputDir, exist_ok=True)
    
    def loadSummaryData(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Load summary data from CSV.
        
        Returns:
            Nested dict: data[executor][load_level][metric] = value
        """
        filepath = os.path.join(self.resultsDir, "rag_summary.csv")
        data = {}
        
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                executor = row["executor"]
                loadLevel = row["load_level"]
                
                if executor not in data:
                    data[executor] = {}
                
                data[executor][loadLevel] = {
                    "avg_latency_ms": float(row["avg_latency_ms"]),
                    "p50_latency_ms": float(row["p50_latency_ms"]),
                    "p90_latency_ms": float(row["p90_latency_ms"]),
                    "p99_latency_ms": float(row["p99_latency_ms"]),
                    "throughput_rps": float(row["throughput_rps"]),
                    "avg_cpu_utilization": float(row["avg_cpu_utilization"]),
                }
        
        return data
    
    def plotP99LatencyComparison(self) -> str:
        """
        Create grouped bar chart comparing P99 latency across configurations.
        
        Returns:
            Path to the saved figure.
        """
        try:
            data = self.loadSummaryData()
        except FileNotFoundError:
            print("Warning: Summary data not found")
            return ""
        
        loadLevels = ["low", "medium", "high"]
        executors = ["adaptive", "static_small", "static_large"]
        executorLabels = ["Adaptive", "Static (4)", "Static (50)"]
        
        x = np.arange(len(loadLevels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for i, (executor, label) in enumerate(zip(executors, executorLabels)):
            if executor not in data:
                continue
            
            values = []
            for load in loadLevels:
                if load in data[executor]:
                    values.append(data[executor][load]["p99_latency_ms"])
                else:
                    values.append(0)
            
            bars = ax.bar(
                x + (i - 1) * width,
                values,
                width,
                label=label,
                color=COLORS.get(executor, COLORS["neutral"]),
                edgecolor="black",
                linewidth=0.5,
            )
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(
                    f"{val:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        
        ax.set_xlabel("Load Level")
        ax.set_ylabel("P99 Latency (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels([l.capitalize() for l in loadLevels])
        ax.legend()
        ax.set_ylim(0, None)
        
        plt.title("Experiment B: P99 Latency Comparison (RAG Pipeline)")
        plt.tight_layout()
        
        outputPath = os.path.join(self.outputDir, "exp_b_p99_latency.pdf")
        plt.savefig(outputPath, format="pdf")
        plt.savefig(outputPath.replace(".pdf", ".png"), format="png")
        plt.close()
        
        print(f"Saved: {outputPath}")
        return outputPath
    
    def plotThroughputComparison(self) -> str:
        """
        Create grouped bar chart comparing throughput across configurations.
        
        Returns:
            Path to the saved figure.
        """
        try:
            data = self.loadSummaryData()
        except FileNotFoundError:
            print("Warning: Summary data not found")
            return ""
        
        loadLevels = ["low", "medium", "high"]
        executors = ["adaptive", "static_small", "static_large"]
        executorLabels = ["Adaptive", "Static (4)", "Static (50)"]
        
        x = np.arange(len(loadLevels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for i, (executor, label) in enumerate(zip(executors, executorLabels)):
            if executor not in data:
                continue
            
            values = []
            for load in loadLevels:
                if load in data[executor]:
                    values.append(data[executor][load]["throughput_rps"])
                else:
                    values.append(0)
            
            ax.bar(
                x + (i - 1) * width,
                values,
                width,
                label=label,
                color=COLORS.get(executor, COLORS["neutral"]),
                edgecolor="black",
                linewidth=0.5,
            )
        
        ax.set_xlabel("Load Level")
        ax.set_ylabel("Throughput (req/sec)")
        ax.set_xticks(x)
        ax.set_xticklabels([l.capitalize() for l in loadLevels])
        ax.legend()
        ax.set_ylim(0, None)
        
        plt.title("Experiment B: Throughput Comparison (RAG Pipeline)")
        plt.tight_layout()
        
        outputPath = os.path.join(self.outputDir, "exp_b_throughput.pdf")
        plt.savefig(outputPath, format="pdf")
        plt.savefig(outputPath.replace(".pdf", ".png"), format="png")
        plt.close()
        
        print(f"Saved: {outputPath}")
        return outputPath
    
    def plotLatencyBreakdown(self) -> str:
        """
        Create stacked bar chart showing latency breakdown by pipeline stage.
        
        Returns:
            Path to the saved figure.
        """
        # This would require loading detailed metrics
        # Placeholder for now
        return ""


class ExperimentCVisualizer:
    """
    Visualizer for Experiment C: GIL Saturation Validation.
    
    Produces dual-axis plots showing throughput and blocking ratio vs thread count,
    demonstrating GIL saturation detection.
    """
    
    def __init__(self, resultsDir: str = "results/experiment_c"):
        """
        Initialize the visualizer.
        
        Args:
            resultsDir: Directory containing experiment results.
        """
        self.resultsDir = resultsDir
        self.outputDir = os.path.join(resultsDir, "figures")
        os.makedirs(self.outputDir, exist_ok=True)
    
    def loadData(self) -> Dict[str, List]:
        """
        Load GIL saturation data from CSV.
        
        Returns:
            Dictionary with column names as keys.
        """
        filepath = os.path.join(self.resultsDir, "gil_saturation.csv")
        data = {
            "thread_count": [],
            "throughput_ops_sec": [],
            "avg_blocking_ratio": [],
            "std_blocking_ratio": [],
            "avg_cpu_utilization": [],
        }
        
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["thread_count"].append(int(row["thread_count"]))
                data["throughput_ops_sec"].append(float(row["throughput_ops_sec"]))
                data["avg_blocking_ratio"].append(float(row["avg_blocking_ratio"]))
                data["std_blocking_ratio"].append(float(row["std_blocking_ratio"]))
                data["avg_cpu_utilization"].append(float(row["avg_cpu_utilization"]))
        
        return data
    
    def plotGilSaturation(self) -> str:
        """
        Create dual-axis plot showing throughput and blocking ratio vs threads.
        
        This is the primary figure for Experiment C, demonstrating that blocking
        ratio correctly signals GIL saturation.
        
        Returns:
            Path to the saved figure.
        """
        try:
            data = self.loadData()
        except FileNotFoundError:
            print("Warning: GIL saturation data not found")
            return ""
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        
        threadCounts = data["thread_count"]
        throughput = data["throughput_ops_sec"]
        blockingRatio = data["avg_blocking_ratio"]
        
        # Plot throughput on left axis
        line1, = ax1.plot(
            threadCounts,
            throughput,
            color=COLORS["primary"],
            marker="o",
            linewidth=2,
            markersize=8,
            label="Throughput (ops/sec)",
        )
        
        # Plot blocking ratio on right axis
        line2, = ax2.plot(
            threadCounts,
            blockingRatio,
            color=COLORS["secondary"],
            marker="s",
            linewidth=2,
            markersize=8,
            label="Blocking Ratio (beta)",
        )
        
        # Add annotation for saturation point
        # Find where throughput plateaus
        maxThroughputIdx = throughput.index(max(throughput))
        ax1.annotate(
            "Peak throughput",
            xy=(threadCounts[maxThroughputIdx], throughput[maxThroughputIdx]),
            xytext=(threadCounts[maxThroughputIdx] + 2, throughput[maxThroughputIdx] * 1.1),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=COLORS["primary"]),
        )
        
        # Find where blocking ratio drops below threshold
        for i, br in enumerate(blockingRatio):
            if br < 0.1:
                ax2.annotate(
                    "GIL saturation signal",
                    xy=(threadCounts[i], blockingRatio[i]),
                    xytext=(threadCounts[i] + 3, blockingRatio[i] + 0.15),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=COLORS["secondary"]),
                )
                break
        
        # Configure axes
        ax1.set_xlabel("Number of Threads")
        ax1.set_ylabel("Throughput (ops/sec)", color=COLORS["primary"])
        ax1.tick_params(axis="y", labelcolor=COLORS["primary"])
        ax1.set_ylim(0, None)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax2.set_ylabel("Blocking Ratio (beta)", color=COLORS["secondary"])
        ax2.tick_params(axis="y", labelcolor=COLORS["secondary"])
        ax2.set_ylim(0, 1)
        
        # Combined legend
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right", framealpha=0.9)
        
        plt.title("Experiment C: GIL Saturation Detection via Blocking Ratio")
        plt.tight_layout()
        
        outputPath = os.path.join(self.outputDir, "exp_c_gil_saturation.pdf")
        plt.savefig(outputPath, format="pdf")
        plt.savefig(outputPath.replace(".pdf", ".png"), format="png")
        plt.close()
        
        print(f"Saved: {outputPath}")
        return outputPath
    
    def plotCpuVsBlockingRatio(self) -> str:
        """
        Create scatter plot comparing CPU utilization to blocking ratio.
        
        Demonstrates why blocking ratio is a better signal than CPU%.
        
        Returns:
            Path to the saved figure.
        """
        try:
            data = self.loadData()
        except FileNotFoundError:
            print("Warning: GIL saturation data not found")
            return ""
        
        fig, ax = plt.subplots(figsize=(7, 5))
        
        scatter = ax.scatter(
            data["avg_cpu_utilization"],
            data["avg_blocking_ratio"],
            c=data["thread_count"],
            cmap="viridis",
            s=100,
            edgecolors="black",
            linewidth=0.5,
        )
        
        # Add thread count labels
        for i, (cpu, br, tc) in enumerate(zip(
            data["avg_cpu_utilization"],
            data["avg_blocking_ratio"],
            data["thread_count"]
        )):
            ax.annotate(
                f"{tc}T",
                (cpu, br),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        
        cbar = plt.colorbar(scatter)
        cbar.set_label("Thread Count")
        
        ax.set_xlabel("CPU Utilization (%)")
        ax.set_ylabel("Blocking Ratio (beta)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        
        plt.title("CPU Utilization vs Blocking Ratio")
        plt.tight_layout()
        
        outputPath = os.path.join(self.outputDir, "exp_c_cpu_vs_blocking.pdf")
        plt.savefig(outputPath, format="pdf")
        plt.savefig(outputPath.replace(".pdf", ".png"), format="png")
        plt.close()
        
        print(f"Saved: {outputPath}")
        return outputPath


def generateAllFigures(resultsDir: str = "results") -> List[str]:
    """
    Generate all figures for the research paper.
    
    Args:
        resultsDir: Base directory containing experiment results.
        
    Returns:
        List of paths to generated figures.
    """
    figures = []
    
    print("="*60)
    print("Generating Publication Figures")
    print("="*60)
    
    # Experiment A figures
    print("\nExperiment A: Square Wave Stress Test")
    vizA = ExperimentAVisualizer(os.path.join(resultsDir, "experiment_a"))
    try:
        figures.append(vizA.plotThreadCountVsThroughput())
        figures.append(vizA.plotBlockingRatioTimeSeries())
    except Exception as e:
        print(f"Error generating Experiment A figures: {e}")
    
    # Experiment B figures
    print("\nExperiment B: RAG Pipeline Simulation")
    vizB = ExperimentBVisualizer(os.path.join(resultsDir, "experiment_b"))
    try:
        figures.append(vizB.plotP99LatencyComparison())
        figures.append(vizB.plotThroughputComparison())
    except Exception as e:
        print(f"Error generating Experiment B figures: {e}")
    
    # Experiment C figures
    print("\nExperiment C: GIL Saturation Validation")
    vizC = ExperimentCVisualizer(os.path.join(resultsDir, "experiment_c"))
    try:
        figures.append(vizC.plotGilSaturation())
        figures.append(vizC.plotCpuVsBlockingRatio())
    except Exception as e:
        print(f"Error generating Experiment C figures: {e}")
    
    print("\n" + "="*60)
    print(f"Generated {len([f for f in figures if f])} figures")
    print("="*60)
    
    return [f for f in figures if f]


if __name__ == "__main__":
    figures = generateAllFigures()
    
    print("\nGenerated figures:")
    for fig in figures:
        print(f"  - {fig}")
