#!/usr/bin/env python3
# Publication Figure Generator for Edge GIL Paper.
# Generates all publication quality figures from experimental data.
# Output conforms to IEEE/ACM style guidelines.

import matplotlib.pyplot as plt
import matplotlib
import random
import csv
import os
import statistics
from collections import defaultdict

# Set seed for reproducibility.
SEED = 17
random.seed(SEED)

# Use non interactive backend for server environments.
matplotlib.use("Agg")

# IEEE/ACM publication style configuration.
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colorblind friendly color palette.
COLORS = {
    "single": "#1f77b4",    # Blue for single core.
    "quad": "#ff7f0e",      # Orange for quad core.
    "io": "#2ca02c",        # Green for I/O baseline.
    "adaptive": "#9467bd",  # Purple for adaptive strategy.
    "naive": "#d62728",     # Red for naive strategy.
    "informed": "#17becf",  # Cyan for informed strategy.
}


def loadCsv(filepath):
    # Load CSV data into dictionary of lists.
    data = {"threads": [], "tps": [], "p99Lat": [], "avgLat": [], "run": []}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["threads"].append(int(row["threads"]))
            data["tps"].append(float(row["tps"]))
            data["p99Lat"].append(float(row.get("p99_lat", row.get("avg_lat", 0))))
            data["avgLat"].append(float(row["avg_lat"]))
            data["run"].append(int(row.get("run", 0)))
    return data


def loadSolutionCsv(filepath):
    # Load solution comparison CSV data.
    data = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "strategy": row["strategy"],
                "threads": row["threads"],
                "run": int(row["run"]),
                "tps": float(row["tps"]),
                "p99Lat": float(row["p99_lat"]),
                "avgLat": float(row["avg_lat"]),
            })
    return data


def aggregateByThreads(data):
    # Average multiple runs per thread count for statistical robustness.
    grouped = defaultdict(lambda: {"tps": [], "p99Lat": [], "avgLat": []})
    for i, t in enumerate(data["threads"]):
        grouped[t]["tps"].append(data["tps"][i])
        grouped[t]["p99Lat"].append(data["p99Lat"][i])
        grouped[t]["avgLat"].append(data["avgLat"][i])
    
    result = {"threads": [], "tps": [], "tpsStd": [], "p99Lat": [], "avgLat": []}
    for t in sorted(grouped.keys()):
        result["threads"].append(t)
        result["tps"].append(statistics.mean(grouped[t]["tps"]))
        result["tpsStd"].append(statistics.stdev(grouped[t]["tps"]) if len(grouped[t]["tps"]) > 1 else 0)
        result["p99Lat"].append(statistics.mean(grouped[t]["p99Lat"]))
        result["avgLat"].append(statistics.mean(grouped[t]["avgLat"]))
    return result


def aggregateSolutionData(data):
    # Aggregate solution comparison data by strategy name.
    grouped = defaultdict(lambda: {"tps": [], "p99Lat": []})
    for row in data:
        s = row["strategy"]
        grouped[s]["tps"].append(row["tps"])
        grouped[s]["p99Lat"].append(row["p99Lat"])
        grouped[s]["threads"] = row["threads"]
    
    result = {}
    for s, vals in grouped.items():
        result[s] = {
            "tps": statistics.mean(vals["tps"]),
            "tpsStd": statistics.stdev(vals["tps"]) if len(vals["tps"]) > 1 else 0,
            "p99": statistics.mean(vals["p99Lat"]),
            "threads": vals["threads"],
        }
    return result


def main():
    # Main entry point for figure generation.
    os.makedirs("figures", exist_ok=True)
    
    print("Loading experimental data.")
    
    # Load all experimental data files.
    singleMixed = aggregateByThreads(loadCsv("results/mixed_workload.csv"))
    singleIo = aggregateByThreads(loadCsv("results/io_baseline.csv"))
    quadMixed = aggregateByThreads(loadCsv("results/quadcore_mixed_workload.csv"))
    quadIo = aggregateByThreads(loadCsv("results/quadcore_io_baseline.csv"))
    solutionData = aggregateSolutionData(loadSolutionCsv("results/solution_comparison_v2.csv"))
    
    # Figure 1: The Saturation Cliff (Main Result).
    print("Generating Figure 1: Saturation Cliff.")
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    
    # Panel (a): Single-core.
    ax = axes[0]
    ax.errorbar(singleMixed["threads"], singleMixed["tps"],
                yerr=singleMixed["tpsStd"], fmt="o-", color=COLORS["single"],
                capsize=2, linewidth=1.5, markersize=4, label="Mixed CPU+I/O")
    ax.plot(singleIo["threads"], singleIo["tps"], "s--", color=COLORS["io"],
            linewidth=1.5, markersize=4, label="Pure I/O (baseline)")
    
    # Mark peak throughput.
    peakIdx = singleMixed["tps"].index(max(singleMixed["tps"]))
    peakT = singleMixed["threads"][peakIdx]
    peakTps = singleMixed["tps"][peakIdx]
    ax.axvline(x=peakT, color="gray", linestyle=":", alpha=0.5)
    ax.annotate(f"Peak\n({peakT} threads)", xy=(peakT, peakTps),
                xytext=(peakT*2, peakTps*0.85), fontsize=7,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))
    
    # Shade cliff region.
    ax.axvspan(singleMixed["threads"][peakIdx], 
               singleMixed["threads"][-1], alpha=0.1, color="red", label="Cliff region")
    
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Thread Count")
    ax.set_ylabel("Throughput (tasks/sec)")
    ax.set_title("(a) Single-Core Edge Device", fontweight="bold", fontsize=10)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(bottom=0)
    
    # Cliff annotation.
    cliffPct = (max(singleMixed["tps"]) - singleMixed["tps"][-1]) / max(singleMixed["tps"]) * 100
    ax.text(0.95, 0.05, f"Cliff: -{cliffPct:.1f}%", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color="red", ha="right", va="bottom")
    
    # Panel (b): Quad-core.
    ax = axes[1]
    ax.errorbar(quadMixed["threads"], quadMixed["tps"],
                yerr=quadMixed["tpsStd"], fmt="o-", color=COLORS["quad"],
                capsize=2, linewidth=1.5, markersize=4, label="Mixed CPU+I/O")
    ax.plot(quadIo["threads"], quadIo["tps"], "s--", color=COLORS["io"],
            linewidth=1.5, markersize=4, label="Pure I/O (baseline)")
    
    peakIdx = quadMixed["tps"].index(max(quadMixed["tps"]))
    peakT = quadMixed["threads"][peakIdx]
    peakTps = quadMixed["tps"][peakIdx]
    ax.axvline(x=peakT, color="gray", linestyle=":", alpha=0.5)
    ax.annotate(f"Peak\n({peakT} threads)", xy=(peakT, peakTps),
                xytext=(peakT*2.5, peakTps*0.85), fontsize=7,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))
    
    ax.axvspan(quadMixed["threads"][peakIdx],
               quadMixed["threads"][-1], alpha=0.1, color="red", label="Cliff region")
    
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Thread Count")
    ax.set_ylabel("Throughput (tasks/sec)")
    ax.set_title("(b) Quad-Core (Raspberry Pi 4)", fontweight="bold", fontsize=10)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(bottom=0)
    
    cliffPct = (max(quadMixed["tps"]) - quadMixed["tps"][-1]) / max(quadMixed["tps"]) * 100
    ax.text(0.95, 0.05, f"Cliff: -{cliffPct:.1f}%", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color="red", ha="right", va="bottom")
    
    plt.tight_layout()
    plt.savefig("figures/fig1_saturation_cliff.pdf", bbox_inches="tight")
    plt.savefig("figures/fig1_saturation_cliff.png", bbox_inches="tight")
    print("  Saved figures/fig1_saturation_cliff.pdf.")
    plt.close()
    
    # Figure 2: Latency Analysis.
    print("Generating Figure 2: Latency Analysis.")
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    
    # Panel (a): P99 Latency.
    ax = axes[0]
    ax.plot(singleMixed["threads"], singleMixed["p99Lat"], "o-",
            color=COLORS["single"], linewidth=1.5, markersize=4, label="Single-core")
    ax.plot(quadMixed["threads"], quadMixed["p99Lat"], "s-",
            color=COLORS["quad"], linewidth=1.5, markersize=4, label="Quad-core")
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.7, label="SLA (10ms)")
    
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Thread Count")
    ax.set_ylabel("P99 Latency (ms)")
    ax.set_title("(a) Tail Latency vs Thread Count", fontweight="bold", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    
    # Panel (b): Latency distribution at key points.
    ax = axes[1]
    configs = ["16 threads", "64 threads", "256 threads", "1024 threads"]
    singleP99 = [singleMixed["p99Lat"][singleMixed["threads"].index(t)] 
                 for t in [16, 64, 256, 1024]]
    quadP99 = [quadMixed["p99Lat"][quadMixed["threads"].index(t)]
               for t in [16, 64, 256, 1024]]
    
    x = range(len(configs))
    width = 0.35
    ax.bar([i - width/2 for i in x], singleP99, width, label="Single-core", color=COLORS["single"])
    ax.bar([i + width/2 for i in x], quadP99, width, label="Quad-core", color=COLORS["quad"])
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.7, label="SLA")
    
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=7)
    ax.set_ylabel("P99 Latency (ms)")
    ax.set_title("(b) Latency at Selected Configurations", fontweight="bold", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    
    plt.tight_layout()
    plt.savefig("figures/fig2_latency_analysis.pdf", bbox_inches="tight")
    plt.savefig("figures/fig2_latency_analysis.png", bbox_inches="tight")
    print("  Saved figures/fig2_latency_analysis.pdf.")
    plt.close()
    
    # Figure 3: Per-Thread Efficiency.
    print("Generating Figure 3: Per-Thread Efficiency.")
    
    fig, ax = plt.subplots(figsize=(4, 3))
    
    singleEff = [tps/t for t, tps in zip(singleMixed["threads"], singleMixed["tps"])]
    quadEff = [tps/t for t, tps in zip(quadMixed["threads"], quadMixed["tps"])]
    
    ax.plot(singleMixed["threads"], singleEff, "o-", color=COLORS["single"],
            linewidth=1.5, markersize=4, label="Single-core")
    ax.plot(quadMixed["threads"], quadEff, "s-", color=COLORS["quad"],
            linewidth=1.5, markersize=4, label="Quad-core")
    
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Thread Count")
    ax.set_ylabel("Efficiency (TPS / thread)")
    ax.set_title("Per-Thread Efficiency Degradation", fontweight="bold", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    plt.savefig("figures/fig3_efficiency.pdf", bbox_inches="tight")
    plt.savefig("figures/fig3_efficiency.png", bbox_inches="tight")
    print("  Saved figures/fig3_efficiency.pdf.")
    plt.close()
    
    # Figure 4: Solution Comparison.
    print("Generating Figure 4: Solution Comparison.")
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    
    # Prepare data.
    naiveStrategies = ["NAIVE-64", "NAIVE-128", "NAIVE-256", "NAIVE-512", "NAIVE-1024", "NAIVE-2048"]
    informedStrategies = ["INFORMED-16", "INFORMED-32", "INFORMED-48"]
    adaptiveStrategy = "ADAPTIVE-VETO"
    
    # Panel (a): Throughput comparison.
    ax = axes[0]
    
    naiveTps = [solutionData[s]["tps"] for s in naiveStrategies if s in solutionData]
    naiveThreads = [64, 128, 256, 512, 1024, 2048]
    informedTps = [solutionData[s]["tps"] for s in informedStrategies if s in solutionData]
    informedThreads = [16, 32, 48]
    adaptiveTps = solutionData[adaptiveStrategy]["tps"]
    
    ax.plot(naiveThreads, naiveTps, "o-", color=COLORS["naive"],
            linewidth=1.5, markersize=5, label="Naive Static")
    ax.scatter(informedThreads, informedTps, marker="s", s=50, 
               color=COLORS["informed"], zorder=5, label="Informed Static")
    ax.axhline(y=adaptiveTps, color=COLORS["adaptive"], linestyle="--",
               linewidth=2, label=f"Adaptive ({adaptiveTps:,.0f} TPS)")
    
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Thread Count")
    ax.set_ylabel("Throughput (TPS)")
    ax.set_title("(a) Throughput: Strategies Compared", fontweight="bold", fontsize=10)
    ax.legend(loc="lower left", fontsize=7)
    ax.set_ylim(bottom=10000)
    
    # Panel (b): Latency vs Throughput trade-off.
    ax = axes[1]
    
    allStrategies = list(solutionData.keys())
    for s in allStrategies:
        if "NAIVE" in s:
            color = COLORS["naive"]
            marker = "o"
        elif "INFORMED" in s:
            color = COLORS["informed"]
            marker = "s"
        else:
            color = COLORS["adaptive"]
            marker = "^"
        
        ax.scatter(solutionData[s]["tps"], solutionData[s]["p99"],
                   c=color, marker=marker, s=60, alpha=0.8)
        ax.annotate(s.replace("NAIVE-", "N").replace("INFORMED-", "I").replace("ADAPTIVE-", "A"),
                    (solutionData[s]["tps"], solutionData[s]["p99"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="SLA (10ms)")
    ax.set_xlabel("Throughput (TPS)")
    ax.set_ylabel("P99 Latency (ms)")
    ax.set_title("(b) Latency-Throughput Trade-off", fontweight="bold", fontsize=10)
    
    # Add legend manually.
    from matplotlib.lines import Line2D
    legendElements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["naive"], markersize=8, label="Naive"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["informed"], markersize=8, label="Informed"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=COLORS["adaptive"], markersize=8, label="Adaptive"),
    ]
    ax.legend(handles=legendElements, loc="upper right", fontsize=7)
    
    plt.tight_layout()
    plt.savefig("figures/fig4_solution_comparison.pdf", bbox_inches="tight")
    plt.savefig("figures/fig4_solution_comparison.png", bbox_inches="tight")
    print("  Saved figures/fig4_solution_comparison.pdf.")
    plt.close()
    
    # Figure 5: Combined 2x2 Panel for main paper.
    print("Generating Figure 5: Combined Panel.")
    
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    
    # Panel (a): Single-core cliff.
    ax = axes[0, 0]
    ax.errorbar(singleMixed["threads"], singleMixed["tps"],
                yerr=singleMixed["tpsStd"], fmt="o-", color=COLORS["single"],
                capsize=2, linewidth=1.5, markersize=4)
    ax.plot(singleIo["threads"], singleIo["tps"], "s--", color=COLORS["io"],
            linewidth=1.5, markersize=4, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Thread Count")
    ax.set_ylabel("Throughput (TPS)")
    ax.set_title("(a) Single-Core: 32.2% Cliff", fontweight="bold", fontsize=9)
    ax.set_ylim(bottom=0)
    
    # Panel (b): Quad-core cliff.
    ax = axes[0, 1]
    ax.errorbar(quadMixed["threads"], quadMixed["tps"],
                yerr=quadMixed["tpsStd"], fmt="o-", color=COLORS["quad"],
                capsize=2, linewidth=1.5, markersize=4)
    ax.plot(quadIo["threads"], quadIo["tps"], "s--", color=COLORS["io"],
            linewidth=1.5, markersize=4, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Thread Count")
    ax.set_ylabel("Throughput (TPS)")
    ax.set_title("(b) Quad-Core (Pi 4): 33.3% Cliff", fontweight="bold", fontsize=9)
    ax.set_ylim(bottom=0)
    
    # Panel (c): Latency comparison.
    ax = axes[1, 0]
    ax.plot(singleMixed["threads"], singleMixed["p99Lat"], "o-",
            color=COLORS["single"], linewidth=1.5, markersize=4, label="Single-core")
    ax.plot(quadMixed["threads"], quadMixed["p99Lat"], "s-",
            color=COLORS["quad"], linewidth=1.5, markersize=4, label="Quad-core")
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Thread Count")
    ax.set_ylabel("P99 Latency (ms)")
    ax.set_title("(c) Tail Latency Explosion", fontweight="bold", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    
    # Panel (d): Solution comparison.
    ax = axes[1, 1]
    strategies = ["NAIVE-256", "NAIVE-1024", "NAIVE-2048", "INFORMED-32", "ADAPTIVE-VETO"]
    labels = ["N-256", "N-1024", "N-2048", "I-32", "Adaptive"]
    tpsVals = [solutionData[s]["tps"] for s in strategies]
    colors = [COLORS["naive"], COLORS["naive"], COLORS["naive"], 
              COLORS["informed"], COLORS["adaptive"]]
    
    bars = ax.bar(range(len(strategies)), tpsVals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Throughput (TPS)")
    ax.set_title("(d) Strategy Comparison", fontweight="bold", fontsize=9)
    ax.set_ylim(bottom=10000)
    
    # Add value labels on bars.
    for bar, val in zip(bars, tpsVals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val/1000:.1f}k", ha="center", va="bottom", fontsize=6)
    
    plt.tight_layout()
    plt.savefig("figures/fig5_combined_panel.pdf", bbox_inches="tight")
    plt.savefig("figures/fig5_combined_panel.png", bbox_inches="tight")
    print("  Saved figures/fig5_combined_panel.pdf.")
    plt.close()
    
    # Summary output.
    print()
    print("-" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("-" * 60)
    
    print()
    print("Generated figures:")
    print("  fig1_saturation_cliff.pdf    - Main cliff result.")
    print("  fig2_latency_analysis.pdf    - Latency breakdown.")
    print("  fig3_efficiency.pdf          - Per-thread efficiency.")
    print("  fig4_solution_comparison.pdf - Strategy comparison.")
    print("  fig5_combined_panel.pdf      - 2x2 summary panel.")
    
    print()
    print("Experimental Summary:")
    singleCliff = (max(singleMixed["tps"]) - singleMixed["tps"][-1]) / max(singleMixed["tps"]) * 100
    quadCliff = (max(quadMixed["tps"]) - quadMixed["tps"][-1]) / max(quadMixed["tps"]) * 100
    
    peakIdx = singleMixed["tps"].index(max(singleMixed["tps"]))
    print(f"  Single-core: Peak {max(singleMixed['tps']):,.0f} TPS at {singleMixed['threads'][peakIdx]} threads.")
    print(f"               Cliff: {singleCliff:.1f}% at {singleMixed['threads'][-1]} threads.")
    
    peakIdx = quadMixed["tps"].index(max(quadMixed["tps"]))
    print(f"  Quad-core:   Peak {max(quadMixed['tps']):,.0f} TPS at {quadMixed['threads'][peakIdx]} threads.")
    print(f"               Cliff: {quadCliff:.1f}% at {quadMixed['threads'][-1]} threads.")
    
    print()
    print("All figures saved to figures/ directory.")


if __name__ == "__main__":
    main()
