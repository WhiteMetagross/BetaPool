#!/usr/bin/env python3
# Publication Figure Generator for Edge GIL Paper.
# Generates all publication quality figures from experimental data.
# Output conforms to IEEE/ACM style guidelines.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
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
    ax.legend(loc="upper left", fontsize=7)
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
    ax.legend(loc="upper left", fontsize=7)
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
    ax.legend(handles=legendElements, loc="lower left", fontsize=7)
    
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

    # Architecture Figures (static, not derived from CSVs).
    print("Generating Architecture Figures.")
    _generate_system_architecture()
    _generate_controller_flow()
    print("  Saved figures/fig_architecture.pdf and figures/fig_controller_flow.pdf.")


# ---------------------------------------------------------------------------
# Architecture diagrams (static). Publication quality using matplotlib.
# Designed for IEEE/ACM conference proceedings.
# ---------------------------------------------------------------------------

def _generate_system_architecture():
    """
    Generate a professional 3-tier system architecture diagram.
    Application Layer -> Adaptive Thread Pool Controller -> Worker Threads -> Python Interpreter (GIL)
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Professional color palette (IEEE style)
    colors = {
        'app': '#4472C4',        # Blue
        'controller': '#7030A0', # Purple  
        'workers': '#00B050',    # Green
        'gil': '#C00000',        # Red
        'internal': '#FFFFFF',
        'text_light': '#FFFFFF',
        'text_dark': '#333333',
        'arrow': '#404040',
        'shadow': '#CCCCCC'
    }
    
    # Helper to draw 3D effect box
    def draw_box(x, y, w, h, color, label, sublabel=None, fontcolor='white'):
        # Shadow
        shadow = patches.FancyBboxPatch((x+0.08, y-0.08), w, h, 
                                        boxstyle="round,pad=0.02,rounding_size=0.15",
                                        facecolor=colors['shadow'], edgecolor='none', zorder=1)
        ax.add_patch(shadow)
        # Main box
        box = patches.FancyBboxPatch((x, y), w, h,
                                     boxstyle="round,pad=0.02,rounding_size=0.15",
                                     facecolor=color, edgecolor='none', zorder=2)
        ax.add_patch(box)
        # Label
        if sublabel:
            ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
                   fontsize=8, fontweight='bold', color=fontcolor, zorder=3)
            ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha='center', va='center',
                   fontsize=6, color=fontcolor, style='italic', zorder=3)
        else:
            ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                   fontsize=8, fontweight='bold', color=fontcolor, zorder=3)
    
    def draw_inner_box(x, y, w, h, label, sublabel=None):
        box = patches.FancyBboxPatch((x, y), w, h,
                                     boxstyle="round,pad=0.02,rounding_size=0.1",
                                     facecolor='#F8F8F8', edgecolor='#666666', 
                                     linewidth=0.8, zorder=4)
        ax.add_patch(box)
        if sublabel:
            ax.text(x + w/2, y + h/2 + 0.12, label, ha='center', va='center',
                   fontsize=6.5, fontweight='bold', color=colors['text_dark'], zorder=5)
            ax.text(x + w/2, y + h/2 - 0.15, sublabel, ha='center', va='center',
                   fontsize=5.5, color='#666666', zorder=5)
        else:
            ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                   fontsize=6.5, fontweight='bold', color=colors['text_dark'], zorder=5)
    
    def draw_arrow(x1, y1, x2, y2, label=None, label_pos='above'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.2,
                                  shrinkA=2, shrinkB=2), zorder=6)
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            offset = 0.22 if label_pos == 'above' else -0.22
            ax.text(mid_x, mid_y + offset, label, ha='center', va='center',
                   fontsize=5.5, color='#555555', zorder=7,
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                            edgecolor='none', alpha=0.9))
    
    # Layer 1: Application Layer
    draw_box(0.3, 1.8, 2.2, 2.4, colors['app'], 'Application', 'Layer')
    
    # Layer 2: Adaptive Controller (large container)
    draw_box(3.2, 0.8, 5.8, 4.2, colors['controller'], '', '')
    ax.text(6.1, 4.7, 'Adaptive Thread Pool Controller', ha='center', va='center',
           fontsize=9, fontweight='bold', color=colors['controller'], zorder=10)
    
    # Controller internals
    draw_inner_box(3.5, 3.4, 1.6, 1.2, 'Instrumentor', 'Wrap tasks')
    draw_inner_box(5.4, 3.4, 1.5, 1.2, 'Monitor', 'Compute β')
    draw_inner_box(7.2, 3.4, 1.5, 1.2, 'Controller', 'Veto logic')
    draw_inner_box(3.5, 1.2, 5.2, 1.6, 'Metrics Engine', 'Queue depth, Blocking ratio β, Thread utilization')
    
    # Internal arrows
    ax.annotate('', xy=(5.4, 4.0), xytext=(5.1, 4.0),
               arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8), zorder=6)
    ax.annotate('', xy=(7.2, 4.0), xytext=(6.9, 4.0),
               arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8), zorder=6)
    ax.annotate('', xy=(6.1, 2.8), xytext=(6.1, 3.4),
               arrowprops=dict(arrowstyle='<->', color='#888888', lw=0.8), zorder=6)
    
    # Layer 3: Worker Threads
    draw_box(9.7, 1.8, 2.0, 2.4, colors['workers'], 'Worker', 'Threads')
    # Thread indicators
    for i in range(3):
        circle = patches.Circle((10.7, 2.3 + i*0.65), 0.18, 
                                facecolor='white', edgecolor=colors['workers'], linewidth=1, zorder=8)
        ax.add_patch(circle)
        ax.text(10.7, 2.3 + i*0.65, f'T{i+1}', ha='center', va='center', fontsize=4.5, zorder=9)
    
    # Layer 4: Python Interpreter
    draw_box(12.4, 1.8, 2.2, 2.4, colors['gil'], 'Python', 'Interpreter')
    # GIL indicator
    gil_box = patches.FancyBboxPatch((12.7, 2.1), 1.6, 0.5,
                                     boxstyle="round,pad=0.02,rounding_size=0.08",
                                     facecolor='#FF6666', edgecolor='white', 
                                     linewidth=1, zorder=8)
    ax.add_patch(gil_box)
    ax.text(13.5, 2.35, 'GIL', ha='center', va='center', fontsize=6, 
           fontweight='bold', color='white', zorder=9)
    
    # Main flow arrows
    draw_arrow(2.5, 3.0, 3.2, 3.0, 'Submit', 'above')
    draw_arrow(9.0, 3.0, 9.7, 3.0, 'Execute', 'above')
    draw_arrow(11.7, 3.0, 12.4, 3.0, 'Acquire', 'above')
    
    # Feedback arrow (curved)
    ax.annotate('', xy=(9.0, 2.0), xytext=(9.7, 2.0),
               arrowprops=dict(arrowstyle='<-', color=colors['arrow'], lw=1.0,
                              connectionstyle='arc3,rad=0.3'), zorder=6)
    ax.text(9.35, 1.55, 'Scale', ha='center', va='center', fontsize=5, color='#555555')
    
    plt.tight_layout(pad=0.5)
    plt.savefig("figures/fig_architecture.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig_architecture.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _generate_controller_flow():
    """
    Generate a professional flowchart for the controller decision logic.
    Standard flowchart symbols: ovals (start/end), diamonds (decisions), rectangles (actions).
    """
    fig, ax = plt.subplots(figsize=(5.5, 6.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 14)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Professional colors
    colors = {
        'start': '#4472C4',      # Blue
        'decision': '#FFC000',   # Yellow/Gold
        'action_pos': '#00B050', # Green
        'action_neg': '#C00000', # Red
        'action_neu': '#7030A0', # Purple
        'end': '#404040',        # Dark gray
        'arrow': '#404040',
        'text': '#333333'
    }
    
    def draw_oval(cx, cy, w, h, color, label, fontcolor='white'):
        oval = patches.Ellipse((cx, cy), w, h, facecolor=color, edgecolor='none', zorder=2)
        shadow = patches.Ellipse((cx+0.06, cy-0.06), w, h, facecolor='#CCCCCC', edgecolor='none', zorder=1)
        ax.add_patch(shadow)
        ax.add_patch(oval)
        ax.text(cx, cy, label, ha='center', va='center', fontsize=7, 
               fontweight='bold', color=fontcolor, zorder=3)
    
    def draw_diamond(cx, cy, w, h, color, label):
        pts = [(cx, cy-h/2), (cx+w/2, cy), (cx, cy+h/2), (cx-w/2, cy)]
        shadow = patches.Polygon([(p[0]+0.06, p[1]-0.06) for p in pts], 
                                facecolor='#CCCCCC', edgecolor='none', zorder=1)
        diamond = patches.Polygon(pts, facecolor=color, edgecolor='#B8860B', 
                                 linewidth=1.2, zorder=2)
        ax.add_patch(shadow)
        ax.add_patch(diamond)
        ax.text(cx, cy, label, ha='center', va='center', fontsize=6.5, 
               color=colors['text'], zorder=3, linespacing=1.2)
    
    def draw_rect(cx, cy, w, h, color, label, fontcolor='white'):
        shadow = patches.FancyBboxPatch((cx-w/2+0.06, cy-h/2-0.06), w, h,
                                        boxstyle="round,pad=0.02,rounding_size=0.12",
                                        facecolor='#CCCCCC', edgecolor='none', zorder=1)
        rect = patches.FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                      boxstyle="round,pad=0.02,rounding_size=0.12",
                                      facecolor=color, edgecolor='none', zorder=2)
        ax.add_patch(shadow)
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha='center', va='center', fontsize=6.5, 
               fontweight='bold', color=fontcolor, zorder=3, linespacing=1.1)
    
    def draw_arrow(x1, y1, x2, y2, label=None, label_side='right'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.3,
                                  shrinkA=3, shrinkB=3), zorder=5)
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            if label_side == 'right':
                offset_x, offset_y = 0.4, 0
            elif label_side == 'left':
                offset_x, offset_y = -0.4, 0
            else:
                offset_x, offset_y = 0, 0.25
            ax.text(mid_x + offset_x, mid_y + offset_y, label, ha='center', va='center',
                   fontsize=6, fontweight='bold', color='#666666', zorder=6)
    
    def draw_line(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color=colors['arrow'], lw=1.3, solid_capstyle='round', zorder=4)
    
    # Nodes
    # Start
    draw_oval(5.5, 13, 2.2, 0.9, colors['start'], 'START')
    
    # Decision 1: Queue Check
    draw_diamond(5.5, 10.8, 3.2, 1.8, colors['decision'], 'Queue\nLength > 0?')
    
    # Decision 2: Beta Check
    draw_diamond(8.0, 8.2, 3.2, 1.8, colors['decision'], 'β > βthreshold?')
    
    # Actions
    draw_rect(2.5, 8.2, 2.2, 1.1, colors['action_neu'], 'Scale Down\n(if idle)')
    draw_rect(8.0, 5.5, 2.2, 1.1, colors['action_pos'], 'Scale Up\n(+1 thread)')
    draw_rect(5.5, 5.5, 2.2, 1.1, colors['action_neg'], 'VETO\n(no scale)')
    
    # Sleep
    draw_rect(5.5, 2.8, 2.5, 1.1, '#666666', 'Sleep Δt\n(500 ms)')
    
    # Loop indicator
    draw_oval(5.5, 0.8, 1.8, 0.7, colors['end'], 'LOOP', fontcolor='white')
    
    # Arrows
    # Start -> Queue
    draw_arrow(5.5, 12.55, 5.5, 11.7)
    
    # Queue -> No -> Scale Down
    draw_line([(3.9, 10.8), (2.5, 10.8)])
    draw_arrow(2.5, 10.8, 2.5, 8.75, 'No', 'left')
    
    # Queue -> Yes -> Beta
    draw_line([(7.1, 10.8), (8.0, 10.8)])
    draw_arrow(8.0, 10.8, 8.0, 9.1, 'Yes', 'right')
    
    # Beta -> Yes -> Scale Up
    draw_arrow(8.0, 7.3, 8.0, 6.05, 'Yes', 'right')
    
    # Beta -> No -> Veto
    draw_line([(6.4, 8.2), (5.5, 8.2)])
    draw_arrow(5.5, 8.2, 5.5, 6.05, 'No', 'left')
    
    # All actions -> Sleep
    draw_line([(2.5, 7.65), (2.5, 2.8), (4.25, 2.8)])
    draw_line([(5.5, 4.95), (5.5, 3.35)])
    draw_line([(8.0, 4.95), (8.0, 2.8), (6.75, 2.8)])
    
    # Sleep -> Loop
    draw_arrow(5.5, 2.25, 5.5, 1.15)
    
    # Loop back (curved line on left side)
    draw_line([(4.6, 0.8), (1.0, 0.8), (1.0, 13.0), (4.4, 13.0)])
    # Add small arrow at the end
    ax.annotate('', xy=(4.4, 13.0), xytext=(3.8, 13.0),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.3), zorder=5)
    
    plt.tight_layout(pad=0.5)
    plt.savefig("figures/fig_controller_flow.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig_controller_flow.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
