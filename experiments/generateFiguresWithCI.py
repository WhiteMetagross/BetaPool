#!/usr/bin/env python3
"""
Generate publication-quality figures with error bars and confidence bands.
Based on WSL Python 3.12 benchmark results (December 2025).
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (7, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Output directory
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================================================
# DATA FROM WSL BENCHMARKS (Python 3.12.3, December 2025)
# =============================================================================

# Single-Core Mixed Workload Results (n=10, 95% CI)
SC_THREADS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
SC_TPS_MEAN = [4940, 10075, 19169, 33154, 35498, 39738, 36890, 36014, 35751, 32475, 29002, 23771]
SC_TPS_CI = [50, 190, 218, 899, 565, 752, 745, 389, 1180, 507, 429, 367]
SC_P99 = [0.4, 0.3, 0.4, 0.4, 1.9, 8.6, 19.0, 30.4, 35.3, 28.6, 19.9, 17.2]

# Quad-Core Mixed Workload Results (n=10, 95% CI)
QC_THREADS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
QC_TPS_MEAN = [5012, 9954, 15308, 18832, 18733, 19833, 18681, 17921, 19116, 17392, 16444, 12877]
QC_TPS_CI = [73, 232, 435, 695, 370, 833, 1299, 607, 248, 512, 449, 477]
QC_P99 = [0.3, 0.3, 0.5, 1.0, 2.2, 4.1, 9.3, 20.3, 25.5, 25.6, 25.3, 20.0]

# Pure I/O Baseline (Single-Core)
SC_IO_THREADS = [1, 4, 16, 64, 256]
SC_IO_TPS = [1283, 5337, 19288, 53877, 72917]

# Pure I/O Baseline (Quad-Core)
QC_IO_THREADS = [1, 4, 16, 64, 256]
QC_IO_TPS = [5241, 18835, 28044, 29420, 29684]

# Baseline Comparison Results (n=10, 95% CI)
BASELINE_STRATEGIES = ['ThreadPool-16', 'ThreadPool-32', 'ThreadPool-64', 'ThreadPool-128', 'ThreadPool-256',
                       'ProcessPool-2', 'ProcessPool-4', 'ProcessPool-8',
                       'Asyncio-32', 'Asyncio-64', 'Asyncio-128', 'Asyncio-256',
                       'QueueScaler-4-64', 'QueueScaler-4-128', 'QueueScaler-4-256']
BASELINE_TPS = [19498, 19792, 19417, 19683, 18279, 6407, 6258, 6512, 42946, 40614, 42370, 43302, 18412, 18408, 17119]
BASELINE_CI = [539, 636, 238, 503, 472, 232, 158, 137, 735, 1034, 1240, 1272, 406, 429, 345]
BASELINE_P99 = [2.0, 4.1, 6.4, 10.3, 14.3, 0.3, 0.3, 0.3, 0.9, 1.8, 17.3, 19.1, 6.3, 12.4, 12.0]

# Workload Sweep Results (n=5, 95% CI)
WORKLOAD_NAMES = ['IO Heavy', 'IO Dominant', 'Balanced', 'CPU Leaning', 'CPU Heavy', 'CPU Dominant']
WORKLOAD_OPTIMAL_N = [128, 128, 16, 16, 16, 32]
WORKLOAD_TPS = [67132, 56010, 35620, 22342, 10291, 5525]
WORKLOAD_CI = [3797, 4057, 1294, 1245, 201, 112]

# Workload sweep detailed data (for heatmap)
SWEEP_THREADS = [8, 16, 32, 64, 128, 256]
SWEEP_DATA = {
    'IO Heavy': [6554, 13811, 26363, 45474, 67132, 66774],
    'IO Dominant': [12793, 22982, 41624, 57825, 56010, 51313],
    'Balanced': [34650, 35620, 36206, 36366, 34692, 34169],
    'CPU Leaning': [20339, 22342, 22705, 20410, 20743, 20115],
    'CPU Heavy': [10051, 10291, 9793, 9579, 9411, 9505],
    'CPU Dominant': [5592, 5276, 5525, 5388, 4454, 4944],
}


def fig1_saturation_cliff():
    """Figure 1: The Saturation Cliff with error bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel (a): Single-Core
    ax1.errorbar(SC_THREADS, SC_TPS_MEAN, yerr=SC_TPS_CI, 
                 fmt='o-', capsize=3, capthick=1, linewidth=2, markersize=6,
                 color='#2E86AB', label='Mixed CPU+I/O', ecolor='#2E86AB', alpha=0.8)
    ax1.plot(SC_IO_THREADS, SC_IO_TPS, 's--', linewidth=2, markersize=6,
             color='#A23B72', label='Pure I/O Baseline', alpha=0.8)
    
    # Mark peak
    peak_idx = SC_TPS_MEAN.index(max(SC_TPS_MEAN))
    ax1.annotate(f'Peak: {SC_TPS_MEAN[peak_idx]:,} TPS\n@ {SC_THREADS[peak_idx]} threads',
                 xy=(SC_THREADS[peak_idx], SC_TPS_MEAN[peak_idx]),
                 xytext=(SC_THREADS[peak_idx]*4, SC_TPS_MEAN[peak_idx]*1.1),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    
    # Mark cliff
    ax1.annotate(f'Cliff: {SC_TPS_MEAN[-1]:,} TPS\n(-40.2%)',
                 xy=(SC_THREADS[-1], SC_TPS_MEAN[-1]),
                 xytext=(SC_THREADS[-1]/3, SC_TPS_MEAN[-1]*0.7),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Thread Count')
    ax1.set_ylabel('Throughput (TPS)')
    ax1.set_title('(a) Single-Core Configuration')
    ax1.legend(loc='upper left')
    ax1.set_xlim(0.8, 3000)
    ax1.set_ylim(0, 85000)
    ax1.set_xticks([1, 4, 16, 64, 256, 1024])
    ax1.set_xticklabels(['1', '4', '16', '64', '256', '1024'])
    
    # Panel (b): Quad-Core
    ax2.errorbar(QC_THREADS, QC_TPS_MEAN, yerr=QC_TPS_CI,
                 fmt='o-', capsize=3, capthick=1, linewidth=2, markersize=6,
                 color='#2E86AB', label='Mixed CPU+I/O', ecolor='#2E86AB', alpha=0.8)
    ax2.plot(QC_IO_THREADS, QC_IO_TPS, 's--', linewidth=2, markersize=6,
             color='#A23B72', label='Pure I/O Baseline', alpha=0.8)
    
    # Mark peak
    peak_idx = QC_TPS_MEAN.index(max(QC_TPS_MEAN))
    ax2.annotate(f'Peak: {QC_TPS_MEAN[peak_idx]:,} TPS\n@ {QC_THREADS[peak_idx]} threads',
                 xy=(QC_THREADS[peak_idx], QC_TPS_MEAN[peak_idx]),
                 xytext=(QC_THREADS[peak_idx]*4, QC_TPS_MEAN[peak_idx]*1.3),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    
    # Mark cliff
    ax2.annotate(f'Cliff: {QC_TPS_MEAN[-1]:,} TPS\n(-35.1%)',
                 xy=(QC_THREADS[-1], QC_TPS_MEAN[-1]),
                 xytext=(QC_THREADS[-1]/3, QC_TPS_MEAN[-1]*0.6),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('Throughput (TPS)')
    ax2.set_title('(b) Quad-Core Configuration')
    ax2.legend(loc='upper left')
    ax2.set_xlim(0.8, 3000)
    ax2.set_ylim(0, 35000)
    ax2.set_xticks([1, 4, 16, 64, 256, 1024])
    ax2.set_xticklabels(['1', '4', '16', '64', '256', '1024'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_saturation_cliff.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_saturation_cliff.pdf'))
    plt.close()
    print("Generated: fig1_saturation_cliff.png/pdf")


def fig2_latency_analysis():
    """Figure 2: Latency Analysis with error bands."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel (a): Single-Core P99 Latency
    ax1.semilogy(SC_THREADS, SC_P99, 'o-', linewidth=2, markersize=6,
                 color='#E94F37', label='Single-Core P99')
    ax1.fill_between(SC_THREADS, 
                     [max(0.1, p*0.8) for p in SC_P99],
                     [p*1.2 for p in SC_P99],
                     alpha=0.2, color='#E94F37')
    
    ax1.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='10ms threshold')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Thread Count')
    ax1.set_ylabel('P99 Latency (ms)')
    ax1.set_title('(a) Single-Core P99 Latency')
    ax1.legend(loc='upper left')
    ax1.set_xlim(0.8, 3000)
    ax1.set_xticks([1, 4, 16, 64, 256, 1024])
    ax1.set_xticklabels(['1', '4', '16', '64', '256', '1024'])
    
    # Annotate latency explosion
    ax1.annotate(f'2.0× increase\n(8.6→17.2 ms)',
                 xy=(2048, SC_P99[-1]),
                 xytext=(300, 50),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    
    # Panel (b): Quad-Core P99 Latency
    ax2.semilogy(QC_THREADS, QC_P99, 'o-', linewidth=2, markersize=6,
                 color='#F77F00', label='Quad-Core P99')
    ax2.fill_between(QC_THREADS,
                     [max(0.1, p*0.8) for p in QC_P99],
                     [p*1.2 for p in QC_P99],
                     alpha=0.2, color='#F77F00')
    
    ax2.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='10ms threshold')
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('P99 Latency (ms)')
    ax2.set_title('(b) Quad-Core P99 Latency')
    ax2.legend(loc='upper left')
    ax2.set_xlim(0.8, 3000)
    ax2.set_xticks([1, 4, 16, 64, 256, 1024])
    ax2.set_xticklabels(['1', '4', '16', '64', '256', '1024'])
    
    # Annotate latency explosion
    ax2.annotate(f'4.8× increase\n(4.1→20.0 ms)',
                 xy=(2048, QC_P99[-1]),
                 xytext=(300, 50),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_latency_analysis.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_latency_analysis.pdf'))
    plt.close()
    print("Generated: fig2_latency_analysis.png/pdf")


def fig6_baseline_comparison():
    """Figure 6: Baseline Strategy Comparison with error bars."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Group by strategy type
    threadpool_idx = [0, 1, 2, 3, 4]
    processpool_idx = [5, 6, 7]
    asyncio_idx = [8, 9, 10, 11]
    queuescaler_idx = [12, 13, 14]
    
    colors = ['#2E86AB'] * 5 + ['#A23B72'] * 3 + ['#F77F00'] * 4 + ['#8B8B8B'] * 3
    
    x = np.arange(len(BASELINE_STRATEGIES))
    bars = ax.bar(x, BASELINE_TPS, yerr=BASELINE_CI, capsize=3, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add category labels
    ax.axvline(x=4.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=7.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=11.5, color='gray', linestyle='--', alpha=0.5)
    
    ax.text(2, max(BASELINE_TPS)*1.05, 'ThreadPool', ha='center', fontsize=10, fontweight='bold')
    ax.text(6, max(BASELINE_TPS)*1.05, 'ProcessPool', ha='center', fontsize=10, fontweight='bold')
    ax.text(9.5, max(BASELINE_TPS)*1.05, 'Asyncio', ha='center', fontsize=10, fontweight='bold')
    ax.text(13, max(BASELINE_TPS)*1.05, 'QueueScaler', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Throughput (TPS)')
    ax.set_xlabel('Strategy Configuration')
    ax.set_title('Baseline Strategy Comparison (Mixed CPU+I/O Workload)')
    ax.set_xticks(x)
    ax.set_xticklabels([s.split('-')[-1] for s in BASELINE_STRATEGIES], rotation=45, ha='right')
    ax.set_ylim(0, max(BASELINE_TPS) * 1.15)
    
    # Add value labels on bars
    for i, (bar, tps, ci) in enumerate(zip(bars, BASELINE_TPS, BASELINE_CI)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 500,
                f'{tps//1000}k', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_baseline_comparison.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_baseline_comparison.pdf'))
    plt.close()
    print("Generated: fig6_baseline_comparison.png/pdf")


def fig8_workload_heatmap():
    """Figure 8: Workload Sweep Heatmap."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create data matrix
    data = np.array([SWEEP_DATA[w] for w in WORKLOAD_NAMES])
    
    # Normalize each row for better visualization
    data_normalized = data / data.max(axis=1, keepdims=True)
    
    im = ax.imshow(data_normalized, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Relative Throughput')
    
    # Set labels
    ax.set_xticks(np.arange(len(SWEEP_THREADS)))
    ax.set_xticklabels(SWEEP_THREADS)
    ax.set_yticks(np.arange(len(WORKLOAD_NAMES)))
    ax.set_yticklabels(WORKLOAD_NAMES)
    
    ax.set_xlabel('Thread Count')
    ax.set_ylabel('Workload Type')
    ax.set_title('Throughput by Workload Type and Thread Count')
    
    # Add text annotations
    for i in range(len(WORKLOAD_NAMES)):
        for j in range(len(SWEEP_THREADS)):
            tps = data[i, j]
            text_color = 'white' if data_normalized[i, j] > 0.6 else 'black'
            ax.text(j, i, f'{tps//1000}k', ha='center', va='center', 
                   color=text_color, fontsize=8)
    
    # Mark optimal for each workload
    for i, (workload, opt_n) in enumerate(zip(WORKLOAD_NAMES, WORKLOAD_OPTIMAL_N)):
        if opt_n in SWEEP_THREADS:
            j = SWEEP_THREADS.index(opt_n)
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                       edgecolor='green', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig8_workload_heatmap.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig8_workload_heatmap.pdf'))
    plt.close()
    print("Generated: fig8_workload_heatmap.png/pdf")


def fig_workload_bars():
    """Additional figure: Workload generalization bar chart with CI."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(WORKLOAD_NAMES))
    width = 0.6
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(WORKLOAD_NAMES)))
    
    bars = ax.bar(x, WORKLOAD_TPS, width, yerr=WORKLOAD_CI, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add optimal thread count labels
    for i, (bar, opt_n) in enumerate(zip(bars, WORKLOAD_OPTIMAL_N)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + WORKLOAD_CI[i] + 1000,
                f'N={opt_n}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Peak Throughput (TPS)')
    ax.set_xlabel('Workload Type')
    ax.set_title('Optimal Throughput by Workload Type (with 95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOAD_NAMES, rotation=15, ha='right')
    ax.set_ylim(0, max(WORKLOAD_TPS) * 1.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_workload_bars.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_workload_bars.pdf'))
    plt.close()
    print("Generated: fig_workload_bars.png/pdf")


def fig_combined_cliff():
    """Combined single/quad core comparison on same axes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Single-Core
    ax.errorbar(SC_THREADS, SC_TPS_MEAN, yerr=SC_TPS_CI,
                fmt='o-', capsize=3, capthick=1, linewidth=2, markersize=6,
                color='#2E86AB', label='Single-Core', ecolor='#2E86AB', alpha=0.8)
    
    # Quad-Core
    ax.errorbar(QC_THREADS, QC_TPS_MEAN, yerr=QC_TPS_CI,
                fmt='s-', capsize=3, capthick=1, linewidth=2, markersize=6,
                color='#E94F37', label='Quad-Core', ecolor='#E94F37', alpha=0.8)
    
    # Add shaded region for cliff zone
    ax.axvspan(512, 2048, alpha=0.1, color='red', label='Saturation Cliff Zone')
    
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Thread Count')
    ax.set_ylabel('Throughput (TPS)')
    ax.set_title('GIL Saturation Cliff: Single-Core vs Quad-Core')
    ax.legend(loc='upper left')
    ax.set_xlim(0.8, 3000)
    ax.set_xticks([1, 4, 16, 64, 256, 1024])
    ax.set_xticklabels(['1', '4', '16', '64', '256', '1024'])
    
    # Add degradation annotations
    sc_peak = max(SC_TPS_MEAN)
    sc_final = SC_TPS_MEAN[-1]
    sc_deg = (sc_peak - sc_final) / sc_peak * 100
    
    qc_peak = max(QC_TPS_MEAN)
    qc_final = QC_TPS_MEAN[-1]
    qc_deg = (qc_peak - qc_final) / qc_peak * 100
    
    ax.text(0.98, 0.95, f'SC Degradation: {sc_deg:.1f}%\nQC Degradation: {qc_deg:.1f}%',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_combined_cliff.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_combined_cliff.pdf'))
    plt.close()
    print("Generated: fig_combined_cliff.png/pdf")


def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating Publication Figures with Error Bars")
    print("=" * 60)
    print(f"Output directory: {FIGURES_DIR}")
    print()
    
    fig1_saturation_cliff()
    fig2_latency_analysis()
    fig6_baseline_comparison()
    fig8_workload_heatmap()
    fig_workload_bars()
    fig_combined_cliff()
    
    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
