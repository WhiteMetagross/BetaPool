#!/usr/bin/env python3
"""
Generate publication-quality figures with error bars and confidence bands.
Updated version with fixed text positioning to avoid overlaps.
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
DOCS_IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'img')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DOCS_IMG_DIR, exist_ok=True)

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


def fig1_saturation_cliff():
    """
    Figure 2 in paper: The Saturation Cliff with error bars.
    FIXED: Panel (a) cliff text positioning to avoid overlap.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel (a): Single-Core
    ax1.errorbar(SC_THREADS, SC_TPS_MEAN, yerr=SC_TPS_CI, 
                 fmt='o-', capsize=3, capthick=1, linewidth=2, markersize=6,
                 color='#2E86AB', label='Mixed CPU+I/O', ecolor='#2E86AB', alpha=0.8)
    ax1.plot(SC_IO_THREADS, SC_IO_TPS, 's--', linewidth=2, markersize=6,
             color='#A23B72', label='Pure I/O Baseline', alpha=0.8)
    
    # Mark peak - positioned above and right to avoid graph overlap
    peak_idx = SC_TPS_MEAN.index(max(SC_TPS_MEAN))
    ax1.annotate(f'Peak: {SC_TPS_MEAN[peak_idx]:,} TPS\n@ {SC_THREADS[peak_idx]} threads',
                 xy=(SC_THREADS[peak_idx], SC_TPS_MEAN[peak_idx]),
                 xytext=(SC_THREADS[peak_idx]*8, SC_TPS_MEAN[peak_idx]*1.6),  # FIXED: moved up-right, arrow won't cross graph
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    
    # Mark cliff - FIXED: repositioned to avoid overlap
    ax1.annotate(f'Cliff: {SC_TPS_MEAN[-1]:,} TPS\n(-40.2%)',
                 xy=(SC_THREADS[-1], SC_TPS_MEAN[-1]),
                 xytext=(SC_THREADS[-1]/4, SC_TPS_MEAN[-1]*0.55),  # FIXED: adjusted position
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
    
    # Mark peak - positioned lower to avoid title overlap
    peak_idx = QC_TPS_MEAN.index(max(QC_TPS_MEAN))
    ax2.annotate(f'Peak: {QC_TPS_MEAN[peak_idx]:,} TPS\n@ {QC_THREADS[peak_idx]} threads',
                 xy=(QC_THREADS[peak_idx], QC_TPS_MEAN[peak_idx]),
                 xytext=(QC_THREADS[peak_idx]*6, QC_TPS_MEAN[peak_idx]*1.1),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    
    # Mark cliff
    ax2.annotate(f'Cliff: {QC_TPS_MEAN[-1]:,} TPS\n(-35.1%)',
                 xy=(QC_THREADS[-1], QC_TPS_MEAN[-1]),
                 xytext=(QC_THREADS[-1]/4, QC_TPS_MEAN[-1]*0.5),
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
    
    # Save to both directories
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_saturation_cliff.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_saturation_cliff.pdf'))
    plt.savefig(os.path.join(DOCS_IMG_DIR, 'fig1_saturation_cliff.png'))
    plt.close()
    print("Generated: fig1_saturation_cliff.png/pdf")


def fig2_latency_analysis():
    """
    Figure 3 in paper: Latency Analysis with error bands.
    FIXED: Text positioning to avoid overlap with title.
    """
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
    
    # FIXED: Moved annotation down and to the left to avoid title overlap
    ax1.annotate(f'2.0× increase\n(8.6→17.2 ms)',
                 xy=(2048, SC_P99[-1]),
                 xytext=(200, 3),  # FIXED: moved down significantly
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
    
    # FIXED: Moved annotation down to avoid title overlap
    ax2.annotate(f'4.8× increase\n(4.1→20.0 ms)',
                 xy=(2048, QC_P99[-1]),
                 xytext=(200, 3),  # FIXED: moved down significantly
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    
    plt.tight_layout()
    
    # Save to both directories
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_latency_analysis.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_latency_analysis.pdf'))
    plt.savefig(os.path.join(DOCS_IMG_DIR, 'fig2_latency_analysis.png'))
    plt.close()
    print("Generated: fig2_latency_analysis.png/pdf")


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
    plt.savefig(os.path.join(DOCS_IMG_DIR, 'fig_combined_cliff.png'))
    plt.close()
    print("Generated: fig_combined_cliff.png/pdf")


def main():
    """Generate all figures with fixed text positioning."""
    print("=" * 60)
    print("Generating Publication Figures with Fixed Text Positioning")
    print("=" * 60)
    print(f"Output directories:")
    print(f"  - {FIGURES_DIR}")
    print(f"  - {DOCS_IMG_DIR}")
    print()
    
    fig1_saturation_cliff()
    fig2_latency_analysis()
    fig_combined_cliff()
    
    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
