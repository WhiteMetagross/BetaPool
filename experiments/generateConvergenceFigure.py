#!/usr/bin/env python3
"""
Generate the Convergence Proof Visualization figure (fig_convergence_proof.png).

This figure illustrates Theorem 3 from the paper: the blocking characteristic
curve B(N) decreases monotonically, and the system converges to N* where
B(N*) = beta_threshold.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up publication-quality figure settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def blocking_characteristic(N, N_critical=16, beta_max=0.95, beta_min=0.10, steepness=0.02):
    """
    Model the blocking characteristic function B(N).
    
    This function models how the blocking ratio decreases as thread count increases
    due to GIL contention. Uses a sigmoid-like decay to represent the saturation cliff.
    
    Parameters:
    - N: Thread count
    - N_critical: Point where significant degradation begins
    - beta_max: Maximum blocking ratio (I/O-bound regime)
    - beta_min: Minimum blocking ratio (CPU/GIL-saturated regime)
    - steepness: Rate of transition
    """
    return beta_min + (beta_max - beta_min) / (1 + np.exp(steepness * (N - N_critical)))


def main():
    # Thread count range
    N = np.linspace(1, 256, 500)
    
    # Compute blocking characteristic curve
    beta = blocking_characteristic(N, N_critical=64, beta_max=0.92, beta_min=0.15, steepness=0.05)
    
    # Threshold value
    beta_thresh = 0.3
    
    # Find N* (intersection point)
    N_star_idx = np.argmin(np.abs(beta - beta_thresh))
    N_star = N[N_star_idx]
    beta_star = beta[N_star_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Fill safe operating region
    safe_region_mask = N <= N_star
    ax.fill_between(N[safe_region_mask], 0, 1, alpha=0.15, color='green', 
                    label='Safe Operating Region')
    
    # Fill dangerous region (beyond N*)
    danger_region_mask = N >= N_star
    ax.fill_between(N[danger_region_mask], 0, 1, alpha=0.10, color='red',
                    label='Saturation Cliff Region')
    
    # Plot blocking characteristic curve
    ax.plot(N, beta, 'b-', linewidth=2.5, label=r'$\mathcal{B}(N)$ (Blocking Characteristic)')
    
    # Plot threshold line
    ax.axhline(y=beta_thresh, color='darkorange', linestyle='--', linewidth=2,
               label=r'$\beta_{\mathrm{thresh}} = 0.3$')
    
    # Mark the intersection point N*
    ax.plot(N_star, beta_star, 'ko', markersize=10, zorder=5)
    ax.annotate(r'$N^* = %d$' % int(N_star), 
                xy=(N_star, beta_star), 
                xytext=(N_star + 30, beta_star + 0.15),
                fontsize=11,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
    
    # Add vertical line at N*
    ax.axvline(x=N_star, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Add annotations for regions
    ax.text(N_star / 2, 0.08, 'VETO Inactive\n(Scale-up allowed)', 
            ha='center', va='bottom', fontsize=9, style='italic', color='darkgreen')
    ax.text(N_star + (256 - N_star) / 2, 0.04, 'VETO Active\n(Scale-up blocked)', 
            ha='center', va='bottom', fontsize=9, style='italic', color='darkred')
    
    # Add arrow showing convergence direction
    arrow_y = 0.55
    ax.annotate('', xy=(N_star - 5, arrow_y), xytext=(20, arrow_y),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(N_star / 2 - 15, arrow_y + 0.05, r'$N_k \rightarrow N^*$', 
            ha='center', va='bottom', fontsize=10, color='blue')
    
    # Formatting
    ax.set_xlabel('Thread Count ($N$)')
    ax.set_ylabel(r'Blocking Ratio $\mathcal{B}(N)$')
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 32, 64, 96, 128, 160, 192, 224, 256])
    ax.set_yticks([0, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.95)
    
    # Title (optional, can be removed for paper)
    # ax.set_title('Convergence to Equilibrium Point $N^*$')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / 'docs' / 'MandalSchedulingAlgorithm' / 'img'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'fig_convergence_proof.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    # Also save to figures directory
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'fig_convergence_proof.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure also saved to: {figures_dir / 'fig_convergence_proof.png'}")
    
    plt.show()


if __name__ == '__main__':
    main()
