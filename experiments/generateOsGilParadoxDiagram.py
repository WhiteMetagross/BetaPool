"""
Generate the OS-GIL Paradox conceptual diagram for the research paper.

This diagram illustrates why adding more cores doesn't solve GIL contention:
- Panel A: OS scheduler distributes threads across cores
- Panel B: GIL mutex blocks all but one thread
- Panel C: CPU timeline showing context switch overhead vs useful work
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
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
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Colors
COLOR_CORE = "#4a90d9"          # Blue for CPU cores
COLOR_THREAD_ACTIVE = "#2ca02c"  # Green for active/useful work
COLOR_THREAD_BLOCKED = "#d62728" # Red for blocked/overhead
COLOR_THREAD_WAITING = "#ff7f0e" # Orange for waiting
COLOR_GIL = "#7f7f7f"           # Gray for GIL
COLOR_BG = "#f5f5f5"            # Light background

def create_os_gil_paradox_diagram():
    """Create the three-panel OS-GIL Paradox diagram."""
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    
    # =========================================================================
    # Panel A: OS Run Queue - 4 threads distributed across 4 cores
    # =========================================================================
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(A) OS View: 4 Runnable Threads', fontsize=11, fontweight='bold')
    
    # Draw 4 CPU cores as boxes
    core_positions = [(1.5, 7), (5.5, 7), (1.5, 3), (5.5, 3)]
    core_labels = ['Core 0', 'Core 1', 'Core 2', 'Core 3']
    thread_labels = ['T1', 'T2', 'T3', 'T4']
    
    for i, (x, y) in enumerate(core_positions):
        # CPU Core box
        core_box = FancyBboxPatch((x, y), 3, 2, boxstyle="round,pad=0.05",
                                   facecolor=COLOR_CORE, edgecolor='black', linewidth=1.5)
        ax1.add_patch(core_box)
        ax1.text(x + 1.5, y + 1.5, core_labels[i], ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
        
        # Thread circle on top of core
        thread_circle = Circle((x + 1.5, y + 0.5), 0.4, facecolor=COLOR_THREAD_ACTIVE,
                               edgecolor='black', linewidth=1)
        ax1.add_patch(thread_circle)
        ax1.text(x + 1.5, y + 0.5, thread_labels[i], ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')
    
    # Add "OS Scheduler" label
    ax1.text(5, 0.5, 'OS Scheduler: "All cores utilized!"', ha='center', va='center',
            fontsize=9, style='italic', color='#333333')
    
    # =========================================================================
    # Panel B: GIL Mutex - Only 1 thread can execute
    # =========================================================================
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('(B) Python View: GIL Serialization', fontsize=11, fontweight='bold')
    
    # Draw GIL as a lock/gate
    gil_box = FancyBboxPatch((3, 4), 4, 3, boxstyle="round,pad=0.1",
                              facecolor=COLOR_GIL, edgecolor='black', linewidth=2)
    ax2.add_patch(gil_box)
    ax2.text(5, 5.5, 'GIL', ha='center', va='center',
            fontsize=14, color='white', fontweight='bold')
    ax2.text(5, 4.7, '(Mutex)', ha='center', va='center',
            fontsize=9, color='white')
    
    # One thread passing through (green - active)
    active_thread = Circle((5, 2), 0.5, facecolor=COLOR_THREAD_ACTIVE,
                           edgecolor='black', linewidth=1.5)
    ax2.add_patch(active_thread)
    ax2.text(5, 2, 'T1', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')
    ax2.annotate('', xy=(5, 3.8), xytext=(5, 2.6),
                arrowprops=dict(arrowstyle='->', color=COLOR_THREAD_ACTIVE, lw=2))
    ax2.text(5, 1.2, 'Executing', ha='center', va='center',
            fontsize=8, color=COLOR_THREAD_ACTIVE, fontweight='bold')
    
    # Three threads blocked (red - waiting)
    blocked_positions = [(1.5, 8), (5, 8.5), (8.5, 8)]
    blocked_labels = ['T2', 'T3', 'T4']
    
    for i, (x, y) in enumerate(blocked_positions):
        blocked_thread = Circle((x, y), 0.4, facecolor=COLOR_THREAD_BLOCKED,
                                edgecolor='black', linewidth=1)
        ax2.add_patch(blocked_thread)
        ax2.text(x, y, blocked_labels[i], ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')
        # Arrow pointing down to GIL (blocked)
        ax2.annotate('', xy=(5, 7.2), xytext=(x, y - 0.5),
                    arrowprops=dict(arrowstyle='->', color=COLOR_THREAD_BLOCKED, 
                                   lw=1.5, linestyle='--'))
    
    ax2.text(5, 9.3, 'Blocked on GIL', ha='center', va='center',
            fontsize=9, color=COLOR_THREAD_BLOCKED, fontweight='bold')
    
    # =========================================================================
    # Panel C: CPU Timeline - Context switch overhead
    # =========================================================================
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('(C) Timeline: Wasted Cycles', fontsize=11, fontweight='bold')
    
    # Draw timeline bars for each core
    core_y_positions = [8, 6, 4, 2]
    core_labels = ['Core 0', 'Core 1', 'Core 2', 'Core 3']
    
    # Time segments: each tuple is (start, width, color, label)
    # Showing rapid context switching with minimal useful work
    timeline_width = 8
    start_x = 1
    
    for i, y in enumerate(core_y_positions):
        # Core label
        ax3.text(0.5, y + 0.4, core_labels[i], ha='right', va='center', fontsize=8)
        
        # Background bar
        bg_bar = Rectangle((start_x, y), timeline_width, 0.8,
                           facecolor='#e0e0e0', edgecolor='black', linewidth=0.5)
        ax3.add_patch(bg_bar)
        
        # Create pattern of work vs overhead
        # Core 0 gets most useful work, others mostly overhead
        if i == 0:  # Core 0 - has GIL most often
            segments = [
                (0, 0.8, COLOR_THREAD_ACTIVE),   # Work
                (0.8, 0.3, COLOR_THREAD_BLOCKED), # CS
                (1.1, 0.9, COLOR_THREAD_ACTIVE),  # Work
                (2.0, 0.2, COLOR_THREAD_BLOCKED), # CS
                (2.2, 1.0, COLOR_THREAD_ACTIVE),  # Work
                (3.2, 0.3, COLOR_THREAD_BLOCKED), # CS
                (3.5, 0.7, COLOR_THREAD_ACTIVE),  # Work
                (4.2, 0.2, COLOR_THREAD_BLOCKED), # CS
                (4.4, 0.9, COLOR_THREAD_ACTIVE),  # Work
                (5.3, 0.3, COLOR_THREAD_BLOCKED), # CS
                (5.6, 0.8, COLOR_THREAD_ACTIVE),  # Work
                (6.4, 0.2, COLOR_THREAD_BLOCKED), # CS
                (6.6, 0.7, COLOR_THREAD_ACTIVE),  # Work
                (7.3, 0.7, COLOR_THREAD_BLOCKED), # CS
            ]
        else:  # Other cores - mostly overhead (fighting for GIL)
            np.random.seed(42 + i)
            segments = []
            pos = 0
            while pos < timeline_width:
                # Mostly red (blocked/CS), occasional tiny green
                if np.random.random() < 0.15:  # 15% chance of brief work
                    segments.append((pos, 0.15, COLOR_THREAD_ACTIVE))
                    pos += 0.15
                else:
                    cs_len = np.random.uniform(0.3, 0.6)
                    segments.append((pos, cs_len, COLOR_THREAD_BLOCKED))
                    pos += cs_len
                # Small gap
                gap = np.random.uniform(0.1, 0.3)
                segments.append((pos, gap, COLOR_THREAD_WAITING))
                pos += gap
    
        # Draw segments
        for seg_start, seg_width, seg_color in segments:
            if seg_start + seg_width <= timeline_width:
                seg_rect = Rectangle((start_x + seg_start, y), seg_width, 0.8,
                                     facecolor=seg_color, edgecolor=None)
                ax3.add_patch(seg_rect)
    
    # Time arrow
    ax3.annotate('', xy=(9.5, 0.8), xytext=(1, 0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax3.text(5, 0.3, 'Time', ha='center', va='center', fontsize=9)
    
    # Legend
    legend_y = 9.5
    # Green = Useful work
    green_patch = Rectangle((1.5, legend_y - 0.2), 0.8, 0.4,
                            facecolor=COLOR_THREAD_ACTIVE, edgecolor='black', linewidth=0.5)
    ax3.add_patch(green_patch)
    ax3.text(2.5, legend_y, 'Useful Work', ha='left', va='center', fontsize=8)
    
    # Red = Context Switch / GIL Wait
    red_patch = Rectangle((5.5, legend_y - 0.2), 0.8, 0.4,
                          facecolor=COLOR_THREAD_BLOCKED, edgecolor='black', linewidth=0.5)
    ax3.add_patch(red_patch)
    ax3.text(6.5, legend_y, 'CS + GIL Wait', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = '../figures/fig_os_gil_paradox.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved diagram to {output_path}")
    
    # Also save as PDF for LaTeX
    pdf_path = '../figures/fig_os_gil_paradox.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to {pdf_path}")
    
    plt.close()

if __name__ == "__main__":
    create_os_gil_paradox_diagram()
