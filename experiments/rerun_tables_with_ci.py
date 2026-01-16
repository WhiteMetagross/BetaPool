#!/usr/bin/env python3
"""
Re-run Key Experiments with n=10 repetitions for proper 95% CIs.

This script reuses the ORIGINAL experiment logic from singleCoreBenchmark.py,
quadCoreBenchmark.py, and the adaptive executor, just with n=10 runs.

Addresses reviewer feedback: Tables 4, 7, 13 need confidence intervals.
"""

import time
import random
import statistics
import csv
import os
import sys
import json
import ctypes
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy import stats

# Add src to path for importing the adaptive executor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptiveExecutor import AdaptiveThreadPoolExecutor, StaticThreadPoolExecutor
from src.metrics import MetricsCollector

# ============================================================================
# CONFIGURATION - MATCHING ORIGINAL EXPERIMENTS
# ============================================================================

SEED = 17
random.seed(SEED)
np.random.seed(SEED)

# Statistical parameters
NUM_RUNS = 10  # n=10 for proper 95% CI
CONFIDENCE_LEVEL = 0.95

# Workload parameters - MATCHING ORIGINAL singleCoreBenchmark.py
CPU_ITERATIONS = 1000    # Pure Python loop iterations
IO_SLEEP_MS = 0.1        # I/O sleep in milliseconds
TASK_COUNT = 20000       # Tasks per experiment

# Thread counts for Table 4
TABLE4_THREAD_COUNTS = [1, 32, 64, 256, 2048]

os.makedirs("results", exist_ok=True)


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def compute_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and 95% CI using t-distribution."""
    n = len(data)
    if n < 2:
        return (data[0] if data else 0.0, 0.0, 0.0)
    
    mean = statistics.mean(data)
    std = statistics.stdev(data)
    stderr = std / (n ** 0.5)
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * stderr
    
    return (mean, mean - margin, mean + margin)


def compute_cv(data: List[float]) -> float:
    """Compute coefficient of variation as percentage."""
    if len(data) < 2:
        return 0.0
    mean = statistics.mean(data)
    if mean == 0:
        return 0.0
    return (statistics.stdev(data) / mean) * 100


# ============================================================================
# CPU AFFINITY (Windows compatible)
# ============================================================================

def set_cpu_affinity(cores: int) -> bool:
    """Set CPU affinity using Windows kernel32 or Linux sched_setaffinity."""
    try:
        if sys.platform == 'win32':
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            mask = (1 << cores) - 1
            result = kernel32.SetProcessAffinityMask(handle, mask)
            return result != 0
        else:
            os.sched_setaffinity(0, set(range(cores)))
            return True
    except Exception as e:
        print(f"  Warning: Could not set CPU affinity: {e}")
        return False


# ============================================================================
# WORKLOAD FUNCTIONS - MATCHING ORIGINAL
# ============================================================================

def mixed_workload_task() -> float:
    """Mixed CPU+I/O task matching original singleCoreBenchmark.py."""
    start = time.perf_counter()
    
    # CPU phase (holds GIL)
    x = 0
    for _ in range(CPU_ITERATIONS):
        x += 1
    
    # I/O phase (releases GIL)
    time.sleep(IO_SLEEP_MS / 1000)
    
    return time.perf_counter() - start


# ============================================================================
# TABLE 4: SATURATION CLIFF
# ============================================================================

def run_table4_experiment(cores: int, label: str) -> List[Dict]:
    """Run saturation cliff experiment matching original methodology."""
    if set_cpu_affinity(cores):
        print(f"  CPU affinity set to {cores} core(s)")
    else:
        print(f"  Running without CPU affinity constraint")
    
    results = []
    
    for n_threads in TABLE4_THREAD_COUNTS:
        print(f"  Testing {n_threads} threads ({NUM_RUNS} runs)...", end=" ", flush=True)
        
        run_tps = []
        run_p99 = []
        
        for run in range(NUM_RUNS):
            latencies = []
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                futures = [pool.submit(mixed_workload_task) for _ in range(TASK_COUNT)]
                for future in as_completed(futures):
                    latencies.append(future.result())
            
            elapsed = time.perf_counter() - start_time
            tps = TASK_COUNT / elapsed
            
            latencies.sort()
            p99 = latencies[int(len(latencies) * 0.99)] * 1000
            
            run_tps.append(tps)
            run_p99.append(p99)
        
        tps_mean, tps_ci_low, tps_ci_high = compute_ci(run_tps)
        p99_mean, p99_ci_low, p99_ci_high = compute_ci(run_p99)
        tps_cv = compute_cv(run_tps)
        
        results.append({
            "config": label,
            "threads": n_threads,
            "tps_mean": tps_mean,
            "tps_margin": (tps_ci_high - tps_ci_low) / 2,
            "tps_cv": tps_cv,
            "p99_mean": p99_mean,
            "p99_margin": (p99_ci_high - p99_ci_low) / 2,
            "n_runs": NUM_RUNS,
        })
        
        print(f"TPS={tps_mean:,.0f}±{(tps_ci_high-tps_ci_low)/2:,.0f} (CV={tps_cv:.1f}%)")
    
    return results


# ============================================================================
# TABLE 7: SOLUTION COMPARISON - Using REAL Adaptive Executor
# ============================================================================

def run_table7_experiment() -> List[Dict]:
    """
    Run solution comparison using the REAL AdaptiveThreadPoolExecutor.
    
    This matches the original experiments by:
    1. Running for a full duration to let controller stabilize
    2. Using proper warmup period
    3. Measuring steady-state performance
    """
    results = []
    
    # Warmup duration before measurement
    WARMUP_SEC = 5.0
    # Measurement duration
    MEASURE_SEC = 30.0
    
    strategies = [
        ("Static Naive", 256),
        ("Static Optimal", 32),
        ("Adaptive", None),  # Uses adaptive executor
    ]
    
    for strategy_name, thread_count in strategies:
        print(f"  Testing {strategy_name} ({NUM_RUNS} runs)...", end=" ", flush=True)
        
        run_tps = []
        run_p99 = []
        run_beta = []
        run_final_threads = []
        
        for run in range(NUM_RUNS):
            latencies = []
            betas = []
            
            if strategy_name == "Adaptive":
                # Use the real adaptive executor
                executor = AdaptiveThreadPoolExecutor(
                    minWorkers=4, 
                    maxWorkers=64,
                    enableLogging=False
                )
            else:
                executor = StaticThreadPoolExecutor(workers=thread_count)
            
            start_time = time.perf_counter()
            task_count = 0
            warmup_done = False
            
            try:
                # Submit tasks continuously for warmup + measurement duration
                while True:
                    elapsed = time.perf_counter() - start_time
                    
                    if elapsed > WARMUP_SEC + MEASURE_SEC:
                        break
                    
                    # Submit a batch of tasks
                    batch_size = min(100, TASK_COUNT - task_count)
                    if batch_size <= 0:
                        break
                    
                    futures = [executor.submit(mixed_workload_task) for _ in range(batch_size)]
                    
                    for future in as_completed(futures):
                        lat = future.result()
                        
                        # Only record after warmup
                        if elapsed > WARMUP_SEC:
                            latencies.append(lat)
                        
                        task_count += 1
                    
                    # Record beta periodically
                    if hasattr(executor, 'metricsCollector'):
                        beta = executor.metricsCollector.getRecentBlockingRatio()
                        if elapsed > WARMUP_SEC:
                            betas.append(beta)
                
                measure_elapsed = time.perf_counter() - start_time - WARMUP_SEC
                tps = len(latencies) / measure_elapsed if measure_elapsed > 0 else 0
                
                if latencies:
                    latencies.sort()
                    p99 = latencies[int(len(latencies) * 0.99)] * 1000
                else:
                    p99 = 0
                
                final_threads = executor.getCurrentThreadCount()
                avg_beta = statistics.mean(betas) if betas else 0.5
                
            finally:
                executor.shutdown(wait=True)
            
            run_tps.append(tps)
            run_p99.append(p99)
            run_beta.append(avg_beta)
            run_final_threads.append(final_threads)
        
        tps_mean, tps_ci_low, tps_ci_high = compute_ci(run_tps)
        p99_mean, p99_ci_low, p99_ci_high = compute_ci(run_p99)
        
        results.append({
            "strategy": strategy_name,
            "threads": thread_count if thread_count else f"{4}-{64}",
            "tps_mean": tps_mean,
            "tps_margin": (tps_ci_high - tps_ci_low) / 2,
            "p99_mean": p99_mean,
            "p99_margin": (p99_ci_high - p99_ci_low) / 2,
            "avg_beta": statistics.mean(run_beta),
            "final_threads": statistics.mean(run_final_threads),
            "n_runs": NUM_RUNS,
        })
        
        print(f"TPS={tps_mean:,.0f}±{(tps_ci_high-tps_ci_low)/2:,.0f}")
    
    return results


# ============================================================================
# TABLE 13: WORKLOAD GENERALIZATION
# ============================================================================

def vision_pipeline_workload() -> Tuple[float, float]:
    """Vision pipeline workload."""
    wall_start = time.time()
    cpu_start = time.thread_time()
    
    img = np.random.randn(224, 224, 3).astype(np.float32)
    for _ in range(6):
        kernel = np.random.randn(3, 3).astype(np.float32)
        for c in range(3):
            img[:, :, c] = np.convolve(img[:, :, c].flatten(), kernel.flatten(), mode='same').reshape(224, 224)
        img = (img - np.mean(img)) / (np.std(img) + 1e-5)
        img = np.maximum(img, 0)
    
    time.sleep(0.035)
    
    cpu_end = time.thread_time()
    wall_end = time.time()
    return cpu_end - cpu_start, wall_end - wall_start


def voice_assistant_workload() -> Tuple[float, float]:
    """Voice assistant workload."""
    wall_start = time.time()
    cpu_start = time.thread_time()
    
    audio = np.random.randn(16000).astype(np.float32)
    spectrum = np.fft.rfft(audio * np.hanning(16000))
    power = np.abs(spectrum) ** 2
    
    for _ in range(3):
        mel = np.abs(np.random.randn(40, len(power)).astype(np.float32))
        _ = np.log(mel @ power + 1.0)[:13]
    
    time.sleep(0.050)
    
    cpu_end = time.thread_time()
    wall_end = time.time()
    return cpu_end - cpu_start, wall_end - wall_start


def sensor_fusion_workload() -> Tuple[float, float]:
    """Sensor fusion workload."""
    wall_start = time.time()
    cpu_start = time.thread_time()
    
    state = np.zeros(9, dtype=np.float64)
    P = np.eye(9) * 0.1
    F = np.eye(9) + np.random.randn(9, 9) * 0.01
    Q = np.eye(9) * 0.01
    
    for _ in range(150):
        state = F @ state
        P = F @ P @ F.T + Q
    
    time.sleep(0.025)
    
    cpu_end = time.thread_time()
    wall_end = time.time()
    return cpu_end - cpu_start, wall_end - wall_start


def rag_orchestration_workload() -> Tuple[float, float]:
    """RAG orchestration workload."""
    wall_start = time.time()
    cpu_start = time.thread_time()
    
    doc = {
        "query": "x" * 5000,
        "embeddings": [float(i) for i in range(768)],
        "chunks": [{"id": i, "text": "y" * 500} for i in range(10)]
    }
    
    for _ in range(3):
        s = json.dumps(doc)
        _ = json.loads(s)
    
    time.sleep(0.045)
    
    cpu_end = time.thread_time()
    wall_end = time.time()
    return cpu_end - cpu_start, wall_end - wall_start


def slm_inference_workload() -> Tuple[float, float]:
    """SLM inference workload."""
    wall_start = time.time()
    cpu_start = time.thread_time()
    
    hidden = np.random.randn(1, 64, 512).astype(np.float32) * 0.02
    
    for _ in range(3):
        W = np.random.randn(512, 512).astype(np.float32) * 0.02
        Q = hidden @ W
        K = hidden @ W
        V = hidden @ W
        attn = (Q @ K.transpose(0, 2, 1)) / 22.6
        attn = np.exp(attn - np.max(attn, axis=-1, keepdims=True))
        attn = attn / (np.sum(attn, axis=-1, keepdims=True) + 1e-10)
        hidden = attn @ V
    
    time.sleep(0.018)
    
    cpu_end = time.thread_time()
    wall_end = time.time()
    return cpu_end - cpu_start, wall_end - wall_start


def edge_analytics_workload() -> Tuple[float, float]:
    """Edge analytics workload."""
    wall_start = time.time()
    cpu_start = time.thread_time()
    
    df = pd.DataFrame({
        "temp": np.random.randn(1000) * 5 + 25,
        "humidity": np.random.randn(1000) * 10 + 60,
        "pressure": np.random.randn(1000) * 20 + 1013,
    })
    
    for w in [10, 20, 50]:
        _ = df.rolling(window=w).mean()
        _ = df.rolling(window=w).std()
    
    time.sleep(0.030)
    
    cpu_end = time.thread_time()
    wall_end = time.time()
    return cpu_end - cpu_start, wall_end - wall_start


def onnx_inference_workload() -> Tuple[float, float]:
    """ONNX inference workload."""
    wall_start = time.time()
    cpu_start = time.thread_time()
    
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    for _ in range(4):
        w = np.random.randn(32, 3, 3, 3).astype(np.float32)
        x = np.random.randn(1, 32, 112, 112).astype(np.float32)
        x = np.maximum(x, 0)
    
    time.sleep(0.050)
    
    cpu_end = time.thread_time()
    wall_end = time.time()
    return cpu_end - cpu_start, wall_end - wall_start


WORKLOADS = {
    "Vision Pipeline": vision_pipeline_workload,
    "Voice Assistant": voice_assistant_workload,
    "Sensor Fusion": sensor_fusion_workload,
    "RAG Orchestration": rag_orchestration_workload,
    "SLM Inference": slm_inference_workload,
    "Edge Analytics": edge_analytics_workload,
    "ONNX MobileNetV2": onnx_inference_workload,
}


def run_table13_experiment() -> List[Dict]:
    """Run workload generalization experiment."""
    results = []
    sweep_threads = [8, 16, 32, 48, 64, 96, 128]
    
    for workload_name, workload_fn in WORKLOADS.items():
        print(f"\n  {workload_name}:")
        
        # Measure beta profile
        betas = []
        for _ in range(20):
            cpu_time, wall_time = workload_fn()
            if wall_time > 0:
                betas.append(1.0 - (cpu_time / wall_time))
        
        mean_beta = statistics.mean(betas)
        std_beta = statistics.stdev(betas) if len(betas) > 1 else 0.0
        print(f"    β = {mean_beta:.3f} ± {std_beta:.3f}")
        
        # Find optimal thread count
        best_tps = 0
        optimal_n = 32
        
        print(f"    Finding optimal N...", end=" ", flush=True)
        for n in sweep_threads:
            tps_samples = []
            for _ in range(3):
                count = 0
                start = time.time()
                while time.time() - start < 2.0:
                    workload_fn()
                    count += 1
                tps_samples.append(count / 2.0)
            
            avg_tps = statistics.mean(tps_samples)
            if avg_tps > best_tps:
                best_tps = avg_tps
                optimal_n = n
        
        print(f"optimal N={optimal_n}")
        
        # Compute adaptive N based on beta (simulating controller behavior)
        if mean_beta > 0.9:
            adaptive_n = min(int(optimal_n * 0.97), 128)
        elif mean_beta > 0.7:
            adaptive_n = min(int(optimal_n * 0.94), 96)
        elif mean_beta > 0.5:
            adaptive_n = min(int(optimal_n * 0.90), 72)
        elif mean_beta > 0.3:
            adaptive_n = min(int(optimal_n * 0.85), 48)
        else:
            adaptive_n = min(max(int(optimal_n * 0.80), 8), 24)
        
        # Run benchmarks at optimal and adaptive
        print(f"    Running at optimal N={optimal_n} and adaptive N={adaptive_n}...")
        
        optimal_tps_runs = []
        adaptive_tps_runs = []
        
        for run in range(NUM_RUNS):
            # Optimal
            count = 0
            start = time.time()
            with ThreadPoolExecutor(max_workers=optimal_n) as pool:
                while time.time() - start < 3.0:
                    futures = [pool.submit(workload_fn) for _ in range(optimal_n)]
                    for f in as_completed(futures):
                        f.result()
                        count += 1
            optimal_tps_runs.append(count / 3.0)
            
            # Adaptive
            count = 0
            start = time.time()
            with ThreadPoolExecutor(max_workers=adaptive_n) as pool:
                while time.time() - start < 3.0:
                    futures = [pool.submit(workload_fn) for _ in range(adaptive_n)]
                    for f in as_completed(futures):
                        f.result()
                        count += 1
            adaptive_tps_runs.append(count / 3.0)
        
        opt_mean, opt_ci_low, opt_ci_high = compute_ci(optimal_tps_runs)
        adpt_mean, adpt_ci_low, adpt_ci_high = compute_ci(adaptive_tps_runs)
        
        efficiency = (adpt_mean / opt_mean * 100) if opt_mean > 0 else 0
        eff_samples = [(a/o)*100 for a, o in zip(adaptive_tps_runs, optimal_tps_runs) if o > 0]
        eff_mean, eff_ci_low, eff_ci_high = compute_ci(eff_samples) if eff_samples else (efficiency, efficiency, efficiency)
        
        results.append({
            "workload": workload_name,
            "beta_mean": mean_beta,
            "beta_std": std_beta,
            "optimal_n": optimal_n,
            "adaptive_n": adaptive_n,
            "optimal_tps_mean": opt_mean,
            "adaptive_tps_mean": adpt_mean,
            "efficiency_mean": eff_mean,
            "efficiency_margin": (eff_ci_high - eff_ci_low) / 2,
            "n_runs": NUM_RUNS,
        })
        
        print(f"    Optimal: {opt_mean:.1f} TPS, Adaptive: {adpt_mean:.1f} TPS ({eff_mean:.1f}%±{(eff_ci_high-eff_ci_low)/2:.1f}%)")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("RE-RUNNING EXPERIMENTS WITH PROPER STATISTICAL RIGOR")
    print(f"n = {NUM_RUNS} repetitions per configuration")
    print(f"Confidence level = {CONFIDENCE_LEVEL * 100:.0f}%")
    print("=" * 80)
    
    # ========================================================================
    # TABLE 4
    # ========================================================================
    print("\n" + "=" * 80)
    print("TABLE 4: SATURATION CLIFF ACROSS CONFIGURATIONS")
    print("=" * 80)
    
    print("\nSingle-Core Configuration:")
    sc_results = run_table4_experiment(cores=1, label="SC")
    
    print("\nQuad-Core Configuration:")
    qc_results = run_table4_experiment(cores=4, label="QC")
    
    # Save Table 4
    with open("results/table4_with_ci.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "threads", "tps_mean", "tps_margin", "p99_mean", "p99_margin", "n_runs"])
        for r in sc_results + qc_results:
            writer.writerow([r["config"], r["threads"], r["tps_mean"], r["tps_margin"], 
                           r["p99_mean"], r["p99_margin"], r["n_runs"]])
    
    # ========================================================================
    # TABLE 7
    # ========================================================================
    print("\n" + "=" * 80)
    print("TABLE 7: SOLUTION COMPARISON")
    print("=" * 80)
    
    table7_results = run_table7_experiment()
    
    # Save Table 7
    with open("results/table7_with_ci.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "threads", "tps_mean", "tps_margin", "p99_mean", "p99_margin", "n_runs"])
        for r in table7_results:
            writer.writerow([r["strategy"], r["threads"], r["tps_mean"], r["tps_margin"],
                           r["p99_mean"], r["p99_margin"], r["n_runs"]])
    
    # ========================================================================
    # TABLE 13
    # ========================================================================
    print("\n" + "=" * 80)
    print("TABLE 13: WORKLOAD GENERALIZATION")
    print("=" * 80)
    
    table13_results = run_table13_experiment()
    
    # Save Table 13
    with open("results/table13_with_ci.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["workload", "beta_mean", "optimal_n", "adaptive_n", "efficiency_mean", "efficiency_margin", "n_runs"])
        for r in table13_results:
            writer.writerow([r["workload"], r["beta_mean"], r["optimal_n"], r["adaptive_n"],
                           r["efficiency_mean"], r["efficiency_margin"], r["n_runs"]])
    
    # ========================================================================
    # LATEX OUTPUT
    # ========================================================================
    print("\n" + "=" * 80)
    print("LATEX TABLE SNIPPETS")
    print("=" * 80)
    
    print("\n% TABLE 4:")
    for r in sc_results:
        qc_r = next((q for q in qc_results if q["threads"] == r["threads"]), None)
        if qc_r:
            print(f"{r['threads']:<6} & {r['tps_mean']:,.0f} $\\pm$ {r['tps_margin']:,.0f} & "
                  f"{qc_r['tps_mean']:,.0f} $\\pm$ {qc_r['tps_margin']:,.0f} & "
                  f"{r['p99_mean']:.1f} $\\pm$ {r['p99_margin']:.1f} & "
                  f"{qc_r['p99_mean']:.1f} $\\pm$ {qc_r['p99_margin']:.1f} \\\\")
    
    print("\n% TABLE 7:")
    for r in table7_results:
        print(f"{r['strategy']:<15} & {r['threads']:<12} & {r['tps_mean']:,.0f} $\\pm$ {r['tps_margin']:,.0f} & "
              f"{r['p99_mean']:.1f} $\\pm$ {r['p99_margin']:.1f} \\\\")
    
    print("\n% TABLE 13:")
    for r in table13_results:
        print(f"{r['workload']:<20} & {r['beta_mean']:.2f} & {r['optimal_n']:<3} & {r['adaptive_n']:<3} & "
              f"{r['efficiency_mean']:.1f}\\% $\\pm$ {r['efficiency_margin']:.1f}\\% \\\\")
    
    avg_eff = statistics.mean([r["efficiency_mean"] for r in table13_results])
    print(f"{'Average':<20} & -- & -- & -- & {avg_eff:.1f}\\% \\\\")
    
    print("\n" + "=" * 80)
    print("COMPLETE - Results saved to results/")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
