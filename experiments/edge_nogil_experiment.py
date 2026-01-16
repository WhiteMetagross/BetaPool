"""
Edge Device GIL vs Free-Threading Experiment
Tests saturation cliff on simulated single-core and quad-core edge configurations
"""
import os
import sys
import time
import json
import concurrent.futures
from statistics import mean, stdev

# Install psutil if not available (needed for cross-platform CPU affinity)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def set_core_affinity(num_cores):
    """Simulate edge device by limiting to N cores (cross-platform)"""
    try:
        cores = list(range(num_cores))
        
        if sys.platform == 'win32':
            # Windows: use psutil for CPU affinity
            if PSUTIL_AVAILABLE:
                p = psutil.Process()
                # Get available CPUs
                available_cpus = list(range(psutil.cpu_count()))
                target_cpus = available_cpus[:num_cores]
                p.cpu_affinity(target_cpus)
                return True
            else:
                print(f"  [Info] Install psutil for CPU affinity on Windows: pip install psutil")
                return False
        else:
            # Linux/macOS: use os.sched_setaffinity
            os.sched_setaffinity(0, cores)
            return True
    except (AttributeError, OSError, Exception) as e:
        print(f"  [Info] CPU affinity not set: {type(e).__name__}")
        return False

def mixed_workload(cpu_ms=10, io_ms=50):
    """Synthetic CPU + I/O workload matching paper methodology"""
    # CPU phase (holds GIL)
    start = time.thread_time()
    result = 0
    target = cpu_ms / 1000
    while (time.thread_time() - start) < target:
        result += 1
    
    # I/O phase (releases GIL)
    time.sleep(io_ms / 1000)
    return result

def benchmark(num_cores, thread_count, duration=30):
    """Benchmark at given core/thread config"""
    affinity_set = set_core_affinity(num_cores)
    
    completed = []
    start_time = time.time()
    task_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = []
        while time.time() - start_time < duration:
            submit_time = time.time()
            f = executor.submit(mixed_workload)
            futures.append((f, submit_time))
            task_count += 1
            
            # Prevent excessive task submission
            if len(futures) > thread_count * 10:
                # Wait for some to complete
                done_futures = []
                for future, st in futures[:thread_count]:
                    try:
                        future.result(timeout=5)
                        latency_ms = (time.time() - st) * 1000
                        completed.append(latency_ms)
                    except:
                        pass
                futures = futures[thread_count:]
        
        # Collect remaining
        for future, submit_time in futures:
            try:
                future.result(timeout=5)
                latency_ms = (time.time() - submit_time) * 1000
                completed.append(latency_ms)
            except:
                pass
    
    elapsed = time.time() - start_time
    tps = len(completed) / elapsed if elapsed > 0 else 0
    
    if completed:
        sorted_latencies = sorted(completed)
        p99_idx = int(len(sorted_latencies) * 0.99)
        p99 = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else sorted_latencies[-1]
        avg_latency = mean(completed)
    else:
        p99 = 0
        avg_latency = 0
    
    return {
        'cores': num_cores,
        'threads': thread_count,
        'tps': round(tps, 1),
        'p99_ms': round(p99, 2),
        'avg_latency_ms': round(avg_latency, 2),
        'completed_tasks': len(completed),
        'affinity_set': affinity_set
    }

def get_gil_status():
    """Check if GIL is enabled (Python 3.13+ only)"""
    try:
        return sys._is_gil_enabled()
    except AttributeError:
        return True  # Older Python versions always have GIL

def main():
    print("=" * 70)
    print("Edge Device GIL vs Free-Threading Experiment")
    print("=" * 70)
    print(f"Python: {sys.version}")
    
    gil_enabled = get_gil_status()
    print(f"GIL enabled: {gil_enabled}")
    print(f"Platform: {sys.platform}")
    print()
    
    # Configuration matching paper methodology
    configs = [
        # Single-core edge (Raspberry Pi Zero)
        (1, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
        # Quad-core edge (Raspberry Pi 4)
        (4, [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ]
    
    # Duration per test (seconds) - shorter for faster iteration
    test_duration = 15
    
    all_results = []
    
    for num_cores, thread_counts in configs:
        print(f"\n{'=' * 70}")
        print(f"Testing {num_cores}-core configuration (simulated edge device)")
        print('=' * 70)
        
        core_results = []
        for tc in thread_counts:
            print(f"  {tc:4d} threads...", end=' ', flush=True)
            result = benchmark(num_cores, tc, duration=test_duration)
            all_results.append(result)
            core_results.append(result)
            print(f"TPS: {result['tps']:8.1f}, P99: {result['p99_ms']:8.2f}ms")
        
        # Find optimal for this core count
        optimal = max(core_results, key=lambda x: x['tps'])
        worst = min(core_results[-3:], key=lambda x: x['tps'])  # Last 3 configs
        
        degradation = ((optimal['tps'] - worst['tps']) / optimal['tps']) * 100 if optimal['tps'] > 0 else 0
        
        print(f"\n  Summary for {num_cores}-core:")
        print(f"    Optimal: {optimal['threads']} threads -> {optimal['tps']:.1f} TPS")
        print(f"    Worst (high threads): {worst['threads']} threads -> {worst['tps']:.1f} TPS")
        print(f"    Degradation: {degradation:.1f}%")
    
    # Save results
    gil_status = "nogil" if not gil_enabled else "gil"
    py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    filename = f"edge_results_{gil_status}_py{py_version}.json"
    
    output_data = {
        'python_version': sys.version,
        'gil_enabled': gil_enabled,
        'platform': sys.platform,
        'results': all_results
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to {filename}")
    print("=" * 70)
    
    return all_results

if __name__ == "__main__":
    main()
