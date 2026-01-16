"""
Context Switch Overhead Measurement
Measures context switches at different thread counts to quantify oversubscription cost
"""
import os
import sys
import time
import json
import concurrent.futures
from statistics import mean

def get_context_switches():
    """Read context switches from /proc/self/status (Linux) or use psutil (cross-platform)"""
    # Try Linux /proc interface first
    try:
        with open('/proc/self/status', 'r') as f:
            vol = nonvol = 0
            for line in f:
                if line.startswith('voluntary_ctxt_switches'):
                    vol = int(line.split()[1])
                elif line.startswith('nonvoluntary_ctxt_switches'):
                    nonvol = int(line.split()[1])
            return vol, nonvol
    except FileNotFoundError:
        pass
    
    # Fallback to psutil for Windows/macOS
    try:
        import psutil
        proc = psutil.Process()
        ctx = proc.num_ctx_switches()
        return ctx.voluntary, ctx.involuntary
    except ImportError:
        print("Installing psutil for cross-platform context switch measurement...")
        os.system(f"{sys.executable} -m pip install psutil -q")
        import psutil
        proc = psutil.Process()
        ctx = proc.num_ctx_switches()
        return ctx.voluntary, ctx.involuntary

def set_core_affinity(num_cores):
    """Simulate edge device by limiting to N cores (cross-platform)"""
    try:
        if sys.platform == 'win32':
            # Windows: use psutil for CPU affinity
            try:
                import psutil
                p = psutil.Process()
                available_cpus = list(range(psutil.cpu_count()))
                target_cpus = available_cpus[:num_cores]
                p.cpu_affinity(target_cpus)
                return True
            except ImportError:
                return False
        else:
            # Linux/macOS: use os.sched_setaffinity
            cores = list(range(num_cores))
            os.sched_setaffinity(0, cores)
            return True
    except (AttributeError, OSError, Exception):
        return False

def mixed_workload(cpu_ms=10, io_ms=50):
    """Synthetic CPU + I/O workload matching paper methodology"""
    start = time.thread_time()
    result = 0
    target = cpu_ms / 1000
    while (time.thread_time() - start) < target:
        result += 1
    time.sleep(io_ms / 1000)
    return result

def benchmark_with_ctx_switches(thread_count, duration=30, num_cores=4):
    """Measure context switches during workload"""
    set_core_affinity(num_cores)
    
    vol_start, nonvol_start = get_context_switches()
    start_time = time.time()
    
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = []
        while time.time() - start_time < duration:
            f = executor.submit(mixed_workload)
            futures.append(f)
            
            # Prevent excessive task buildup
            if len(futures) > thread_count * 5:
                for fut in futures[:thread_count]:
                    try:
                        fut.result(timeout=5)
                        completed += 1
                    except:
                        pass
                futures = futures[thread_count:]
        
        for f in futures:
            try:
                f.result(timeout=5)
                completed += 1
            except:
                pass
    
    vol_end, nonvol_end = get_context_switches()
    elapsed = time.time() - start_time
    
    vol_switches = vol_end - vol_start
    nonvol_switches = nonvol_end - nonvol_start
    total_switches = vol_switches + nonvol_switches
    
    tps = completed / elapsed if elapsed > 0 else 0
    switches_per_task = total_switches / completed if completed > 0 else 0
    
    return {
        'threads': thread_count,
        'tps': round(tps, 1),
        'completed_tasks': completed,
        'total_ctx_switches': total_switches,
        'ctx_switches_per_task': round(switches_per_task, 2),
        'voluntary': vol_switches,
        'nonvoluntary': nonvol_switches
    }

def main():
    print("=" * 70)
    print("Context Switch Overhead Measurement")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    # Test context switch reading
    try:
        vol, nonvol = get_context_switches()
        print(f"Context switch measurement available: voluntary={vol}, nonvoluntary={nonvol}")
    except Exception as e:
        print(f"Error reading context switches: {e}")
        return
    
    print("\nMeasuring context switches across thread counts...")
    print("-" * 70)
    
    thread_counts = [4, 8, 16, 32, 64, 128, 256, 512]
    results = []
    
    # Duration per test (seconds)
    test_duration = 15
    
    for tc in thread_counts:
        print(f"  {tc:4d} threads...", end=' ', flush=True)
        result = benchmark_with_ctx_switches(tc, duration=test_duration, num_cores=4)
        results.append(result)
        print(f"TPS: {result['tps']:8.1f}, Ctx/task: {result['ctx_switches_per_task']:6.2f}, "
              f"Vol: {result['voluntary']:8d}, NonVol: {result['nonvoluntary']:6d}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    
    # Find optimal (highest TPS in first half)
    optimal_candidates = results[:4]
    optimal = max(optimal_candidates, key=lambda x: x['tps'])
    
    # Find over-provisioned (last config)
    overprovisioned = results[-1]
    
    overhead_ratio = overprovisioned['ctx_switches_per_task'] / optimal['ctx_switches_per_task'] if optimal['ctx_switches_per_task'] > 0 else 0
    
    print(f"Optimal ({optimal['threads']} threads):")
    print(f"  TPS: {optimal['tps']:.1f}")
    print(f"  Context switches per task: {optimal['ctx_switches_per_task']:.2f}")
    
    print(f"\nOver-provisioned ({overprovisioned['threads']} threads):")
    print(f"  TPS: {overprovisioned['tps']:.1f}")
    print(f"  Context switches per task: {overprovisioned['ctx_switches_per_task']:.2f}")
    
    print(f"\nOverhead increase: {overhead_ratio:.1f}x more context switches per task")
    
    # Save results
    output_data = {
        'python_version': sys.version,
        'platform': sys.platform,
        'results': results,
        'analysis': {
            'optimal_threads': optimal['threads'],
            'optimal_ctx_per_task': optimal['ctx_switches_per_task'],
            'overprovisioned_threads': overprovisioned['threads'],
            'overprovisioned_ctx_per_task': overprovisioned['ctx_switches_per_task'],
            'overhead_ratio': round(overhead_ratio, 2)
        }
    }
    
    with open('ctx_switch_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to ctx_switch_results.json")
    
    return results

if __name__ == "__main__":
    main()
