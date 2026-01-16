"""
Memory Pressure Test: ThreadPool vs ProcessPool
Demonstrates memory overhead that makes multiprocessing infeasible on edge devices
"""
import os
import sys
import time
import json
import concurrent.futures
import multiprocessing

def ensure_psutil():
    """Ensure psutil is installed"""
    try:
        import psutil
        return psutil
    except ImportError:
        print("Installing psutil...")
        os.system(f"{sys.executable} -m pip install psutil -q")
        import psutil
        return psutil

def get_memory_mb(psutil_module):
    """Get current process memory in MB"""
    process = psutil_module.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_total_memory_with_children(psutil_module):
    """Get total memory including child processes"""
    parent = psutil_module.Process(os.getpid())
    total_mem = parent.memory_info().rss
    for child in parent.children(recursive=True):
        try:
            total_mem += child.memory_info().rss
        except:
            pass
    return total_mem / 1024 / 1024

def mixed_workload(cpu_ms=10, io_ms=50):
    """Synthetic CPU + I/O workload matching paper methodology"""
    start = time.thread_time()
    result = 0
    target = cpu_ms / 1000
    while (time.thread_time() - start) < target:
        result += 1
    time.sleep(io_ms / 1000)
    return result

def test_threadpool(num_workers, duration=10, psutil_module=None):
    """Test ThreadPoolExecutor memory"""
    mem_before = get_memory_mb(psutil_module)
    
    completed = 0
    start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        while time.time() - start < duration:
            f = executor.submit(mixed_workload)
            futures.append(f)
            
            if len(futures) > num_workers * 5:
                for fut in futures[:num_workers]:
                    try:
                        fut.result(timeout=5)
                        completed += 1
                    except:
                        pass
                futures = futures[num_workers:]
        
        for f in futures:
            try:
                f.result(timeout=5)
                completed += 1
            except:
                pass
    
    elapsed = time.time() - start
    mem_after = get_memory_mb(psutil_module)
    mem_increase = mem_after - mem_before
    
    return {
        'type': 'ThreadPool',
        'workers': num_workers,
        'memory_mb': round(mem_after, 1),
        'memory_increase_mb': round(mem_increase, 1),
        'tps': round(completed / elapsed, 1) if elapsed > 0 else 0,
        'completed': completed
    }

def test_processpool(num_workers, duration=10, psutil_module=None):
    """Test ProcessPoolExecutor memory"""
    mem_before = get_memory_mb(psutil_module)
    
    try:
        completed = 0
        start = time.time()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            while time.time() - start < duration:
                f = executor.submit(mixed_workload)
                futures.append(f)
                
                if len(futures) > num_workers * 3:
                    for fut in futures[:num_workers]:
                        try:
                            fut.result(timeout=10)
                            completed += 1
                        except:
                            pass
                    futures = futures[num_workers:]
            
            # Measure memory while processes are running
            mem_with_children = get_total_memory_with_children(psutil_module)
            
            for f in futures:
                try:
                    f.result(timeout=10)
                    completed += 1
                except:
                    pass
        
        elapsed = time.time() - start
        mem_increase = mem_with_children - mem_before
        
        return {
            'type': 'ProcessPool',
            'workers': num_workers,
            'memory_mb': round(mem_with_children, 1),
            'memory_increase_mb': round(mem_increase, 1),
            'tps': round(completed / elapsed, 1) if elapsed > 0 else 0,
            'completed': completed,
            'per_worker_overhead_mb': round(mem_increase / num_workers, 1) if num_workers > 0 else 0
        }
    except Exception as e:
        return {
            'type': 'ProcessPool',
            'workers': num_workers,
            'memory_mb': 0,
            'memory_increase_mb': 0,
            'tps': 0,
            'completed': 0,
            'error': str(e)
        }

def main():
    psutil = ensure_psutil()
    
    print("=" * 70)
    print("Memory Pressure Test: ThreadPool vs ProcessPool")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    baseline_mem = get_memory_mb(psutil)
    print(f"Baseline memory: {baseline_mem:.1f} MB")
    print()
    
    results = []
    
    # Test ThreadPool
    print("ThreadPool (threads share memory):")
    print("-" * 50)
    for workers in [4, 8, 16, 32, 64]:
        print(f"  {workers:3d} workers...", end=' ', flush=True)
        result = test_threadpool(workers, duration=10, psutil_module=psutil)
        results.append(result)
        print(f"Mem: {result['memory_mb']:6.1f} MB (+{result['memory_increase_mb']:5.1f}), "
              f"TPS: {result['tps']:7.1f}")
    
    print()
    
    # Test ProcessPool
    print("ProcessPool (each process has separate memory):")
    print("-" * 50)
    for workers in [2, 4, 8, 16]:
        print(f"  {workers:3d} workers...", end=' ', flush=True)
        result = test_processpool(workers, duration=10, psutil_module=psutil)
        results.append(result)
        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Mem: {result['memory_mb']:6.1f} MB (+{result['memory_increase_mb']:5.1f}), "
                  f"TPS: {result['tps']:7.1f}, Per-worker: {result['per_worker_overhead_mb']:.1f} MB")
    
    # Analysis
    print("\n" + "=" * 70)
    print("Analysis: Memory Overhead Comparison")
    print("=" * 70)
    
    thread_results = [r for r in results if r['type'] == 'ThreadPool']
    process_results = [r for r in results if r['type'] == 'ProcessPool' and 'error' not in r]
    
    if thread_results and process_results:
        # Compare at similar worker counts
        thread_32 = next((r for r in thread_results if r['workers'] == 32), thread_results[-1])
        process_8 = next((r for r in process_results if r['workers'] == 8), process_results[-1])
        
        print(f"\nThreadPool with {thread_32['workers']} workers:")
        print(f"  Total memory: {thread_32['memory_mb']:.1f} MB")
        print(f"  Memory overhead: {thread_32['memory_increase_mb']:.1f} MB")
        print(f"  Throughput: {thread_32['tps']:.1f} TPS")
        
        print(f"\nProcessPool with {process_8['workers']} workers:")
        print(f"  Total memory: {process_8['memory_mb']:.1f} MB")
        print(f"  Memory overhead: {process_8['memory_increase_mb']:.1f} MB")
        print(f"  Per-worker overhead: {process_8.get('per_worker_overhead_mb', 0):.1f} MB")
        print(f"  Throughput: {process_8['tps']:.1f} TPS")
        
        if thread_32['memory_mb'] > 0 and process_8['memory_mb'] > 0:
            ratio = process_8['memory_mb'] / thread_32['memory_mb']
            print(f"\nMemory ratio (ProcessPool/ThreadPool): {ratio:.1f}x")
        
        print("\nEdge Device Implications (2GB Raspberry Pi 4):")
        print(f"  Available RAM after system: ~1500 MB")
        print(f"  ThreadPool 32 workers: {thread_32['memory_mb']:.0f} MB -> OK")
        if process_8['memory_mb'] > 0:
            remaining = 1500 - process_8['memory_mb']
            print(f"  ProcessPool 8 workers: {process_8['memory_mb']:.0f} MB -> {remaining:.0f} MB remaining for model")
    
    # Save results
    output_data = {
        'python_version': sys.version,
        'platform': sys.platform,
        'baseline_memory_mb': round(baseline_mem, 1),
        'results': results
    }
    
    with open('memory_pressure_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to memory_pressure_results.json")
    
    return results

if __name__ == "__main__":
    # Required for ProcessPoolExecutor on Windows
    multiprocessing.freeze_support()
    main()
