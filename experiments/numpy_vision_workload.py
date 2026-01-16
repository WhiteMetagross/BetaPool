"""
NumPy based ML Inference Workload Benchmark
Simulates MobileNetV2 style inference using NumPy convolutions
This provides a representative ML workload without requiring TensorFlow
"""
import os
import sys
import time
import json
import concurrent.futures
from statistics import mean
import numpy as np
import threading

def numpy_convolution_inference(input_size=224, channels=3, num_filters=32, kernel_size=3):
    """
    Simulate CNN inference using NumPy convolutions
    This models MobileNetV2 style computation patterns
    """
    # Generate random input image
    img = np.random.randn(input_size, input_size, channels).astype(np.float32)
    
    # Generate random convolution kernel
    kernel = np.random.randn(kernel_size, kernel_size, channels, num_filters).astype(np.float32)
    
    # Perform convolution (simplified, not optimized)
    output_size = input_size - kernel_size + 1
    output = np.zeros((output_size, output_size, num_filters), dtype=np.float32)
    
    for f in range(num_filters):
        for i in range(output_size):
            for j in range(output_size):
                patch = img[i:i+kernel_size, j:j+kernel_size, :]
                output[i, j, f] = np.sum(patch * kernel[:, :, :, f])
    
    # ReLU activation
    output = np.maximum(output, 0)
    
    # Global average pooling
    features = np.mean(output, axis=(0, 1))
    
    # Simulated fully connected layer
    weights = np.random.randn(num_filters, 1000).astype(np.float32)
    logits = np.dot(features, weights)
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    return int(np.argmax(probs))

def vision_pipeline_workload(io_latency_ms=50):
    """
    Complete vision pipeline workload:
    1. CPU: Image preprocessing and inference
    2. I/O: Simulated network call to send results
    """
    # Inference (CPU bound, holds GIL via NumPy)
    result = numpy_convolution_inference(
        input_size=56,  # Reduced for faster execution
        channels=3,
        num_filters=16,
        kernel_size=3
    )
    
    # Simulated I/O (releases GIL)
    time.sleep(io_latency_ms / 1000)
    
    return result

def benchmark_vision_pipeline(thread_count, duration=60):
    """Benchmark vision pipeline at given thread count"""
    completed = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = []
        while time.time() - start_time < duration:
            submit_time = time.time()
            f = executor.submit(vision_pipeline_workload)
            futures.append((f, submit_time))
            
            # Prevent excessive task buildup
            if len(futures) > thread_count * 5:
                for future, st in futures[:thread_count]:
                    try:
                        future.result(timeout=10)
                        latency = (time.time() - st) * 1000
                        completed.append(latency)
                    except:
                        pass
                futures = futures[thread_count:]
        
        # Collect remaining
        for future, submit_time in futures:
            try:
                future.result(timeout=10)
                latency = (time.time() - submit_time) * 1000
                completed.append(latency)
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
        'threads': thread_count,
        'tps': round(tps, 2),
        'p99_ms': round(p99, 2),
        'avg_latency_ms': round(avg_latency, 2),
        'completed_tasks': len(completed)
    }

def compute_blocking_ratio(thread_count, duration=30):
    """Compute blocking ratio for vision pipeline workload"""
    beta_values = []
    
    def worker_with_timing():
        wall_start = time.time()
        cpu_start = time.thread_time()
        
        vision_pipeline_workload()
        
        cpu_end = time.thread_time()
        wall_end = time.time()
        
        cpu_time = cpu_end - cpu_start
        wall_time = wall_end - wall_start
        
        if wall_time > 0:
            beta = 1 - (cpu_time / wall_time)
            beta_values.append(beta)
        
        return True
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = []
        while time.time() - start_time < duration:
            f = executor.submit(worker_with_timing)
            futures.append(f)
            
            if len(futures) > thread_count * 3:
                for fut in futures[:thread_count]:
                    try:
                        fut.result(timeout=10)
                    except:
                        pass
                futures = futures[thread_count:]
        
        for f in futures:
            try:
                f.result(timeout=10)
            except:
                pass
    
    avg_beta = mean(beta_values) if beta_values else 0
    return round(avg_beta, 3)

def main():
    print("=" * 70)
    print("NumPy Vision Pipeline ML Workload Benchmark")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"NumPy version: {np.__version__}")
    print()
    
    print("Workload: NumPy convolution simulating MobileNetV2 inference + 50ms I/O")
    print("-" * 70)
    
    thread_counts = [1, 4, 8, 16, 32, 64, 128, 256]
    results = []
    
    for tc in thread_counts:
        print(f"  {tc:4d} threads...", end=' ', flush=True)
        result = benchmark_vision_pipeline(tc, duration=60)
        results.append(result)
        print(f"TPS: {result['tps']:8.2f}, P99: {result['p99_ms']:8.2f}ms")
    
    # Find optimal
    optimal = max(results, key=lambda x: x['tps'])
    worst = min(results[-3:], key=lambda x: x['tps'])
    
    degradation = ((optimal['tps'] - worst['tps']) / optimal['tps']) * 100 if optimal['tps'] > 0 else 0
    
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    print(f"Optimal: {optimal['threads']} threads -> {optimal['tps']:.2f} TPS")
    print(f"Worst (high threads): {worst['threads']} threads -> {worst['tps']:.2f} TPS")
    print(f"Degradation at high thread count: {degradation:.1f}%")
    
    # Compute blocking ratio at optimal thread count
    print(f"\nComputing blocking ratio at {optimal['threads']} threads...")
    avg_beta = compute_blocking_ratio(optimal['threads'], duration=30)
    print(f"Average blocking ratio (beta): {avg_beta:.3f}")
    
    # Save results
    output_data = {
        'python_version': sys.version,
        'platform': sys.platform,
        'numpy_version': np.__version__,
        'workload': 'NumPy convolution (MobileNetV2 simulation)',
        'io_latency_ms': 50,
        'results': results,
        'analysis': {
            'optimal_threads': optimal['threads'],
            'optimal_tps': optimal['tps'],
            'worst_threads': worst['threads'],
            'worst_tps': worst['tps'],
            'degradation_percent': round(degradation, 1),
            'blocking_ratio': avg_beta
        }
    }
    
    with open('numpy_vision_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to numpy_vision_results.json")
    
    return results

if __name__ == "__main__":
    main()
