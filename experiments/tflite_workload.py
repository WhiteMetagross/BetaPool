"""
TensorFlow Lite Real ML Workload Benchmark
Tests adaptive controller with actual ML inference workload
"""
import os
import sys
import time
import json
import concurrent.futures
from statistics import mean
import numpy as np

# Configuration
MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v2_1.0_224_quant_and_labels.zip"
MODEL_FILE = "mobilenet_v2_1.0_224_quant.tflite"

def ensure_dependencies():
    """Ensure TensorFlow Lite and dependencies are installed"""
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        print("Installing TensorFlow...")
        os.system(f"{sys.executable} -m pip install tensorflow -q")
        import tensorflow as tf
        return tf

def download_model():
    """Download MobileNetV2 model if not present"""
    if os.path.exists(MODEL_FILE):
        print(f"Model already exists: {MODEL_FILE}")
        return True
    
    print(f"Downloading MobileNetV2 model...")
    try:
        import urllib.request
        import zipfile
        
        zip_file = "mobilenet_v2.zip"
        urllib.request.urlretrieve(MODEL_URL, zip_file)
        
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall('.')
        
        os.remove(zip_file)
        print(f"Model downloaded: {MODEL_FILE}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def create_interpreter(tf):
    """Create TFLite interpreter"""
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    return interpreter

def tflite_inference_workload(interpreter, input_details, output_details, io_latency_ms=50):
    """Real TFLite inference + simulated I/O latency"""
    # Generate random 224x224x3 image (uint8 quantized input)
    img = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
    
    # Inference (releases GIL via TFLite C++ code)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Simulate I/O (API call to send results)
    time.sleep(io_latency_ms / 1000)
    
    return int(np.argmax(output))

def benchmark_tflite(tf, thread_count, duration=60):
    """Benchmark TFLite workload at given thread count"""
    # Create one interpreter per thread would be ideal, but for simplicity
    # we create thread-local interpreters
    import threading
    
    thread_local = threading.local()
    
    def get_interpreter():
        if not hasattr(thread_local, 'interpreter'):
            thread_local.interpreter = create_interpreter(tf)
            thread_local.input_details = thread_local.interpreter.get_input_details()
            thread_local.output_details = thread_local.interpreter.get_output_details()
        return thread_local.interpreter, thread_local.input_details, thread_local.output_details
    
    def worker():
        interpreter, input_details, output_details = get_interpreter()
        return tflite_inference_workload(interpreter, input_details, output_details)
    
    completed = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = []
        while time.time() - start_time < duration:
            submit_time = time.time()
            f = executor.submit(worker)
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
    """Compute blocking ratio for TFLite workload"""
    import threading
    
    tf = ensure_dependencies()
    thread_local = threading.local()
    
    def get_interpreter():
        if not hasattr(thread_local, 'interpreter'):
            thread_local.interpreter = create_interpreter(tf)
            thread_local.input_details = thread_local.interpreter.get_input_details()
            thread_local.output_details = thread_local.interpreter.get_output_details()
        return thread_local.interpreter, thread_local.input_details, thread_local.output_details
    
    beta_values = []
    
    def worker_with_timing():
        wall_start = time.time()
        cpu_start = time.thread_time()
        
        interpreter, input_details, output_details = get_interpreter()
        tflite_inference_workload(interpreter, input_details, output_details)
        
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
    print("TensorFlow Lite Real ML Workload Benchmark")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    # Ensure dependencies
    tf = ensure_dependencies()
    print(f"TensorFlow version: {tf.__version__}")
    
    # Download model
    if not download_model():
        print("Failed to download model. Exiting.")
        return
    
    print()
    print("Benchmarking TFLite MobileNetV2 workload...")
    print("-" * 70)
    
    thread_counts = [1, 4, 8, 16, 32, 64, 128, 256]
    results = []
    
    for tc in thread_counts:
        print(f"  {tc:4d} threads...", end=' ', flush=True)
        result = benchmark_tflite(tf, tc, duration=60)
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
        'tensorflow_version': tf.__version__,
        'model': 'MobileNetV2 (quantized)',
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
    
    with open('tflite_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to tflite_results.json")
    
    return results

if __name__ == "__main__":
    main()
