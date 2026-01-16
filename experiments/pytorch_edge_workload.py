"""
PyTorch/ONNX Runtime Edge AI Workload Benchmark
Tests adaptive controller with real ML inference workload using ONNX Runtime
ONNX Runtime is widely used for edge AI deployment (Raspberry Pi, Jetson, etc.)
"""
import os
import sys
import time
import json
import concurrent.futures
import threading
from statistics import mean
import numpy as np

def ensure_dependencies():
    """Ensure ONNX Runtime is installed"""
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        print("Installing onnxruntime...")
        os.system(f"{sys.executable} -m pip install onnxruntime -q")
        import onnxruntime as ort
        return ort

def download_mobilenet_onnx():
    """Download MobileNetV2 ONNX model if not present"""
    model_path = "mobilenetv2-7.onnx"
    if os.path.exists(model_path):
        print(f"Model already exists: {model_path}")
        return model_path
    
    print("Downloading MobileNetV2 ONNX model...")
    try:
        import urllib.request
        url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
        urllib.request.urlretrieve(url, model_path)
        print(f"Model downloaded: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Creating synthetic model instead...")
        return create_synthetic_model()

def create_synthetic_model():
    """Create a synthetic ONNX model for testing if download fails"""
    try:
        import onnx
        from onnx import helper, TensorProto
        
        # Create a simple conv model
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1000])
        
        # Simple model: flatten + matmul (simulates inference compute)
        flatten = helper.make_node('Flatten', ['input'], ['flat'], axis=1)
        
        # Create weight tensor
        W_data = np.random.randn(3*224*224, 1000).astype(np.float32) * 0.01
        W = helper.make_tensor('W', TensorProto.FLOAT, [3*224*224, 1000], W_data.flatten())
        
        matmul = helper.make_node('MatMul', ['flat', 'W'], ['output'])
        
        graph = helper.make_graph([flatten, matmul], 'synthetic_mobilenet', [X], [Y], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        
        model_path = "synthetic_mobilenet.onnx"
        onnx.save(model, model_path)
        print(f"Created synthetic model: {model_path}")
        return model_path
    except ImportError:
        return None

def create_numpy_inference_workload():
    """
    NumPy-based inference simulation that matches edge AI compute patterns.
    This simulates MobileNetV2-like computation without external dependencies.
    """
    # Simulate MobileNetV2 depthwise separable convolutions
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Depthwise conv simulation (channel-wise operations)
    for _ in range(5):  # Simulate multiple conv blocks
        # Depthwise: per-channel convolution
        kernel = np.random.randn(3, 3, 3).astype(np.float32) * 0.1
        # Simplified conv operation
        output = np.sum(input_data * kernel.reshape(1, 3, 1, 1), axis=1, keepdims=True)
        output = np.maximum(output, 0)  # ReLU
        
        # Pointwise: 1x1 convolution (matrix multiply)
        weights = np.random.randn(32, 3).astype(np.float32) * 0.1
        
    # Final classification layer
    flat = input_data.flatten()[:1000]
    logits = flat @ np.random.randn(1000, 1000).astype(np.float32) * 0.01
    
    return int(np.argmax(logits))

class ONNXInferenceWorker:
    """Thread-local ONNX Runtime session for inference"""
    
    def __init__(self, model_path, io_latency_ms=50):
        self.model_path = model_path
        self.io_latency_ms = io_latency_ms
        self.thread_local = threading.local()
    
    def get_session(self):
        if not hasattr(self.thread_local, 'session'):
            import onnxruntime as ort
            # Use single thread for inference to avoid nested threading issues
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            self.thread_local.session = ort.InferenceSession(
                self.model_path, 
                sess_options,
                providers=['CPUExecutionProvider']
            )
        return self.thread_local.session
    
    def inference(self):
        """Run inference + simulated I/O"""
        session = self.get_session()
        
        # Generate random input image
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run inference (releases GIL during ONNX Runtime C++ execution)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_data})
        
        # Simulate I/O (API call to send results)
        time.sleep(self.io_latency_ms / 1000)
        
        return int(np.argmax(output[0]))

def benchmark_inference(workload_fn, thread_count, duration=60):
    """Benchmark inference workload at given thread count"""
    completed = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = []
        while time.time() - start_time < duration:
            submit_time = time.time()
            f = executor.submit(workload_fn)
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

def compute_blocking_ratio(workload_fn, thread_count, duration=30):
    """Compute blocking ratio for the workload"""
    beta_values = []
    
    def worker_with_timing():
        wall_start = time.time()
        cpu_start = time.thread_time()
        
        workload_fn()
        
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

def numpy_inference_with_io():
    """NumPy inference + 50ms I/O simulation"""
    result = create_numpy_inference_workload()
    time.sleep(0.050)  # 50ms network latency
    return result

def main():
    print("=" * 70)
    print("Edge AI Inference Workload Benchmark")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check GIL status
    try:
        gil_enabled = sys._is_gil_enabled()
        print(f"GIL enabled: {gil_enabled}")
    except AttributeError:
        gil_enabled = True
        print("GIL enabled: True (Python < 3.13)")
    
    print()
    
    # Try ONNX Runtime first, fall back to NumPy
    use_onnx = False
    workload_name = "NumPy Vision Inference"
    workload_fn = numpy_inference_with_io
    
    try:
        ort = ensure_dependencies()
        model_path = download_mobilenet_onnx()
        if model_path and os.path.exists(model_path):
            worker = ONNXInferenceWorker(model_path, io_latency_ms=50)
            workload_fn = worker.inference
            workload_name = "ONNX Runtime MobileNetV2"
            use_onnx = True
            print(f"Using ONNX Runtime: {ort.__version__}")
    except Exception as e:
        print(f"ONNX Runtime not available: {e}")
        print("Falling back to NumPy-based inference simulation")
    
    print(f"\nWorkload: {workload_name}")
    print(f"I/O latency simulation: 50ms")
    print("-" * 70)
    
    # Benchmark configuration
    thread_counts = [1, 4, 8, 16, 32, 64, 128, 256]
    test_duration = 30  # seconds per test
    
    results = []
    
    for tc in thread_counts:
        print(f"  {tc:4d} threads...", end=' ', flush=True)
        result = benchmark_inference(workload_fn, tc, duration=test_duration)
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
    avg_beta = compute_blocking_ratio(workload_fn, optimal['threads'], duration=15)
    print(f"Average blocking ratio (beta): {avg_beta:.3f}")
    
    # Save results
    gil_status = "nogil" if not gil_enabled else "gil"
    py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    
    output_data = {
        'python_version': sys.version,
        'gil_enabled': gil_enabled,
        'platform': sys.platform,
        'workload': workload_name,
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
    
    filename = f"pytorch_edge_results_{gil_status}_py{py_version}.json"
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    
    return results

if __name__ == "__main__":
    main()
