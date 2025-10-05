import onnxruntime
import numpy as np
import time
import psutil
import os

def benchmark_onnx_model(model_path, N=200, batch_size=1, input_shape=(3, 224, 224)):
    ort_session = onnxruntime.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    input_data = np.random.randn(batch_size, *input_shape).astype(np.float32)

    times = []
    process = psutil.Process(os.getpid())

    # Warm-up runs
    for _ in range(10):
        ort_session.run(None, {input_name: input_data})

    # Benchmark runs
    mem_usages = []
    for _ in range(N):
        start_mem = process.memory_info().rss
        start = time.time()
        ort_session.run(None, {input_name: input_data})
        times.append(time.time() - start)
        end_mem = process.memory_info().rss
        mem_usages.append(end_mem - start_mem)

    latency_p50 = np.percentile(times, 50)
    latency_p95 = np.percentile(times, 95)
    throughput = batch_size / latency_p50
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

    print(f"Latency p50: {latency_p50 * 1000:.2f} ms")
    print(f"Latency p95: {latency_p95 * 1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} FPS")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Avg RSS delta: {np.mean(mem_usages)/1024:.2f} KB")

    # Return metrics for reporting
    return {
        "latency_p50": latency_p50, 
        "latency_p95": latency_p95,
        "throughput": throughput,
        "model_size": model_size,
        "memory": np.mean(mem_usages)
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Path to ONNX model file")
    parser.add_argument('--N', type=int, default=200, help="Number of benchmark runs")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size")
    args = parser.parse_args()

    benchmark_onnx_model(args.model, N=args.N, batch_size=args.batch_size)
