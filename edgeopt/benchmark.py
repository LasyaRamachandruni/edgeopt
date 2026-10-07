import os
import time

import numpy as np
import onnxruntime


def summarize_timings(times_s, batch_size):
    """Latency percentiles (ms) and throughput (images/s) from per-run wall times.

    Throughput is total images processed divided by total measured wall time,
    not batch_size / p50, so slow outliers are counted.
    """
    times_s = np.asarray(times_s, dtype=np.float64)
    total = float(times_s.sum())
    return {
        "latency_p50_ms": float(np.percentile(times_s, 50) * 1000),
        "latency_p95_ms": float(np.percentile(times_s, 95) * 1000),
        "latency_mean_ms": float(times_s.mean() * 1000),
        "throughput_ips": len(times_s) * batch_size / total,
        "runs": int(len(times_s)),
        "batch_size": int(batch_size),
    }


def make_session(model_path, threads=0):
    opts = onnxruntime.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])


def benchmark_onnx_model(model_path, N=200, batch_size=1, input_shape=(3, 224, 224), warmup=20, threads=0, seed=0):
    sess = make_session(model_path, threads)
    input_name = sess.get_inputs()[0].name
    x = np.random.RandomState(seed).randn(batch_size, *input_shape).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: x})

    times = []
    for _ in range(N):
        start = time.perf_counter()
        sess.run(None, {input_name: x})
        times.append(time.perf_counter() - start)

    metrics = summarize_timings(times, batch_size)
    metrics["warmup"] = warmup
    metrics["threads"] = threads
    metrics["model_size_mb"] = os.path.getsize(model_path) / 1e6

    print(f"Latency p50: {metrics['latency_p50_ms']:.2f} ms")
    print(f"Latency p95: {metrics['latency_p95_ms']:.2f} ms")
    print(f"Throughput:  {metrics['throughput_ips']:.1f} images/s ({N} runs, batch {batch_size}, {warmup} warmup)")
    print(f"Model size:  {metrics['model_size_mb']:.2f} MB")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--N", type=int, default=200, help="Number of timed runs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = runtime default)")
    args = parser.parse_args()
    benchmark_onnx_model(args.model, N=args.N, batch_size=args.batch_size, warmup=args.warmup, threads=args.threads)
