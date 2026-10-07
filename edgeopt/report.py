"""Benchmark (and, when CIFAR-10 is available, evaluate) a set of ONNX models and
write results/results.json plus a markdown table."""

import datetime
import json
import os
import platform

import numpy as np
import onnx
import onnxruntime

from edgeopt.benchmark import benchmark_onnx_model

_QPARAM_SUFFIXES = ("_scale", "_zero_point")


def onnx_param_count(path):
    """Number of weight elements stored in the graph (quantization scales/zero points excluded)."""
    model = onnx.load(path)
    total = 0
    for init in model.graph.initializer:
        if init.name.endswith(_QPARAM_SUFFIXES):
            continue
        n = int(np.prod(init.dims)) if len(init.dims) else 1
        if n > 1:  # skip scalar constants (shape helpers etc.)
            total += n
    return total


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def collect(models, N=200, warmup=20, threads=1, eval_subset=None, accuracy=True, data_root="./data"):
    testset, acc_note = None, None
    if accuracy:
        try:
            from edgeopt.data import fixed_subset, load_cifar10

            testset = fixed_subset(load_cifar10(train=False, root=data_root), eval_subset)
        except Exception as e:
            acc_note = f"CIFAR-10 unavailable: {str(e).splitlines()[-1]}"
            print(acc_note)
    else:
        acc_note = "accuracy not requested"

    rows = []
    for name, path in models:
        print(f"== {name}: {path}")
        bench = benchmark_onnx_model(path, N=N, warmup=warmup, threads=threads)
        row = {
            "model": name,
            "path": os.path.basename(path),
            "params": onnx_param_count(path),
            "size_mb": round(os.path.getsize(path) / 1e6, 2),
            "latency_p50_ms": round(bench["latency_p50_ms"], 2),
            "latency_p95_ms": round(bench["latency_p95_ms"], 2),
            "throughput_ips": round(bench["throughput_ips"], 1),
            "accuracy": None,
            "input_shape": bench["input_shape"],
        }
        if testset is not None:
            from edgeopt.eval import accuracy_onnx

            row["accuracy"] = round(accuracy_onnx(path, testset), 2)
        rows.append(row)

    meta = {
        "date": datetime.date.today().isoformat(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "ort_threads": threads,
        "onnxruntime": onnxruntime.__version__,
        "benchmark": f"batch 1, random input of the model's input shape, {warmup} warmup + {N} timed runs",
        "accuracy_set": None if testset is None else f"CIFAR-10 test, {len(testset)} images",
        "accuracy_note": acc_note,
    }
    return rows, meta


def markdown_table(rows):
    head = ("| Model | Top-1 acc (%) | Params | ONNX size (MB) | p50 (ms) | p95 (ms) | Throughput (img/s) |\n"
            "|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        acc = "pending" if r["accuracy"] is None else f"{r['accuracy']:.2f}"
        body += (f"| {r['model']} | {acc} | {r['params']:,} | {r['size_mb']:.2f} | "
                 f"{r['latency_p50_ms']:.2f} | {r['latency_p95_ms']:.2f} | {r['throughput_ips']:.1f} |\n")
    return head + body


def write_report(rows, meta, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": rows}, f, indent=2)
    md = markdown_table(rows) + "\n" + "\n".join(f"- {k}: {v}" for k, v in meta.items() if v is not None) + "\n"
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(md)
    print(md)
    return md


def parse_model_args(items):
    """['FP32=a.onnx', 'INT8=b.onnx'] -> [('FP32', 'a.onnx'), ('INT8', 'b.onnx')]"""
    out = []
    for item in items:
        name, sep, path = item.partition("=")
        if not sep:
            name, path = os.path.splitext(os.path.basename(item))[0], item
        out.append((name, path))
    return out
