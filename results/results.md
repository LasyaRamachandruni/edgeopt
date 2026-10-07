| Model | Top-1 acc (%) | Params | ONNX size (MB) | p50 (ms) | p95 (ms) | Throughput (img/s) |
|---|---|---|---|---|---|---|
| FP32 baseline | pending | 2,219,626 | 8.92 | 6.33 | 9.43 | 148.7 |
| Pruned 30% (not fine-tuned) | pending | 1,079,842 | 4.36 | 3.61 | 5.45 | 255.8 |
| Pruned 50% (not fine-tuned) | pending | 577,586 | 2.35 | 2.68 | 4.21 | 319.4 |
| INT8 static QDQ (synthetic calib) | pending | 2,219,626 | 2.61 | 3.51 | 5.15 | 263.5 |
| Pruned 30% + INT8 static (synthetic calib) | pending | 1,079,842 | 1.39 | 2.56 | 6.44 | 331.9 |
| Pruned 50% + INT8 static (synthetic calib) | pending | 577,586 | 0.83 | 2.23 | 5.78 | 370.4 |
| INT8 dynamic (previous method) | pending | 2,219,834 | 2.42 | 23.06 | 32.95 | 40.7 |

- date: 2026-10-07
- cpu: Intel(R) Xeon(R) Processor @ 2.10GHz
- logical_cpus: 2
- ort_threads: 1
- onnxruntime: 1.29.0
- benchmark: batch 1, random input of the model's input shape, 30 warmup + 300 timed runs per repeat; median of 5 repeat(s), models interleaved
- accuracy_note: CIFAR-10 unavailable: huggingface uoft-cs/cifar10: 403 Forbidden
