#!/usr/bin/env bash
# Commands that produced results/results.json and results/results.md (2026-10-07).
# Hardware: 2 logical CPUs (Intel Xeon @ 2.10GHz, AVX-512 VNNI), no GPU, shared with
# another job. ONNX Runtime pinned to 1 intra-op thread.
set -euo pipefail
export OMP_NUM_THREADS=1
mkdir -p models

# Fine-tuned FP32 checkpoint (trained earlier with scripts/finetune_mobilenet_cifar.py;
# it was committed in 20db4db and later untracked).
git show 20db4db:mobilenetv2_cifar10.pth > mobilenetv2_cifar10.pth

edgeopt export -w mobilenetv2_cifar10.pth -o models/mobilenetv2_fp32.onnx --verify
edgeopt prune  -w mobilenetv2_cifar10.pth --amount 0.3 -o models/mobilenetv2_pruned30.onnx --save models/pruned30.pt
edgeopt prune  -w mobilenetv2_cifar10.pth --amount 0.5 -o models/mobilenetv2_pruned50.onnx --save models/pruned50.pt

# CIFAR-10 could not be downloaded here (torchvision and Hugging Face both blocked
# by the network proxy), so static INT8 was calibrated on random noise. That is fine
# for size and latency (same graph, only the scale values differ) but not for accuracy.
for m in fp32 pruned30 pruned50; do
  edgeopt quantize -i models/mobilenetv2_$m.onnx -o models/mobilenetv2_${m}_int8_synthcalib.onnx \
    --mode static --calib-data synthetic --calib-size 300
done
edgeopt quantize -i models/mobilenetv2_fp32.onnx -o models/mobilenetv2_int8_dynamic.onnx --mode dynamic

edgeopt report --threads 1 --N 300 --warmup 30 --repeats 5 --out-dir results --models \
  "FP32 baseline=models/mobilenetv2_fp32.onnx" \
  "Pruned 30% (not fine-tuned)=models/mobilenetv2_pruned30.onnx" \
  "Pruned 50% (not fine-tuned)=models/mobilenetv2_pruned50.onnx" \
  "INT8 static QDQ (synthetic calib)=models/mobilenetv2_fp32_int8_synthcalib.onnx" \
  "Pruned 30% + INT8 static (synthetic calib)=models/mobilenetv2_pruned30_int8_synthcalib.onnx" \
  "Pruned 50% + INT8 static (synthetic calib)=models/mobilenetv2_pruned50_int8_synthcalib.onnx" \
  "INT8 dynamic (previous method)=models/mobilenetv2_int8_dynamic.onnx"
