#!/usr/bin/env bash
# Re-run pending: everything that needs CIFAR-10 images. Run on a machine that can
# download CIFAR-10 (torchvision or the Hugging Face "uoft-cs/cifar10" dataset).
# Budget on a 2-core CPU: fine-tuning 5k images x 1 epoch at 224px is the slow part.
set -euo pipefail
export OMP_NUM_THREADS=1
mkdir -p models

# 1. Recovery fine-tune after pruning (fixed 5,000-image training subset, 1 epoch, 1 thread).
for a in 30 50; do
  python scripts/prune_finetune_export.py --weights mobilenetv2_cifar10.pth --amount 0.$a \
    --train-subset 5000 --epochs 1 --threads 1 \
    --save models/pruned${a}_ft.pt --output models/mobilenetv2_pruned${a}_ft.onnx
done

# 2. Static INT8 calibrated on 300 real CIFAR-10 training images.
edgeopt quantize -i models/mobilenetv2_fp32.onnx       -o models/mobilenetv2_int8.onnx          --calib-size 300
edgeopt quantize -i models/mobilenetv2_pruned50_ft.onnx -o models/mobilenetv2_pruned50_ft_int8.onnx --calib-size 300

# 3. Benchmark + accuracy on a fixed 2,000-image subset of the CIFAR-10 test set
#    (drop --eval-subset to use all 10,000).
edgeopt report --threads 1 --N 300 --warmup 30 --repeats 5 --eval-subset 2000 --out-dir results --models \
  "FP32 baseline=models/mobilenetv2_fp32.onnx" \
  "Pruned 30% + FT=models/mobilenetv2_pruned30_ft.onnx" \
  "Pruned 50% + FT=models/mobilenetv2_pruned50_ft.onnx" \
  "INT8 static=models/mobilenetv2_int8.onnx" \
  "Pruned 50% + FT + INT8 static=models/mobilenetv2_pruned50_ft_int8.onnx"
