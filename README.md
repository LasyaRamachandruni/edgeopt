# EdgeOpt

A small CLI for taking a MobileNetV2 fine-tuned on CIFAR-10 to ONNX and measuring what
pruning and INT8 quantization actually buy you on a CPU with ONNX Runtime.

- `export`: PyTorch state_dict to ONNX (dynamic batch axis, opset 13)
- `prune`: structured channel pruning with [torch-pruning](https://github.com/VainF/Torch-Pruning).
  Channels are physically removed (conv, BN, depthwise and residual-coupled layers are sliced
  together), so parameters, file size and latency go down, not just the number of zeros
- `quantize`: ONNX Runtime static INT8 (QDQ, per-channel weights, activations calibrated on
  CIFAR-10 training images). Dynamic INT8 and FP16 are still available
- `benchmark`: batch-1 latency p50/p95 and throughput (images / total timed wall time)
- `evaluate`: top-1 accuracy on the CIFAR-10 test set or a fixed seeded subset
- `report`: runs benchmark (+ evaluate when CIFAR-10 is available) for a list of models and
  writes `results/results.json` and `results/results.md`

## Install

```bash
git clone https://github.com/LasyaRamachandruni/edgeopt.git
cd edgeopt
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Workflow

```bash
# 1. Fine-tune ImageNet-pretrained MobileNetV2 on CIFAR-10 (224x224 inputs)
python scripts/finetune_mobilenet_cifar.py            # -> mobilenetv2_cifar10.pth

# 2. Export the FP32 baseline
edgeopt export -w mobilenetv2_cifar10.pth -o models/mobilenetv2_fp32.onnx --verify

# 3. Prune 30% of channels per layer, fine-tune briefly, export
python scripts/prune_finetune_export.py --weights mobilenetv2_cifar10.pth --amount 0.3 \
    --train-subset 5000 --epochs 1 --output models/mobilenetv2_pruned30_ft.onnx

# 4. Static INT8, calibrated on 300 CIFAR-10 training images
edgeopt quantize -i models/mobilenetv2_fp32.onnx -o models/mobilenetv2_int8.onnx --calib-size 300

# 5. Compare
edgeopt report --threads 1 --eval-subset 2000 --models \
    "FP32=models/mobilenetv2_fp32.onnx" "Pruned 30%=models/mobilenetv2_pruned30_ft.onnx" \
    "INT8=models/mobilenetv2_int8.onnx"
```

`edgeopt prune` alone (no fine-tuning) is also available. The pruned module is saved whole
(`--save pruned.pt`) because its layer shapes no longer match torchvision's `mobilenet_v2`.

## Results

Measured 2026-10-07 on a 2-vCPU Intel Xeon @ 2.10GHz (AVX-512 VNNI), no GPU, shared with
another job. ONNX Runtime 1.29.0 with 1 intra-op thread. Batch 1, 3x224x224 input,
30 warmup + 300 timed runs, repeated 5 times with the models interleaved; the table shows
the median of the 5 repeats (every repeat is in `results/results.json`). Exact commands:
[`results/commands.sh`](results/commands.sh).

| Model | Top-1 acc (%) | Params | ONNX size (MB) | p50 (ms) | p95 (ms) | Throughput (img/s) |
|---|---|---|---|---|---|---|
| FP32 baseline | pending | 2,219,626 | 8.92 | 6.33 | 9.43 | 148.7 |
| Pruned 30% (not fine-tuned) | pending | 1,079,842 | 4.36 | 3.61 | 5.45 | 255.8 |
| Pruned 50% (not fine-tuned) | pending | 577,586 | 2.35 | 2.68 | 4.21 | 319.4 |
| INT8 static QDQ (synthetic calib) | pending | 2,219,626 | 2.61 | 3.51 | 5.15 | 263.5 |
| Pruned 30% + INT8 static (synthetic calib) | pending | 1,079,842 | 1.39 | 2.56 | 6.44 | 331.9 |
| Pruned 50% + INT8 static (synthetic calib) | pending | 577,586 | 0.83 | 2.23 | 5.78 | 370.4 |
| INT8 dynamic (previous method) | pending | 2,219,834 | 2.42 | 23.06 | 32.95 | 40.7 |

Params are the weight elements stored in the ONNX graph (BatchNorm is folded into the convs on
export, so the FP32 count is slightly below the 2,236,682 PyTorch parameters). "Pruned 30%"
removes 30% of the channels in every layer except the classifier output, rounded to multiples
of 8; since most convs lose channels on both their input and output side, parameters drop by
about half.

**Accuracy is pending.** CIFAR-10 could not be downloaded on the benchmark machine: both
torchvision (cs.toronto.edu) and the Hugging Face `uoft-cs/cifar10` dataset were blocked by
the network proxy (HTTP 403). That also means:

- the pruned models in the table have not been fine-tuned after pruning, and
- the INT8 models were calibrated on random noise instead of training images.

Neither affects the size, parameter or latency columns (fine-tuning and calibration change
weight and scale values, not the graph), but the accuracy of these exact files would be poor
and is deliberately not reported. Everything that needs data is in
[`results/pending.sh`](results/pending.sh): fine-tune each pruned model for 1 epoch on a fixed
5,000-image training subset, calibrate INT8 on 300 real training images, then run
`edgeopt report ... --eval-subset 2000` (fixed seeded 2,000-image test subset). A smoke test of
the fine-tune loop on this machine took about 0.13 s/image for the 50% model at 224x224 with one
thread, so roughly 10-15 minutes per pruned model.

### What the numbers say

- Structural pruning works as advertised now: 30% of channels gives 2.0x fewer parameters, a
  2.0x smaller file and 1.75x lower p50 latency. The previous `ln_structured` version only
  zeroed weights and the ONNX file stayed at 8.9 MB.
- Static INT8 is 3.4x smaller and 1.8x faster than FP32 on this CPU. Dynamic INT8, which the
  project used before, is 3.6x *slower* than FP32 (23 ms vs 6.3 ms): ORT has no fused integer
  path for dynamically quantized Conv, so it runs ConvInteger and re-quantizes activations on
  every call. This CPU has VNNI int8 dot-product instructions; on CPUs without them, static INT8
  gains will be smaller.
- Pruning and INT8 stack, but with diminishing returns on latency: at batch 1 the 50% pruned
  model is already small enough that quantize/dequantize overhead and per-op dispatch are a
  large share of the 2-3 ms. p95 is noisier for the INT8 models because the machine was shared.

## Layout

| File | |
|---|---|
| `edgeopt/export.py` | build MobileNetV2 with a 10-class head, export to ONNX |
| `edgeopt/prune.py` | structured pruning (torch-pruning dependency graph) |
| `edgeopt/train.py` | short recovery fine-tune and PyTorch accuracy |
| `edgeopt/quantize.py` | static QDQ INT8, dynamic INT8, FP16 |
| `edgeopt/benchmark.py` | latency/throughput |
| `edgeopt/eval.py` | ONNX accuracy on CIFAR-10 |
| `edgeopt/data.py` | CIFAR-10 loading (torchvision, then Hugging Face) |
| `edgeopt/report.py` | results.json / results.md |
| `scripts/finetune_mobilenet_cifar.py` | initial CIFAR-10 fine-tune |
| `scripts/prune_finetune_export.py` | prune, fine-tune, export |

## Contact

Swathi Sri Lasya Mayukha Ramachandruni - swathisrilasyamayukha.ramachandruni@sjsu.edu -
[@LasyaRamachandruni](https://github.com/LasyaRamachandruni)
