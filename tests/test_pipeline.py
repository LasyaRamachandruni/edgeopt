"""Tests for pruning, export, quantization and benchmarking (CPU, no dataset needed)."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
import torchvision.models as models

from edgeopt.benchmark import benchmark_onnx_model
from edgeopt.prune import export_pruned_model_onnx, prune_mobilenetv2_model
from edgeopt.quantize import quantize_onnx_dynamic


def zero_channel_fraction(conv: nn.Conv2d) -> float:
    w = conv.weight.detach().flatten(1)
    return (w.abs().sum(dim=1) == 0).float().mean().item()


def mobilenet(num_classes=10):
    torch.manual_seed(0)
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m.eval()


def test_prune_zeroes_channels_only_in_later_blocks():
    model = prune_mobilenetv2_model(mobilenet(), amount=0.3, min_layer=5)
    for idx, block in enumerate(model.features):
        for conv in (m for m in block.modules() if isinstance(m, nn.Conv2d)):
            frac = zero_channel_fraction(conv)
            if idx < 5:
                assert frac == 0.0, f"block {idx} should be untouched"
            else:
                expected = round(0.3 * conv.out_channels) / conv.out_channels
                assert frac == pytest.approx(expected, abs=1e-6), f"block {idx}"


def test_pruning_is_permanent():
    # prune.remove() should leave plain weights, not a mask + original pair.
    model = prune_mobilenetv2_model(mobilenet(), amount=0.2, min_layer=5)
    names = {n for n, _ in model.named_parameters()}
    assert not any(n.endswith("weight_orig") for n in names)


def test_export_pruned_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # export also writes a .pth checkpoint to the cwd
    out = tmp_path / "pruned.onnx"
    export_pruned_model_onnx(amount=0.1, output_path=str(out), num_classes=10)
    onnx.checker.check_model(onnx.load(str(out)))

    sess = ort.InferenceSession(str(out))
    batch = np.random.randn(2, 3, 224, 224).astype(np.float32)
    (logits,) = sess.run(None, {sess.get_inputs()[0].name: batch})
    assert logits.shape == (2, 10)  # dynamic batch axis works


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def small_onnx(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "small.onnx"
    torch.onnx.export(SmallNet().eval(), torch.randn(1, 3, 16, 16), str(path),
                      input_names=["input"], output_names=["output"], opset_version=13,
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, dynamo=False)
    return path


@pytest.mark.parametrize("quant_type", ["int8", "fp16"])
def test_quantize_keeps_outputs_close(small_onnx, tmp_path, quant_type):
    out = tmp_path / f"small_{quant_type}.onnx"
    quantize_onnx_dynamic(str(small_onnx), str(out), quant_type=quant_type)

    x = np.random.randn(4, 3, 16, 16).astype(np.float32)
    ref = ort.InferenceSession(str(small_onnx)).run(None, {"input": x})[0]
    got = ort.InferenceSession(str(out)).run(None, {"input": x})[0]
    assert np.abs(ref - got).max() < 0.05 * np.abs(ref).max() + 1e-3
    ratio = out.stat().st_size / small_onnx.stat().st_size
    assert ratio < (0.4 if quant_type == "int8" else 0.6)  # ~4x smaller for int8, ~2x for fp16


def test_quantize_rejects_unknown_type(small_onnx, tmp_path):
    with pytest.raises(ValueError):
        quantize_onnx_dynamic(str(small_onnx), str(tmp_path / "x.onnx"), quant_type="int4")


def test_benchmark_reports_metrics(small_onnx):
    m = benchmark_onnx_model(str(small_onnx), N=20, batch_size=2, input_shape=(3, 16, 16))
    assert m["latency_p95"] >= m["latency_p50"] > 0
    assert m["throughput"] == pytest.approx(2 / m["latency_p50"])
    assert m["model_size"] > 0
