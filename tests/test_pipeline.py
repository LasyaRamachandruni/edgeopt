"""Tests for pruning, export, quantization and benchmarking (CPU, no dataset needed)."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
import torchvision.models as models

from edgeopt.benchmark import benchmark_onnx_model, summarize_timings
from edgeopt.export import export_model_onnx
from edgeopt.prune import count_params, export_pruned_model_onnx, load_pruned, prune_mobilenetv2_model
from edgeopt.quantize import quantize_onnx_dynamic, quantize_onnx_static, synthetic_calibration_batches


def mobilenet(num_classes=10):
    torch.manual_seed(0)
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m.eval()


@pytest.mark.parametrize("amount", [0.3, 0.5])
def test_prune_removes_parameters_and_keeps_output_shape(amount):
    model = mobilenet()
    before = count_params(model)
    prune_mobilenetv2_model(model, amount=amount, img_size=64)
    after = count_params(model)
    # Channels are removed on both sides of most convs, so params shrink by more than `amount`.
    assert after < (1 - amount) * before
    out = model(torch.randn(3, 3, 64, 64))
    assert out.shape == (3, 10)


def test_prune_shrinks_conv_tensors():
    model = prune_mobilenetv2_model(mobilenet(), amount=0.5, img_size=64)
    ref = mobilenet()
    pruned_convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    ref_convs = [m for m in ref.modules() if isinstance(m, nn.Conv2d)]
    assert len(pruned_convs) == len(ref_convs)
    assert sum(c.out_channels for c in pruned_convs) < 0.6 * sum(c.out_channels for c in ref_convs)
    assert model.classifier[1].out_features == 10


def test_prune_rejects_bad_amount():
    with pytest.raises(ValueError):
        prune_mobilenetv2_model(mobilenet(), amount=1.0)


def test_export_pruned_model(tmp_path):
    full = tmp_path / "full.onnx"
    out = tmp_path / "pruned.onnx"
    export_model_onnx(mobilenet(), str(full))
    export_pruned_model_onnx(amount=0.3, output_path=str(out), num_classes=10, save_path=str(tmp_path / "p.pt"))
    onnx.checker.check_model(onnx.load(str(out)))
    assert out.stat().st_size < 0.6 * full.stat().st_size

    sess = ort.InferenceSession(str(out))
    batch = np.random.randn(2, 3, 224, 224).astype(np.float32)
    (logits,) = sess.run(None, {sess.get_inputs()[0].name: batch})
    assert logits.shape == (2, 10)  # dynamic batch axis works

    reloaded = load_pruned(str(tmp_path / "p.pt"))
    assert count_params(reloaded) < count_params(mobilenet())


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


class SmallConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc(self.features(x))


def test_static_quantized_model_loads_and_runs(tmp_path):
    torch.manual_seed(0)
    src = tmp_path / "conv.onnx"
    torch.onnx.export(SmallConvNet().eval(), torch.randn(1, 3, 32, 32), str(src),
                      input_names=["input"], output_names=["output"], opset_version=13,
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, dynamo=False)
    out = tmp_path / "conv_int8.onnx"
    calib = synthetic_calibration_batches(n=32, input_shape=(3, 32, 32))
    quantize_onnx_static(str(src), str(out), calib)

    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    ops = {n.op_type for n in model.graph.node}
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops

    x = calib[0]
    ref = ort.InferenceSession(str(src)).run(None, {"input": x})[0]
    got = ort.InferenceSession(str(out)).run(None, {"input": x})[0]
    assert got.shape == (len(x), 10)
    assert np.abs(ref - got).max() < 0.1 * np.abs(ref).max() + 1e-3


def test_quantize_rejects_unknown_type(small_onnx, tmp_path):
    with pytest.raises(ValueError):
        quantize_onnx_dynamic(str(small_onnx), str(tmp_path / "x.onnx"), quant_type="int4")


def test_benchmark_reports_metrics(small_onnx):
    m = benchmark_onnx_model(str(small_onnx), N=20, batch_size=2, input_shape=(3, 16, 16), warmup=2)
    assert m["latency_p95_ms"] >= m["latency_p50_ms"] > 0
    assert m["runs"] == 20 and m["batch_size"] == 2
    assert m["model_size_mb"] > 0


def test_throughput_uses_total_wall_time():
    # 3 runs of 10 ms and one 70 ms outlier: 4 runs x 4 images in 0.1 s = 160 images/s.
    # batch / p50 would have reported 400 images/s.
    m = summarize_timings([0.01, 0.01, 0.01, 0.07], batch_size=4)
    assert m["throughput_ips"] == pytest.approx(160.0)
    assert m["latency_p50_ms"] == pytest.approx(10.0)
    assert m["latency_mean_ms"] == pytest.approx(25.0)
    assert m["latency_p95_ms"] > m["latency_p50_ms"]
