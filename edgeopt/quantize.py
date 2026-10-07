"""ONNX Runtime quantization.

static (default for conv nets): INT8 weights and activations in QDQ format,
per-channel weight scales, activation ranges calibrated on real images. ORT
fuses the QDQ pairs into QLinearConv / integer kernels.

dynamic: only weights are stored as INT8 and activations are quantized on the
fly. ORT only has integer kernels for MatMul/Gemm here; Conv becomes
ConvInteger plus extra quantize/dequantize work per call, which is why the
dynamic INT8 MobileNetV2 was several times slower than FP32. Kept for
Linear/Transformer-style models.

fp16: precision conversion, not quantization.
"""

import os
import tempfile

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


class BatchListReader(CalibrationDataReader):
    def __init__(self, input_name, batches):
        self.input_name = input_name
        self._iter = iter(batches)

    def get_next(self):
        batch = next(self._iter, None)
        return None if batch is None else {self.input_name: batch}


def cifar10_calibration_batches(n=300, img_size=224, batch_size=10, seed=2):
    """n images from a fixed random subset of the CIFAR-10 *training* set."""
    from edgeopt.data import fixed_subset, load_cifar10, numpy_batches

    ds = fixed_subset(load_cifar10(train=True, img_size=img_size), n, seed=seed)
    return [x.astype(np.float32) for x, _ in numpy_batches(ds, batch_size)]


def synthetic_calibration_batches(n=32, input_shape=(3, 224, 224), batch_size=8, seed=0):
    """Gaussian noise. Only useful for tests or latency/size measurements:
    the resulting activation ranges are wrong for real images."""
    rng = np.random.RandomState(seed)
    return [rng.randn(min(batch_size, n - i), *input_shape).astype(np.float32) for i in range(0, n, batch_size)]


def _input_name(path):
    return onnx.load(path, load_external_data=False).graph.input[0].name


def quantize_onnx_static(input_path, output_path, calib_batches, per_channel=True, preprocess=True):
    print(f"Quantizing (static, QDQ, per_channel={per_channel}) {input_path} -> {output_path} "
          f"with {sum(len(b) for b in calib_batches)} calibration images...")
    with tempfile.TemporaryDirectory() as tmp:
        src = input_path
        if preprocess:
            # Shape inference + graph optimisation (e.g. folding) before inserting QDQ nodes,
            # as recommended by ONNX Runtime.
            src = os.path.join(tmp, "pre.onnx")
            quant_pre_process(input_path, src, skip_symbolic_shape=False)
        quantize_static(
            src,
            output_path,
            BatchListReader(_input_name(src), calib_batches),
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
    print("Static quantization done.")


def quantize_onnx_dynamic(input_path, output_path, quant_type="int8"):
    print(f"Quantizing (dynamic) {input_path} to {output_path} as {quant_type.upper()}...")
    if quant_type.lower() == "int8":
        quantize_dynamic(input_path, output_path, weight_type=QuantType.QUInt8)
    elif quant_type.lower() == "fp16":
        # keep_io_types leaves inputs/outputs as float32 so callers don't need to change.
        from onnxruntime.transformers.float16 import convert_float_to_float16

        onnx.save(convert_float_to_float16(onnx.load(input_path), keep_io_types=True), output_path)
    else:
        raise ValueError("Only 'int8' and 'fp16' quantization are supported")
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="model_int8.onnx")
    parser.add_argument("--calib-size", type=int, default=300)
    args = parser.parse_args()
    quantize_onnx_static(args.input, args.output, cifar10_calibration_batches(args.calib_size))
