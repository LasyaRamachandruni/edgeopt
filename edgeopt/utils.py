import inspect

import torch


def onnx_export_kwargs() -> dict:
    """Extra arguments for torch.onnx.export that keep the classic exporter.

    PyTorch 2.9+ switched torch.onnx.export to a new exporter by default, which needs
    the separate `onnxscript` package and handles `dynamic_axes` differently. Our
    exports are written for the classic TorchScript-based exporter, so request it
    explicitly on versions that support the `dynamo` flag (2.5+).
    """
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        return {"dynamo": False}
    return {}
