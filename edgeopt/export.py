import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import torchvision.models as models

from edgeopt.utils import onnx_export_kwargs


def build_mobilenetv2(num_classes=10, weights_path=None):
    """MobileNetV2 with a num_classes head, optionally loading a state_dict."""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model.eval()


def export_model_onnx(model, output_path, img_size=224, opset=13):
    model.eval()
    torch.onnx.export(
        model,
        torch.randn(1, 3, img_size, img_size),
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        **onnx_export_kwargs(),
    )


def export_finetuned_to_onnx(weights_path="mobilenetv2_cifar10.pth", output_path="mobilenetv2_cifar10.onnx", num_classes=10):
    export_model_onnx(build_mobilenetv2(num_classes, weights_path), output_path)
    print(f"Fine-tuned CIFAR-10 model exported to {output_path}")


def verify_onnx_model(onnx_path="mobilenetv2_cifar10.onnx"):
    onnx.checker.check_model(onnx.load(onnx_path))
    print("ONNX model is valid.")
    sess = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})
    print("ONNX inference output shape:", out[0].shape)


if __name__ == "__main__":
    export_finetuned_to_onnx()
    verify_onnx_model()
