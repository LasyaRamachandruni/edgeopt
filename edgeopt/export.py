import torch
import torchvision.models as models
import onnx
import onnxruntime
import numpy as np


def export_pytorch_to_onnx(output_path="mobilenetv2.onnx"):
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)
    model.eval()

    # Create a sample input with batch size 1, 3 channels, 224x224 size
    sample_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    export_pytorch_to_onnx()


def verify_onnx_model(onnx_path="mobilenetv2.onnx"):
    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    # Run inference with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(onnx_path)
    sample_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: sample_input}
    ort_outs = ort_session.run(None, ort_inputs)
    print("ONNX inference output shape:", ort_outs[0].shape)

if __name__ == "__main__":
    export_pytorch_to_onnx()
    verify_onnx_model()
