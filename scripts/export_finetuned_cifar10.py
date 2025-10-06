import torch
import torchvision.models as models
import onnx
import onnxruntime
import numpy as np

def export_finetuned_to_onnx(weights_path="mobilenetv2_cifar10.pth", output_path="mobilenetv2_cifar10.onnx"):
    # Load MobileNetV2 architecture
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Sample input
    sample_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        export_params=True,
        opset_version=13,  # 17 only if ONNX Runtime supports it for all ops you use
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Fine-tuned CIFAR-10 model exported to {output_path}")

def verify_onnx_model(onnx_path="mobilenetv2_cifar10.onnx"):
    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    # Run inference
    ort_session = onnxruntime.InferenceSession(onnx_path)
    sample_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: sample_input}
    ort_outs = ort_session.run(None, ort_inputs)
    print("ONNX inference output shape:", ort_outs[0].shape)

if __name__ == "__main__":
    export_finetuned_to_onnx()
    verify_onnx_model()
