import os
import torch
import torch.nn.utils.prune as prune
import torchvision.models as models
import torch.nn as nn

def prune_mobilenetv2_model(model, amount=0.3, min_layer=5):
    """
    Prunes only Conv2d layers in Bottleneck blocks at index >= min_layer in model.features.
    By default, skips the first several blocks to avoid hurting initial feature extraction.
    """
    for idx, block in enumerate(model.features):
        if idx >= min_layer:  # Only prune after min_layer (e.g., deeper feature extractors)
            for name, module in block.named_modules():
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
                    prune.remove(module, 'weight')  # Finalize pruning
    return model

def export_pruned_model_onnx(amount=0.3, input_path=None, output_path="mobilenetv2_pruned.onnx", num_classes: int = 10, min_prune_layer=5):
    """
    Prune a MobileNetV2 model and export to ONNX after loading weights.
    Optionally, only prunes Conv2d layers at features[min_prune_layer:] and beyond.
    """
    # Build and load model
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    if input_path and os.path.isfile(input_path):
        model.load_state_dict(torch.load(input_path, map_location='cpu'))
    
    model.eval()
    pruned_model = prune_mobilenetv2_model(model, amount=amount, min_layer=min_prune_layer)
    
    # OPTIONAL: Save pruned weights for retraining before export!
    torch.save(pruned_model.state_dict(), "pruned_mobilenetv2_cifar10.pth")

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        pruned_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Pruned model (prune ratio={amount}, from layer={min_prune_layer}) exported to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--amount', type=float, default=0.1, help="Fraction of channels to prune (e.g., 0.1 is safe)")
    parser.add_argument('--weights', type=str, required=True, help="Fine-tuned weights path (.pth) to prune")
    parser.add_argument('--output', default='mobilenetv2_pruned.onnx', help="Output ONNX path")
    parser.add_argument('--num-classes', type=int, default=10, help="Number of classes (CIFAR-10 = 10)")
    parser.add_argument('--min-prune-layer', type=int, default=5, help="Index of feature block to start pruning")
    args = parser.parse_args()

    export_pruned_model_onnx(args.amount, args.weights, args.output, num_classes=args.num_classes, min_prune_layer=args.min_prune_layer)
