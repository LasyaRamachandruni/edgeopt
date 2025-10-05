import torch
import torch.nn.utils.prune as prune
import torchvision.models as models

def prune_model(model, amount=0.3):
    """
    Structurally prunes (removes) a fraction of channels from each Conv2d layer.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, 'weight')  # Removes mask, finalizes pruning
    return model

def export_pruned_model_onnx(amount=0.3, output_path="mobilenetv2_pruned.onnx"):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.eval()
    pruned_model = prune_model(model, amount=amount)
    sample_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        pruned_model,
        sample_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Pruned model (prune ratio={amount}) exported to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--amount', type=float, default=0.3, help="Fraction of channels to prune (e.g., 0.3 for 30%)")
    parser.add_argument('--output', default='mobilenetv2_pruned.onnx', help="Output ONNX path")
    args = parser.parse_args()

    export_pruned_model_onnx(args.amount, args.output)
