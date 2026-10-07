"""Structured channel pruning that actually removes channels.

The earlier version used torch.nn.utils.prune.ln_structured, which only zeroes
filters: tensor shapes, parameter count, ONNX size and latency stayed the same.
Here torch-pruning builds a dependency graph of the network (depthwise convs,
BatchNorms and residual adds that must lose the same channels) and slices the
weights, so the pruned model is genuinely smaller.
"""

import torch
import torch.nn as nn
import torch_pruning as tp

from edgeopt.export import build_mobilenetv2, export_model_onnx


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def prune_mobilenetv2_model(model, amount=0.3, img_size=224, round_to=8):
    """Remove `amount` of the channels in every prunable layer (L2-magnitude ranking).

    The classifier output layer is left alone so the model still predicts the same
    classes. Channel counts are rounded to multiples of `round_to`, which keeps
    them friendly to SIMD kernels. Modifies `model` in place and returns it.
    """
    if not 0.0 <= amount < 1.0:
        raise ValueError("amount must be in [0, 1)")
    model.eval()
    if amount == 0:
        return model
    example = torch.randn(1, 3, img_size, img_size)
    head = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    pruner = tp.pruner.MagnitudePruner(
        model,
        example,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=amount,
        ignored_layers=[head],
        round_to=round_to,
    )
    pruner.step()
    return model


def save_pruned(model, path):
    # The architecture changed, so a plain state_dict can't be loaded back into
    # torchvision's mobilenet_v2. Save the whole module instead.
    torch.save(model, path)


def load_pruned(path):
    return torch.load(path, map_location="cpu", weights_only=False).eval()


def export_pruned_model_onnx(amount=0.3, input_path=None, output_path="mobilenetv2_pruned.onnx",
                             num_classes=10, save_path=None, img_size=224):
    model = build_mobilenetv2(num_classes, input_path)
    before = count_params(model)
    prune_mobilenetv2_model(model, amount=amount, img_size=img_size)
    after = count_params(model)
    if save_path:
        save_pruned(model, save_path)
    export_model_onnx(model, output_path, img_size=img_size)
    print(f"Pruned {amount:.0%} of channels: {before:,} -> {after:,} params ({after / before:.1%}). "
          f"Exported to {output_path}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--amount", type=float, default=0.3, help="Fraction of channels to remove per layer")
    parser.add_argument("--weights", default=None, help="Fine-tuned state_dict (.pth) to prune")
    parser.add_argument("--output", default="mobilenetv2_pruned.onnx")
    parser.add_argument("--save", default=None, help="Also save the pruned PyTorch module here")
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()
    export_pruned_model_onnx(args.amount, args.weights, args.output, args.num_classes, args.save)
