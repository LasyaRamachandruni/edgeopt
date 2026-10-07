"""Prune a fine-tuned MobileNetV2 (CIFAR-10), briefly fine-tune it, export to ONNX.

Example:

    OMP_NUM_THREADS=1 python scripts/prune_finetune_export.py \
        --weights mobilenetv2_cifar10.pth --amount 0.3 \
        --train-subset 5000 --epochs 1 --output models/mobilenetv2_pruned30.onnx
"""

import argparse

import torch

from edgeopt.data import fixed_subset, load_cifar10
from edgeopt.export import build_mobilenetv2, export_model_onnx
from edgeopt.prune import count_params, prune_mobilenetv2_model, save_pruned
from edgeopt.train import finetune


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="mobilenetv2_cifar10.pth", help="Fine-tuned FP32 state_dict")
    parser.add_argument("--amount", type=float, default=0.3, help="Fraction of channels to remove per layer")
    parser.add_argument("--epochs", type=int, default=1, help="Fine-tuning epochs after pruning")
    parser.add_argument("--train-subset", type=int, default=5000, help="Fine-tune on a fixed subset of N training images")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1, help="torch.set_num_threads")
    parser.add_argument("--output", default="mobilenetv2_pruned_finetuned.onnx")
    parser.add_argument("--save", default=None, help="Also save the pruned PyTorch module (.pt)")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)

    model = build_mobilenetv2(10, args.weights)
    before = count_params(model)
    prune_mobilenetv2_model(model, amount=args.amount)
    print(f"params {before:,} -> {count_params(model):,}")

    if args.epochs > 0:
        trainset = fixed_subset(load_cifar10(train=True, augment=True), args.train_subset, seed=1)
        finetune(model, trainset, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    if args.save:
        save_pruned(model, args.save)
    export_model_onnx(model, args.output)
    print(f"Exported pruned model to {args.output}")


if __name__ == "__main__":
    main()
