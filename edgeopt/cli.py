import argparse

NUM_CLASSES = 10  # the whole workflow is MobileNetV2 fine-tuned on CIFAR-10


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export, prune, quantize and benchmark MobileNetV2 (CIFAR-10) with ONNX Runtime")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("export", help="Export fine-tuned PyTorch weights to ONNX")
    p.add_argument("--weights", "-w", default="mobilenetv2_cifar10.pth")
    p.add_argument("--output", "-o", default="mobilenetv2_cifar10.onnx")
    p.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    p.add_argument("--verify", action="store_true")

    p = sub.add_parser("prune", help="Structurally prune channels and export to ONNX")
    p.add_argument("--weights", "-w", default=None, help="Fine-tuned state_dict (.pth); random init if omitted")
    p.add_argument("--amount", type=float, default=0.3, help="Fraction of channels to remove per layer")
    p.add_argument("--output", "-o", default="mobilenetv2_pruned.onnx")
    p.add_argument("--save", default=None, help="Also save the pruned PyTorch module (.pt) for fine-tuning")
    p.add_argument("--num-classes", type=int, default=NUM_CLASSES)

    p = sub.add_parser("quantize", help="Quantize an ONNX model")
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", default="model_int8.onnx")
    p.add_argument("--mode", default="static", choices=["static", "dynamic", "fp16"],
                   help="static: INT8 QDQ with calibration (use for conv nets); dynamic: INT8 weights only; fp16")
    p.add_argument("--calib-size", type=int, default=300, help="CIFAR-10 training images used for calibration")
    p.add_argument("--calib-data", default="cifar10", choices=["cifar10", "synthetic"],
                   help="synthetic = random noise; only valid for size/latency checks, not accuracy")
    p.add_argument("--no-per-channel", action="store_true")

    p = sub.add_parser("benchmark", help="Measure batch latency and throughput of an ONNX model")
    p.add_argument("--model", required=True)
    p.add_argument("--N", type=int, default=200, help="Timed runs")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (0 = runtime default)")

    p = sub.add_parser("evaluate", help="Top-1 accuracy of an ONNX model on CIFAR-10 test")
    p.add_argument("--model", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--subset", type=int, default=None, help="Fixed seeded subset of N test images")

    p = sub.add_parser("report", help="Benchmark/evaluate several models, write results.json + results.md")
    p.add_argument("--models", nargs="+", required=True, help="NAME=path.onnx entries, in table order")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3, help="Benchmark each model this many times (interleaved), report the median")
    p.add_argument("--eval-subset", type=int, default=None, help="Evaluate on N test images instead of all 10k")
    p.add_argument("--no-accuracy", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "export":
        from .export import export_finetuned_to_onnx, verify_onnx_model

        export_finetuned_to_onnx(args.weights, args.output, num_classes=args.num_classes)
        if args.verify:
            verify_onnx_model(args.output)
    elif args.command == "prune":
        from .prune import export_pruned_model_onnx

        export_pruned_model_onnx(args.amount, args.weights, args.output, num_classes=args.num_classes, save_path=args.save)
    elif args.command == "quantize":
        from . import quantize as q

        if args.mode == "static":
            calib = (q.cifar10_calibration_batches(args.calib_size) if args.calib_data == "cifar10"
                     else q.synthetic_calibration_batches(args.calib_size))
            q.quantize_onnx_static(args.input, args.output, calib, per_channel=not args.no_per_channel)
        else:
            q.quantize_onnx_dynamic(args.input, args.output, quant_type="int8" if args.mode == "dynamic" else "fp16")
    elif args.command == "benchmark":
        from .benchmark import benchmark_onnx_model

        benchmark_onnx_model(args.model, N=args.N, batch_size=args.batch_size, warmup=args.warmup, threads=args.threads)
    elif args.command == "evaluate":
        from .eval import eval_onnx_classifier

        eval_onnx_classifier(args.model, batch_size=args.batch_size, subset=args.subset)
    elif args.command == "report":
        from .report import collect, parse_model_args, write_report

        rows, meta = collect(parse_model_args(args.models), N=args.N, warmup=args.warmup, threads=args.threads,
                             repeats=args.repeats, eval_subset=args.eval_subset, accuracy=not args.no_accuracy)
        write_report(rows, meta, args.out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
