import argparse
from .export import export_pytorch_to_onnx, verify_onnx_model
from .quantize import quantize_onnx_dynamic
from .prune import export_pruned_model_onnx
from .benchmark import benchmark_onnx_model
from .eval import eval_onnx_classifier




def main():
    parser = argparse.ArgumentParser(description="Edge-Optimized ONNX Model Converter & Benchmark")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Quantize subcommand
    quant_parser = subparsers.add_parser('quantize', help='Quantize an ONNX model')
    quant_parser.add_argument('--input', '-i', required=True, help='Input ONNX file')
    quant_parser.add_argument('--output', '-o', default='model_quant.onnx', help='Output ONNX file')
    quant_parser.add_argument('--quant-type', default='int8', choices=['int8', 'fp16'], help='Quantization type')

    # Prune subcommand
    prune_parser = subparsers.add_parser('prune', help='Prune a PyTorch model and export to ONNX')
    prune_parser.add_argument('--amount', type=float, default=0.3, help='Fraction of channels to prune, e.g. 0.3')
    prune_parser.add_argument('--output', '-o', default='mobilenetv2_pruned.onnx', help='Output ONNX file')

    # Benchmark subcommand
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark an ONNX model')
    bench_parser.add_argument('--model', required=True, help='ONNX file to benchmark')
    bench_parser.add_argument('--N', type=int, default=200, help='Number of runs')
    bench_parser.add_argument('--batch-size', type=int, default=1, help='Batch size')

    # Evaluate subcommand
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate ONNX model accuracy on CIFAR-10')
    eval_parser.add_argument('--model', required=True, help='ONNX model to evaluate')
    eval_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    eval_parser.add_argument('--max-batches', type=int, default=None, help='Max batches for quick accuracy test')

    # Export subcommand
    export_parser = subparsers.add_parser('export', help='Export PyTorch/TF model to ONNX')
    export_parser.add_argument('--output', '-o', default='mobilenetv2.onnx', 
                              help='Output ONNX file path (default: mobilenetv2.onnx)')
    export_parser.add_argument('--verify', action='store_true', 
                              help='Verify exported ONNX model')

    args = parser.parse_args()
    
    if args.command == 'export':
        print(f"Exporting PyTorch model to {args.output}...")
        export_pytorch_to_onnx(args.output)
        
        if args.verify:
            print("Verifying ONNX model...")
            verify_onnx_model(args.output)
    elif args.command == 'quantize':
        quantize_onnx_dynamic(args.input, args.output, quant_type=args.quant_type)
    elif args.command == 'prune':
        export_pruned_model_onnx(args.amount, args.output)
    elif args.command == 'benchmark':
        benchmark_onnx_model(args.model, N=args.N, batch_size=args.batch_size)
    elif args.command == 'evaluate':
        eval_onnx_classifier(args.model, batch_size=args.batch_size, max_batches=args.max_batches)


    else:
        parser.print_help()

if __name__ == "__main__":
    main()

