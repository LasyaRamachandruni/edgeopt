import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_dynamic(input_path, output_path, quant_type="int8"):
    print(f"Quantizing (dynamic) {input_path} to {output_path} as {quant_type.upper()}...")
    if quant_type.lower() == "int8":
        # Use QUInt8 instead of QInt8 to avoid ConvInteger issues
        quantize_dynamic(
            input_path, output_path, 
            weight_type=QuantType.QUInt8  # Changed from QInt8
        )
    elif quant_type.lower() == "fp16":
        quantize_dynamic(
            input_path, output_path, 
            weight_type=QuantType.QFloat16
        )
    else:
        raise ValueError("Only 'int8' and 'fp16' quantization are supported")
    print("Dynamic quantization done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--output", default="model_quant.onnx", help="Output quantized model path")
    parser.add_argument("--quant-type", default="int8", choices=["int8", "fp16"], help="Quantization type")
    args = parser.parse_args()

    quantize_onnx_dynamic(args.input, args.output, quant_type=args.quant_type)
