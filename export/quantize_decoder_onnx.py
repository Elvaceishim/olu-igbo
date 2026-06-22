import os
import onnx
import onnxslim
from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT = "export/onnx/whisper_decoder.onnx"
SLIM = "export/onnx/whisper_decoder_slim.onnx"
OUTPUT = "export/onnx/whisper_decoder_int8.onnx"

print("Slimming decoder...")
model = onnx.load(INPUT)
slimmed = onnxslim.slim(model)
onnx.save(slimmed, SLIM)
print(f"Slimmed size: {os.path.getsize(SLIM) / 1e6:.1f} MB")

print("Quantising decoder to INT8 (MatMul only)...")
# matmul only keeps the embeddings and lm-head in float
quantize_dynamic(
    SLIM,
    OUTPUT,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul"],
)
print(f"INT8 size: {os.path.getsize(OUTPUT) / 1e6:.1f} MB")
