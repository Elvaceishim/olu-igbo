import os
import onnx
import onnxslim
from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT = "export/onnx/whisper_encoder.onnx"
SLIM = "export/onnx/whisper_encoder_slim.onnx"
OUTPUT = "export/onnx/whisper_encoder_int8.onnx"

print("Slimming model...")
model = onnx.load(INPUT)
slimmed = onnxslim.slim(model)
onnx.save(slimmed, SLIM)
size_slim = os.path.getsize(SLIM) / 1e6
print(f"Slimmed size: {size_slim:.1f} MB")

print("Quantising to INT8...")
# matmul only: quantizing the conv stem emits ConvInteger, which the cpu EP can't run
quantize_dynamic(
    SLIM,
    OUTPUT,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul"],
)
size_int8 = os.path.getsize(OUTPUT) / 1e6
print(f"INT8 size: {size_int8:.1f} MB")

print("Verifying INT8 model...")
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(OUTPUT)
dummy = np.zeros((1, 80, 3000), dtype=np.float32)
outputs = session.run(None, {"input_features": dummy})
print(f"Output shape: {outputs[0].shape}")
print("INT8 export verified.")
