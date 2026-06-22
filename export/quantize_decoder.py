import os
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np

INPUT  = "export/onnx/whisper_decoder.onnx"
OUTPUT = "export/onnx/whisper_decoder_int8.onnx"

print("Quantising decoder to INT8...")
quantize_dynamic(
    INPUT,
    OUTPUT,
    weight_type=QuantType.QInt8,
)

size_mb = os.path.getsize(OUTPUT) / 1e6
print(f"Decoder INT8 size: {size_mb:.1f} MB")

print("Verifying...")
session = ort.InferenceSession(OUTPUT)
dummy_ids = np.zeros((1, 4), dtype=np.int64)
dummy_enc = np.zeros((1, 1500, 768), dtype=np.float32)
outputs = session.run(None, {
    "input_ids": dummy_ids,
    "encoder_hidden_states": dummy_enc,
})
print(f"Logits shape: {outputs[0].shape}")
print("Decoder INT8 verified.")
