import os
from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT  = "export/onnx/whisper_decoder_kvcache.onnx"
OUTPUT = "export/onnx/whisper_decoder_kvcache_int8.onnx"

print("Quantising KV-cache decoder to INT8...")
quantize_dynamic(
    INPUT,
    OUTPUT,
    weight_type=QuantType.QInt8,
)

size_mb = os.path.getsize(OUTPUT) / 1e6
print(f"KV-cache decoder INT8 size: {size_mb:.1f} MB")

import onnxruntime as ort
import numpy as np

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
ENC_SEQ = 1500
D_MODEL = 768

session = ort.InferenceSession(OUTPUT)
# cross-attn caches are full length (enc_seq); self-attn caches start empty
feeds = {}
for inp in session.get_inputs():
    name = inp.name
    if name == "input_ids":
        feeds[name] = np.zeros((1, 1), dtype=np.int64)
    elif "cross" in name:
        feeds[name] = np.zeros((1, NUM_HEADS, ENC_SEQ, HEAD_DIM), dtype=np.float32)
    else:
        feeds[name] = np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32)

outputs = session.run(None, feeds)
print(f"Logits shape: {outputs[0].shape}")
print(f"Present self_k_0 shape: {outputs[1].shape}")
print("KV-cache INT8 decoder verified.")
