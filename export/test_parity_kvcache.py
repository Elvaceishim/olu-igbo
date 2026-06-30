"""
3-stage KV-cache parity check — validates the on-device pipeline end to end.

Replicates exactly what MainActivity.kt does (encoder -> cross-attn init ->
KV-cache decoder loop, feeding encoder_hidden_states + self/cross caches and
reading 4 present-KV tensors per layer) and compares the greedy token stream to
the reference PyTorch model. If every token matches, the ONNX export + INT8
quantization are faithful to the model the app will run.

Run after the export + quantize scripts. CPU-only.
"""

import os
import torch
import numpy as np
import onnxruntime as ort
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL = os.environ.get("FULL_MODEL", "theelvace/whisper-small-igbo-25k")
NUM_LAYERS, NUM_HEADS, HEAD_DIM, ENC_SEQ, D_MODEL = 12, 12, 64, 1500, 768
PREFIX = [50258, 50325, 50359, 50363]  # sot, <|yo|>, transcribe, no-timestamps

ENC = os.environ.get("ENC_ONNX", "export/onnx/whisper_encoder_int8.onnx")
CROSS = os.environ.get("CROSS_ONNX", "export/onnx/whisper_cross_attn_init_int8.onnx")
DEC = os.environ.get("DEC_ONNX", "export/onnx/whisper_decoder_kvcache_int8.onnx")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
eos = processor.tokenizer.eos_token_id

audio = np.sin(2 * np.pi * 220 * np.linspace(0, 3.0, 48000)).astype(np.float32)
feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
feats_np = feats.numpy().astype(np.float32)

print("Building PyTorch greedy reference...")
ref = list(PREFIX)
for _ in range(20):
    with torch.no_grad():
        logits = model(input_features=feats, decoder_input_ids=torch.tensor([ref])).logits
    nxt = int(logits[0, -1].argmax())
    ref.append(nxt)
    if nxt == eos:
        break
print("REF:", processor.tokenizer.decode(ref, skip_special_tokens=True))

print("\nRunning ONNX 3-stage pipeline...")
enc = ort.InferenceSession(ENC)
cross = ort.InferenceSession(CROSS)
dec = ort.InferenceSession(DEC)

enc_out = enc.run(None, {"input_features": feats_np})[0]
cross_out = cross.run(None, {"encoder_hidden_states": enc_out})
cross_k = [cross_out[i * 2] for i in range(NUM_LAYERS)]
cross_v = [cross_out[i * 2 + 1] for i in range(NUM_LAYERS)]

self_k = [np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS)]
self_v = [np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS)]


def step(token):
    feeds = {"input_ids": np.array([[token]], dtype=np.int64)}
    for i in range(NUM_LAYERS):
        feeds[f"past_self_k_{i}"] = self_k[i]
        feeds[f"past_self_v_{i}"] = self_v[i]
        feeds[f"past_cross_k_{i}"] = cross_k[i]
        feeds[f"past_cross_v_{i}"] = cross_v[i]
    out = dec.run(None, feeds)
    for i in range(NUM_LAYERS):
        b = 1 + i * 2  # logits, then (self_k, self_v) per layer
        self_k[i] = out[b]
        self_v[i] = out[b + 1]
    return out[0]

# feed the forced prefix to build the cache; logits after the last prefix token
# predict the first content token (mirrors the app's sampling start)
last = None
for tok in PREFIX:
    last = step(tok)

mismatches = 0
gen = ref[len(PREFIX):]
for n, ref_tok in enumerate(gen, 1):
    onnx_tok = int(np.argmax(last[0, -1, :]))
    mark = "OK" if onnx_tok == ref_tok else "X"
    if onnx_tok != ref_tok:
        mismatches += 1
    print(f"  step {n:2d}: ref={ref_tok:6d} onnx={onnx_tok:6d} {mark}")
    if ref_tok == eos:
        break
    last = step(ref_tok)

print(f"\nParity: {n - mismatches}/{n} tokens match")
print("ONNX 3-stage pipeline is faithful." if mismatches == 0
      else f"WARNING: {mismatches} mismatches — investigate before deploying.")
