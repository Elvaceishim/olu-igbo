import os
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

HF_REPO = "theelvace/whisper-small-igbo"
BASE_MODEL = "openai/whisper-small"
OUTPUT_DIR = "export/onnx"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
D_MODEL = 768
ENC_SEQ = 1500

# Default to the full fine-tuned model; FULL_MODEL="" falls back to base + adapter.
FULL_MODEL = os.environ.get("FULL_MODEL", "theelvace/whisper-small-igbo-25k")

print("Loading model...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)
if FULL_MODEL:
    print(f"Loading full fine-tuned model: {FULL_MODEL}")
    model = WhisperForConditionalGeneration.from_pretrained(FULL_MODEL, torch_dtype=torch.float32)
else:
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, HF_REPO).merge_and_unload()
model.eval()

class WhisperDecoderWithCache(nn.Module):
    def __init__(self, decoder, proj_out):
        super().__init__()
        self.decoder = decoder
        self.proj_out = proj_out

    def forward(self, input_ids, encoder_hidden_states, past_key_values):
        out = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = self.proj_out(out.last_hidden_state)
        present = out.past_key_values
        outputs = [logits]
        # each layer_kv is (self_k, self_v, cross_k, cross_v); the cross K/V are
        # constant across steps, so output only the self caches — outputting cross
        # every step wastes ~110 MB of native memory per call and OOM-crashes phones.
        for layer_kv in present:
            outputs.append(layer_kv[0])  # present self_k
            outputs.append(layer_kv[1])  # present self_v
        return tuple(outputs)

decoder_with_cache = WhisperDecoderWithCache(
    model.model.decoder,
    model.proj_out
)
decoder_with_cache.eval()

input_ids = torch.zeros(1, 1, dtype=torch.long)

# first step: self-attn cache empty (seq=0), cross-attn already full at enc_seq
past_kv = tuple(
    (
        torch.zeros(1, NUM_HEADS, 0, HEAD_DIM),
        torch.zeros(1, NUM_HEADS, 0, HEAD_DIM),
        torch.zeros(1, NUM_HEADS, ENC_SEQ, HEAD_DIM),
        torch.zeros(1, NUM_HEADS, ENC_SEQ, HEAD_DIM),
    )
    for _ in range(NUM_LAYERS)
)

encoder_hidden = torch.zeros(1, ENC_SEQ, D_MODEL)

print("Exporting decoder with KV-cache...")

input_names = ["input_ids", "encoder_hidden_states"]
for i in range(NUM_LAYERS):
    input_names += [f"past_self_k_{i}", f"past_self_v_{i}",
                    f"past_cross_k_{i}", f"past_cross_v_{i}"]

output_names = ["logits"]
for i in range(NUM_LAYERS):
    output_names += [f"present_self_k_{i}", f"present_self_v_{i}"]

dynamic_axes = {
    "input_ids": {0: "batch", 1: "seq"},
    "encoder_hidden_states": {0: "batch"},
    "logits": {0: "batch", 1: "seq"},
}
for i in range(NUM_LAYERS):
    dynamic_axes[f"past_self_k_{i}"] = {0: "batch", 2: "past_seq"}
    dynamic_axes[f"past_self_v_{i}"] = {0: "batch", 2: "past_seq"}
    dynamic_axes[f"past_cross_k_{i}"] = {0: "batch", 2: "enc_seq"}
    dynamic_axes[f"past_cross_v_{i}"] = {0: "batch", 2: "enc_seq"}
    dynamic_axes[f"present_self_k_{i}"] = {0: "batch", 2: "present_seq"}
    dynamic_axes[f"present_self_v_{i}"] = {0: "batch", 2: "present_seq"}

flat_past = []
for layer_kv in past_kv:
    flat_past.extend(layer_kv)

torch.onnx.export(
    decoder_with_cache,
    (input_ids, encoder_hidden, past_kv),
    f"{OUTPUT_DIR}/whisper_decoder_kvcache.onnx",
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=17,
    do_constant_folding=True,
)

size_mb = os.path.getsize(f"{OUTPUT_DIR}/whisper_decoder_kvcache.onnx") / 1e6
print(f"Decoder with KV-cache: {size_mb:.1f} MB")

import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(f"{OUTPUT_DIR}/whisper_decoder_kvcache.onnx")
print("Input names:", [i.name for i in session.get_inputs()])
print("Output names:", [o.name for o in session.get_outputs()])

feeds = {
    "input_ids": np.zeros((1, 1), dtype=np.int64),
}
for i in range(NUM_LAYERS):
    feeds[f"past_self_k_{i}"] = np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32)
    feeds[f"past_self_v_{i}"] = np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32)
    feeds[f"past_cross_k_{i}"] = np.zeros((1, NUM_HEADS, ENC_SEQ, HEAD_DIM), dtype=np.float32)
    feeds[f"past_cross_v_{i}"] = np.zeros((1, NUM_HEADS, ENC_SEQ, HEAD_DIM), dtype=np.float32)

outputs = session.run(None, feeds)
print(f"Logits shape: {outputs[0].shape}")
print(f"Present self_k_0 shape: {outputs[1].shape}")
print(f"Num outputs: {len(outputs)} (expect {1 + 2 * NUM_LAYERS})")
print("KV-cache decoder export verified.")
