import os
import torch
from torch import nn
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
FULL_MODEL = os.environ.get("FULL_MODEL", "theelvace/whisper-small-igbo-fullft")

print("Loading model...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)
if FULL_MODEL:
    print(f"Loading full fine-tuned model: {FULL_MODEL}")
    model = WhisperForConditionalGeneration.from_pretrained(FULL_MODEL, torch_dtype=torch.float32)
else:
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, HF_REPO).merge_and_unload()
model.eval()


class CrossAttnInit(nn.Module):
    # The decoder's cross-attention keys/values depend only on the (fixed) encoder
    # output, so they're computed once per utterance instead of every decode step.
    # Mirrors Whisper's own cross-attn cache: k = k_proj(enc), v = v_proj(enc),
    # reshaped to (batch, num_heads, enc_seq, head_dim). No scaling is applied to
    # keys (Whisper scales the query). Outputs are interleaved per layer
    # (cross_k_0, cross_v_0, cross_k_1, ...) to match the KV-cache decoder's
    # past_cross_k_i / past_cross_v_i inputs.
    def __init__(self, decoder):
        super().__init__()
        self.layers = decoder.layers

    def forward(self, encoder_hidden_states):
        bsz, seq, _ = encoder_hidden_states.shape
        outputs = []
        for layer in self.layers:
            attn = layer.encoder_attn
            k = attn.k_proj(encoder_hidden_states).view(bsz, seq, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            v = attn.v_proj(encoder_hidden_states).view(bsz, seq, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            outputs.append(k)
            outputs.append(v)
        return tuple(outputs)


cross_init = CrossAttnInit(model.model.decoder).eval()
dummy_enc = torch.zeros(1, ENC_SEQ, D_MODEL)

output_names = []
for i in range(NUM_LAYERS):
    output_names += [f"cross_k_{i}", f"cross_v_{i}"]

dynamic_axes = {"encoder_hidden_states": {0: "batch"}}
for name in output_names:
    dynamic_axes[name] = {0: "batch"}

print("Exporting cross-attention initializer...")
# Kept FP32 deliberately — INT8 here introduced enough drift to derail decoder
# attention (see README). ~54 MB.
torch.onnx.export(
    cross_init,
    dummy_enc,
    f"{OUTPUT_DIR}/whisper_cross_attn_init.onnx",
    input_names=["encoder_hidden_states"],
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=17,
    do_constant_folding=True,
)

size_mb = os.path.getsize(f"{OUTPUT_DIR}/whisper_cross_attn_init.onnx") / 1e6
print(f"Cross-attn init exported → {OUTPUT_DIR}/whisper_cross_attn_init.onnx ({size_mb:.1f} MB)")

import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(f"{OUTPUT_DIR}/whisper_cross_attn_init.onnx")
outputs = session.run(None, {"encoder_hidden_states": np.zeros((1, ENC_SEQ, D_MODEL), dtype=np.float32)})
print(f"Num outputs: {len(outputs)} (expect {2 * NUM_LAYERS})")
print(f"cross_k_0 shape: {outputs[0].shape}")  # (1, 12, 1500, 64)
print(f"cross_v_0 shape: {outputs[1].shape}")
print("Cross-attn init export verified.")
