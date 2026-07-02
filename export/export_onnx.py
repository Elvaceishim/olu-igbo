import dataclasses
import json
import os
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

HF_REPO = "theelvace/whisper-small-igbo"
BASE_MODEL = "openai/whisper-small"
OUTPUT_DIR = "export/onnx"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_compatible_adapter_dir(repo):
    local_dir = snapshot_download(repo)
    cfg_path = os.path.join(local_dir, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    # adapter was saved by a newer peft; drop config keys this version can't parse
    valid = {field.name for field in dataclasses.fields(LoraConfig)}
    dropped = [k for k in cfg if k not in valid]
    if dropped:
        print(f"Stripping unsupported adapter_config keys: {dropped}")
        for k in dropped:
            cfg.pop(k)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    return local_dir


# Default to the full fine-tuned model. Set FULL_MODEL="" to fall back to the
# original base + LoRA-adapter path that produced the 62.45% model.
FULL_MODEL = os.environ.get("FULL_MODEL", "theelvace/whisper-small-igbo-25k")

print("Loading model...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)

if FULL_MODEL:
    print(f"Loading full fine-tuned model: {FULL_MODEL}")
    model = WhisperForConditionalGeneration.from_pretrained(FULL_MODEL, torch_dtype=torch.float32)
else:
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, load_compatible_adapter_dir(HF_REPO)).merge_and_unload()
model.eval()
print(f"Model type: {type(model)}")

import torch.nn.functional as F


class VariableEncoder(torch.nn.Module):
    # Whisper's encoder hard-codes a 3000-frame (30s) input and adds a fixed 1500-position
    # embedding. This wrapper slices the positional embedding to the ACTUAL input length,
    # so we can encode shorter windows (e.g. 5s) and skip the wasted compute on padded
    # silence. Mathematically identical to the original at the full 3000 frames.
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features):
        enc = self.encoder
        embeds = F.gelu(enc.conv1(input_features))
        embeds = F.gelu(enc.conv2(embeds))
        embeds = embeds.permute(0, 2, 1)
        hidden = embeds + enc.embed_positions.weight[: embeds.shape[1]]
        for layer in enc.layers:
            hidden = layer(hidden, None, None)[0]
        return enc.layer_norm(hidden)


print("Exporting encoder (variable-length)...")
encoder = VariableEncoder(model.model.encoder).eval()
dummy_input = torch.zeros(1, 80, 3000)  # exported with a dynamic mel-frames axis

torch.onnx.export(
    encoder,
    dummy_input,
    f"{OUTPUT_DIR}/whisper_encoder.onnx",
    input_names=["input_features"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_features": {0: "batch_size", 2: "mel_frames"},
        "last_hidden_state": {0: "batch_size", 1: "enc_seq"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print(f"Encoder exported → {OUTPUT_DIR}/whisper_encoder.onnx")

size_mb = os.path.getsize(f"{OUTPUT_DIR}/whisper_encoder.onnx") / 1e6
print(f"Encoder size: {size_mb:.1f} MB")

import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(f"{OUTPUT_DIR}/whisper_encoder.onnx")
dummy_np = np.zeros((1, 80, 3000), dtype=np.float32)
outputs = session.run(None, {"input_features": dummy_np})
print(f"Encoder output shape: {outputs[0].shape}")
print("Encoder export verified.")
