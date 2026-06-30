import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

HF_REPO = "theelvace/whisper-small-igbo"
BASE_MODEL = "openai/whisper-small"
OUTPUT_DIR = "export/onnx"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default to the full fine-tuned model; FULL_MODEL="" falls back to base + adapter.
FULL_MODEL = os.environ.get("FULL_MODEL", "theelvace/whisper-small-igbo-25k")

print("Loading model...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)
if FULL_MODEL:
    print(f"Loading full fine-tuned model: {FULL_MODEL}")
    model = WhisperForConditionalGeneration.from_pretrained(FULL_MODEL, torch_dtype=torch.float32)
else:
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, HF_REPO).merge_and_unload()
model.eval()

decoder = model.model.decoder
proj_out = model.proj_out

batch = 1
seq_len = 4
enc_seq = 1500
d_model = 768

dummy_input_ids        = torch.zeros(batch, seq_len, dtype=torch.long)
dummy_encoder_hidden   = torch.zeros(batch, enc_seq, d_model)

# wrap decoder + lm head so the exported graph outputs logits directly
class DecoderWithHead(torch.nn.Module):
    def __init__(self, decoder, proj_out):
        super().__init__()
        self.decoder  = decoder
        self.proj_out = proj_out

    def forward(self, input_ids, encoder_hidden_states):
        out = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        logits = self.proj_out(out.last_hidden_state)
        return logits

decoder_with_head = DecoderWithHead(decoder, proj_out)
decoder_with_head.eval()

print("Exporting decoder...")
torch.onnx.export(
    decoder_with_head,
    (dummy_input_ids, dummy_encoder_hidden),
    f"{OUTPUT_DIR}/whisper_decoder.onnx",
    input_names=["input_ids", "encoder_hidden_states"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids":              {0: "batch", 1: "seq_len"},
        "encoder_hidden_states":  {0: "batch"},
        "logits":                 {0: "batch", 1: "seq_len"},
    },
    opset_version=17,
    do_constant_folding=True,
)

size_mb = os.path.getsize(f"{OUTPUT_DIR}/whisper_decoder.onnx") / 1e6
print(f"Decoder size: {size_mb:.1f} MB")

print("Verifying decoder...")
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(f"{OUTPUT_DIR}/whisper_decoder.onnx")
dummy_ids = np.zeros((1, 4), dtype=np.int64)
dummy_enc = np.zeros((1, 1500, 768), dtype=np.float32)
outputs = session.run(None, {
    "input_ids": dummy_ids,
    "encoder_hidden_states": dummy_enc,
})
print(f"Logits shape: {outputs[0].shape}")
print("Decoder export verified.")
