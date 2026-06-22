import torch
import numpy as np
import onnxruntime as ort
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

print("Loading merged PyTorch model...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
base = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small", torch_dtype=torch.float32,
)
model = PeftModel.from_pretrained(base, "theelvace/whisper-small-igbo")
model = model.merge_and_unload()
model.eval()

import os
ENC_ONNX = os.environ.get("ENC_ONNX", "export/onnx/whisper_encoder_int8.onnx")
DEC_ONNX = os.environ.get("DEC_ONNX", "export/onnx/whisper_decoder_int8.onnx")
print(f"Encoder: {ENC_ONNX}\nDecoder: {DEC_ONNX}")
enc_session = ort.InferenceSession(ENC_ONNX)
dec_session = ort.InferenceSession(DEC_ONNX)

sr = 16000
audio = np.sin(2 * np.pi * 220 * np.linspace(0, 3.0, 48000)).astype(np.float32)
feats = processor.feature_extractor(
    audio, sampling_rate=sr, return_tensors="pt",
).input_features

feats_np = feats.numpy().astype(np.float32)

# sot, lang, transcribe, no-timestamps
prefix = [50258, 50325, 50359, 50363]
EOS = processor.tokenizer.eos_token_id

# manual greedy: generate() auto-detects language, and igbo isn't a whisper language
print("Building PyTorch greedy reference...")
ref_list = list(prefix)
for _ in range(20):
    with torch.no_grad():
        logits = model(
            input_features=feats,
            decoder_input_ids=torch.tensor([ref_list], dtype=torch.long),
        ).logits
    nxt = int(logits[0, -1].argmax())
    ref_list.append(nxt)
    if nxt == EOS:
        break

print(f"Reference token ids: {ref_list}")
print(f"Decoded: {processor.tokenizer.decode(ref_list, skip_special_tokens=True)}")

print("\nRunning ONNX parity check...")
enc_out = enc_session.run(None, {"input_features": feats_np})[0]

input_ids_np = np.array([prefix], dtype=np.int64)

mismatches = 0
for step, ref_token in enumerate(ref_list[len(prefix):], start=1):
    logits = dec_session.run(None, {
        "input_ids": input_ids_np,
        "encoder_hidden_states": enc_out,
    })[0]
    onnx_token = int(np.argmax(logits[0, -1, :]))
    match = "OK" if onnx_token == ref_token else "X"
    if onnx_token != ref_token:
        mismatches += 1
    print(f"  step {step:2d}: ref={ref_token:6d} onnx={onnx_token:6d} {match}")
    input_ids_np = np.concatenate(
        [input_ids_np, np.array([[ref_token]], dtype=np.int64)], axis=1
    )
    if ref_token == processor.tokenizer.eos_token_id:
        break

print(f"\nParity: {step - mismatches}/{step} tokens match")
if mismatches == 0:
    print("ONNX export is faithful to PyTorch model.")
else:
    print(f"WARNING: {mismatches} mismatches — quantisation drift.")
