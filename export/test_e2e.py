import numpy as np
import onnxruntime as ort
from transformers import WhisperProcessor
import re

print("Loading processor and ONNX sessions...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
tokenizer = processor.tokenizer

enc_session = ort.InferenceSession("export/onnx/whisper_encoder_int8.onnx")
dec_session = ort.InferenceSession("export/onnx/whisper_decoder_int8.onnx")

sr = 16000
duration = 3.0
audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration))).astype(np.float32)
print(f"Synthetic audio shape: {audio.shape}, sr: {sr}")

print("Extracting features...")
feats = processor.feature_extractor(
    audio, sampling_rate=sr, return_tensors="np", padding=True,
).input_features[0]
# encoder needs exactly 3000 frames; pad short clips, clip long ones
if feats.shape[-1] < 3000:
    feats = np.pad(feats, ((0,0),(0, 3000 - feats.shape[-1])))
else:
    feats = feats[:, :3000]
feats = feats[np.newaxis, :, :].astype(np.float32)
print(f"Features shape: {feats.shape}")

print("Running encoder...")
encoder_out = enc_session.run(None, {"input_features": feats})[0]
print(f"Encoder output shape: {encoder_out.shape}")

print("Running greedy decoder (10 steps)...")
# sot, lang, transcribe, no-timestamps
input_ids = np.array([[50258, 50325, 50359, 50363]], dtype=np.int64)
eos_token_id = tokenizer.eos_token_id

for step in range(10):
    logits = dec_session.run(None, {
        "input_ids": input_ids,
        "encoder_hidden_states": encoder_out,
    })[0]
    next_token = int(np.argmax(logits[0, -1, :]))
    input_ids = np.concatenate(
        [input_ids, np.array([[next_token]], dtype=np.int64)], axis=1
    )
    print(f"  step {step+1}: token {next_token} → {tokenizer.decode([next_token])}")
    if next_token == eos_token_id:
        break

print("\nONNX pipeline end-to-end: OK")
print(f"Encoder: {enc_session.get_providers()}")
print(f"Decoder: {dec_session.get_providers()}")
