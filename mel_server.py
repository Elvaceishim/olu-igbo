import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from transformers import WhisperProcessor

app = FastAPI()
processor = WhisperProcessor.from_pretrained("openai/whisper-small", local_files_only=True)
print("Mel server ready.")

@app.post("/mel")
async def get_mel(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    # android sends raw little-endian int16 pcm; scale to float [-1, 1)
    audio_int16 = np.frombuffer(audio_bytes, dtype="<i2")
    audio_np = audio_int16.astype(np.float32) / 32768.0

    print(f"Received audio: shape={audio_np.shape}, min={audio_np.min():.4f}, max={audio_np.max():.4f}, mean={audio_np.mean():.4f}")

    inputs = processor.feature_extractor(audio_np, sampling_rate=16000, return_tensors="np")
    mel = inputs.input_features[0]
    print(f"Mel output: shape={mel.shape}, min={mel.min():.4f}, max={mel.max():.4f}, mean={mel.mean():.4f}")

    return Response(content=mel.astype(np.float32).tobytes(), media_type="application/octet-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)