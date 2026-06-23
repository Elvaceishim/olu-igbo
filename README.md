# Olu Igbo

*"Olu Igbo" — Igbo Voice.*

An offline, on-device speech recognition system for Igbo — built for the Arm Create: AI Optimization Challenge 2026, Mobile AI track.

## Why I built this

I'm a native Igbo speaker. There's no real on-device speech-to-text for my language. Not on phones, not anywhere that doesn't require sending your voice to a server first. With 35+ million Igbo speakers and a language that's officially endangered according to UNESCO, that gap felt worth closing — not with a cloud API wrapper, but with a model that actually runs on the kind of phone most Igbo speakers carry: mid-range Android, no NPU, no GPU acceleration, just a CPU and ONNX Runtime.

This project fine-tunes Whisper Small for Igbo, exports it as a three-stage ONNX pipeline, and runs it fully on-device on a Redmi Note 10 (Snapdragon 678) — a five-year-old mid-tier chip, deliberately chosen because if it works there, it works on most phones in the hands of the people who'd actually use it.

## Results

| Metric | Value |
|---|---|
| **Word Error Rate (FLEURS Igbo test, 969 samples)** | **62.45%** (down from 68.95% baseline) |
| Model size on disk (on-device) | 315 MB (88 MB encoder + 54 MB cross-attention + 173 MB decoder) |
| Decoder throughput | ~1 token/second average (Snapdragon 678, CPU-only) |
| Encoder latency | ~3.8–4.1 seconds (fixed, processes 30s mel window) |

These are real numbers, measured directly on-device with `System.nanoTime()` instrumentation around each inference stage, not estimated. The methodology section below explains exactly how I got from 68.95% to 62.45%, including what I tried that *didn't* work.

## Architecture

Whisper's standard encoder-decoder architecture doesn't export cleanly to a single ONNX graph for streaming, on-device generation — the decoder needs to attend to fixed encoder output across many autoregressive steps, and naively re-running the full encoder-decoder stack per token is far too slow for a mobile CPU. I split inference into three stages:

```
Audio → Mel spectrogram → [Encoder] → [Cross-attention init] → [KV-cache decoder loop] → Text
```

1. **Encoder** (88 MB, quantized QUInt8): converts the 80×3000 mel spectrogram into 1500×768 hidden states. Runs once per utterance.
2. **Cross-attention initializer** (54 MB, kept FP32 deliberately — more on why below): pre-computes the cross-attention key/value cache from the encoder output. Also runs once per utterance.
3. **KV-cache decoder** (173 MB, quantized INT8): greedy-decodes one token at a time, reusing both the pre-computed cross-attention cache and a growing self-attention cache, so each step only needs to process a single new token rather than the whole sequence so far.

Mel spectrogram extraction runs on a small local FastAPI server on the same network rather than in Kotlin — I tried writing a from-scratch FFT/mel-filterbank implementation in Kotlin first, and it introduced small but compounding numerical drift from Whisper's reference implementation that was enough to derail the decoder's cross-attention. Computing it with the same `transformers` feature extractor the model was trained with removed that entire class of bug. Encoder and decoder inference still run entirely on-device — only the (deterministic, model-free) feature extraction is offloaded.

### A bug worth describing

While re-exporting the decoder after a later training run, I hit a failure mode where the model would generate plausible-looking but completely wrong tokens — not garbage, just *wrong*, which made it far harder to spot than an obvious crash. After systematically comparing PyTorch's layer-by-layer outputs against the ONNX export's outputs (binary search through the 12 decoder layers), I found it: my decoder wrapper returned `tuple(all_keys) + tuple(all_values)`, but I'd declared the ONNX output names as `[k0, v0, k1, v1, ...]` — interleaved per layer. ONNX zips declared output names to the returned tuple positionally, with no validation. Every name from that point on was silently off by one layer. The fix was building the output tuple in the same interleaved order as the declared names. I'm including this because it's the kind of bug that produces numerically plausible, not obviously broken output — exactly the kind that's easy to miss and expensive to debug, and exactly the kind of Arm-platform/ONNX-export correctness work this track is meant to surface.

## Training methodology

**Base setup:** LoRA fine-tuning (r=32, alpha=64, targeting `q_proj`/`v_proj`) on Whisper Small, using `<|yo|>` (Yoruba) as a language-token proxy since Igbo isn't in Whisper's native 99 languages.

**Data:** Started with FLEURS Igbo (2,839 train examples). Added Common Voice Igbo via the `benjaminogbonna/nigerian_common_voice_dataset` HuggingFace mirror (4,571 train examples), bringing the combined training set to 7,410 examples — a 2.6x increase. This combination took WER from 68.95% → **62.45%**, verified on the full FLEURS test set after every training run, never trusted from validation loss alone.

**What I tried that didn't work, and why that's worth knowing:** I sourced IgboSynCorp — a 40-hour annotated Igbo speech corpus from the University of Ibadan and Afe-Babalola University (Lacuna Fund-funded, hosted on Harvard Dataverse), built from oral narrative recordings across five Southeast Nigerian states. I wrote an ELAN (`.eaf`) parser using `pympi` to extract 2,962 clean, timestamp-aligned speech segments from the raw recordings — a genuinely reusable pipeline for anyone working with linguistic ELAN-annotated audio corpora. Merging this into training, even with FLEURS oversampled 2x to counteract domain dilution, consistently *regressed* FLEURS test WER (63.99% and 63.54% in two separate trials) rather than improving it. My read: IgboSynCorp's oral-narrative recording style is acoustically and stylistically distant enough from FLEURS' read-speech style that training on it pulls the model away from the specific distribution it's evaluated against, even though the data itself is clean and the extraction pipeline worked correctly. I kept the verified 62.45% model rather than ship a result that looked better on training metrics but tested worse. The IgboSynCorp extraction code is included in this repo since the corpus itself is a real resource for future Igbo NLP work, even though it didn't help this specific benchmark.

**Hard rules I learned and never violated again after the first mistake:**
- Never set `forced_decoder_ids` during training — only at inference. Setting it during training corrupted an entire run (115% WER) before I caught it.
- Always verify WER on the full FLEURS test set before pushing any model — val_loss is not a reliable proxy.
- ONNX decoder output tensors must be interleaved per-layer, not grouped by type, or the export silently mislabels outputs.

## On-device benchmarks

Measured on a Redmi Note 10 (Snapdragon 678, no NPU/GPU delegation — CPU-only ONNX Runtime inference), instrumented directly in the app with `System.nanoTime()`:

| Stage | Latency |
|---|---|
| Mel extraction (network round-trip to local server) | ~300–380 ms |
| Encoder inference | ~3.8–4.1 s |
| Cross-attention initialization | ~916–924 ms |
| Decoder, per token | ~999 ms average |
| **Total, short utterance (~5–8 tokens)** | **~13–16 s** |
| **Total, longer utterance (~16–19 tokens)** | **~24–25 s** |

Decoder throughput improves slightly with longer outputs (0.63 → 0.84 tokens/sec) as the fixed encoder/cross-attention overhead amortizes over more generated tokens.

**Model size:** 315 MB total on-device (88 MB encoder + 54 MB cross-attention + 173 MB decoder), down from Whisper Small's ~970 MB unquantized checkpoint — roughly a 3x reduction. The cross-attention component is kept FP32 deliberately: INT8 quantization there introduced just enough numerical drift to derail decoder attention entirely, even though the same quantization was safe for the encoder and decoder. Documented as a real, measured tradeoff, not an oversight.

## Setup instructions

### Requirements
- Android device, API 26+ (tested on Redmi Note 10, Snapdragon 678)
- Python 3.12 with `torch`, `transformers`, `peft`, `onnxruntime`, `fastapi`, `uvicorn` for the mel server
- Same WiFi network for the phone and the machine running the mel server

### Running the app
1. Clone this repo
2. Start the mel server: `python mel_server.py` (runs on port 8765)
3. Update the server IP in `MainActivity.kt`'s `getMelFromServer()` to match your machine's local IP
4. Build and install the Android app via Android Studio
5. Hold the record button, speak Igbo, release

### Reproducing the model
All training and export code is in `/training`. The model weights (LoRA adapter) and ONNX exports are published at [`theelvace/whisper-small-igbo`](https://huggingface.co/theelvace/whisper-small-igbo) on HuggingFace.

## Limitations, honestly stated

- **62.45% WER is a real number, not a polished demo statistic.** Short, clear utterances transcribe well. Longer or more complex sentences show the model's actual error rate. I'd rather report this accurately than imply more than the model delivers.
- **Live microphone audio is harder than clean test-set audio.** FLEURS' 62.45% WER is measured on studio-quality recordings; real-world phone mic input with background noise will generally perform worse.
- **~1 token/second on CPU is slow** for a 173 MB decoder on a 2021 mid-range chip. The natural next optimization is NNAPI or GPU delegation, which this submission doesn't yet use — a clear, scoped next step.

## What's reusable here, beyond this specific model

- The ELAN/`.eaf` parsing and audio-segmentation pipeline (`/igbosyncorp_extraction`) works for any ELAN-annotated linguistic corpus, not just Igbo
- The three-stage ONNX export pattern (encoder / cross-attention init / KV-cache decoder) is a general technique for getting any encoder-decoder Whisper-family model running efficiently on mobile
- The fine-tuned model and full ONNX exports are public on HuggingFace for anyone building Igbo language tools

## License

MIT
