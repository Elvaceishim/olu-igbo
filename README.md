# Olu Igbo

_"Olu Igbo" — Igbo Voice._

An offline, on-device speech recognition system for the Igbo language, built for the Arm Create: AI Optimization Challenge 2026, Mobile AI track.

## Why I built this

I'm a native Igbo speaker, and there's no real on-device speech-to-text for my language. Not on phones, not anywhere that doesn't ship your voice off to the cloud first. With 35+ million Igbo speakers and a language that's officially endangered according to UNESCO, that gap felt worth closing — not with a cloud API wrapper, but with a model that actually runs on the kind of phone most Igbo speakers carry: mid-range Android, no NPU, no GPU acceleration, just a CPU and ONNX Runtime.

This project fine-tunes Whisper Small for Igbo, exports it as a three-stage ONNX pipeline, and runs it fully on-device on a Redmi Note 10 (Snapdragon 678) — a five-year-old mid-tier chip. This device is deliberately chosen because if it works there, it works on most phones in the hands of the people who'd actually use it.

## Results

| Metric                                              | Value                                                           |
| --------------------------------------------------- | --------------------------------------------------------------- |
| **Word Error Rate (FLEURS Igbo test, 969 samples)** | **41.95%** (down from 68.95% baseline)                          |
| Model size on disk (on-device)                      | 319 MB (93 MB encoder + 54 MB cross-attention + 172 MB decoder) |
| Decoder throughput                                  | ~9 tokens/second (Snapdragon 678, CPU-only)                     |
| End-to-end latency, short utterance                 | ~6–8 seconds                                                    |

These are real numbers, measured directly on-device with `System.nanoTime()` instrumentation around each inference stage. Not estimated.

**How WER got here:** 68.95% (zero-shot baseline) → 62.45% (LoRA on FLEURS + Common Voice) → 48.58% (warm-start full fine-tune) → **41.95%** (+ ~25k utterances of NaijaVoices read speech). The deployed model is [`theelvace/whisper-small-igbo-25k`](https://huggingface.co/theelvace/whisper-small-igbo-25k). Full story in [Training methodology](#training-methodology).

## Architecture

Whisper's standard encoder-decoder architecture doesn't export cleanly to a single ONNX graph for streaming, on-device generation — the decoder needs to attend to fixed encoder output across many autoregressive steps, and naively re-running the full encoder-decoder stack per token is far too slow for a mobile CPU. I split inference into three stages:

```
Audio → Mel spectrogram → [Encoder] → [Cross-attention init] → [KV-cache decoder loop] → Text
```

1. **Encoder** (93 MB, INT8): converts the 80×3000 mel spectrogram into 1500×768 hidden states. Runs once per utterance.
2. **Cross-attention initializer** (54 MB, kept FP32 deliberately — more on why below): pre-computes the cross-attention key/value cache from the encoder output. Also runs once per utterance.
3. **KV-cache decoder** (172 MB, INT8): greedy-decodes one token at a time, reusing both the pre-computed cross-attention cache and a growing self-attention cache, so each step only processes a single new token rather than the whole sequence so far. Because the cross-attention K/V are constant across steps, the decoder outputs only the self-attention cache and the app reuses the cross tensors — a change that cut per-token time ~10× (see On-device benchmarks).

Mel spectrogram extraction runs on-device in Kotlin (`computeLogMel` in `MainActivity.kt`). Getting it numerically identical to Whisper's reference was the subtlest correctness problem in the project: a first from-scratch implementation drifted enough to derail the decoder's cross-attention, so I initially offloaded it to a local server running the same `transformers` feature extractor the model was trained with. I later traced the drift to three specific mismatches — using the magnitude spectrum instead of power, framing without the centered/reflect-padded STFT (which also produced the wrong frame count), and a triangular filterbank missing Slaney mel-scale normalization — and fixed all three. The corrected Kotlin mel now matches the `transformers` extractor to ~1e-5 (verified in `mel_parity.py`), so the entire pipeline runs on-device with nothing offloaded.

### A bug worth describing

While re-exporting the decoder after a later training run, I hit a failure mode where the model would generate plausible-looking but completely wrong tokens — not garbage, just _wrong_, which made it far harder to spot than an obvious crash. After systematically comparing PyTorch's layer-by-layer outputs against the ONNX export's outputs (binary search through the 12 decoder layers), I found it: my decoder wrapper returned `tuple(all_keys) + tuple(all_values)`, but I'd declared the ONNX output names as `[k0, v0, k1, v1, ...]` — interleaved per layer. ONNX zips declared output names to the returned tuple positionally, with no validation. Every name from that point on was silently off by one layer. The fix was building the output tuple in the same interleaved order as the declared names. I'm including this because it's the kind of bug that produces numerically plausible, not obviously broken output — exactly the kind that's easy to miss and expensive to debug, and exactly the kind of Arm-platform/ONNX-export correctness work this track is meant to surface.

## Training methodology

**Base setup:** LoRA fine-tuning (r=32, alpha=64, targeting `q_proj`/`v_proj`) on Whisper Small, using `<|yo|>` (Yoruba) as a language-token proxy since Igbo isn't in Whisper's native 99 languages.

**Data:** Started with FLEURS Igbo (2,839 train examples). Added Common Voice Igbo via the `benjaminogbonna/nigerian_common_voice_dataset` HuggingFace mirror (4,571 train examples), bringing the combined training set to 7,410 examples — a 2.6x increase. This combination took WER from 68.95% → **62.45%**, verified on the full FLEURS test set after every training run, never trusted from validation loss alone.

**What I tried that didn't work, and why that's worth knowing:** I sourced IgboSynCorp — a 40-hour annotated Igbo speech corpus from the University of Ibadan and Afe-Babalola University (Lacuna Fund-funded, hosted on Harvard Dataverse), built from oral narrative recordings across five Southeast Nigerian states. I wrote an ELAN (`.eaf`) parser using `pympi` to extract 2,962 clean, timestamp-aligned speech segments from the raw recordings — a genuinely reusable pipeline for anyone working with linguistic ELAN-annotated audio corpora. Merging this into training, even with FLEURS oversampled 2x to counteract domain dilution, consistently _regressed_ FLEURS test WER (63.99% and 63.54% in two separate trials) rather than improving it. My read: IgboSynCorp's oral-narrative recording style is acoustically and stylistically distant enough from FLEURS' read-speech style that training on it pulls the model away from the specific distribution it's evaluated against, even though the data itself is clean and the extraction pipeline worked correctly. I kept the verified 62.45% model rather than ship a result that looked better on training metrics but tested worse. The IgboSynCorp extraction code is included in this repo since the corpus itself is a real resource for future Igbo NLP work, even though it didn't help this specific benchmark.

**Pushing further — full fine-tune + NaijaVoices:** With the 62.45% LoRA baseline established, I switched to a warm-start full fine-tune — merge the adapter into the base model, then fine-tune the whole network. The inference model stays the same size (still whisper-small, on-device-safe), but a full network has far more capacity than a 3.5M-parameter adapter. With SpecAugment added and checkpoints selected on FLEURS _validation_ (not test, to avoid fitting the reported number), this reached **48.58% on the FLEURS test set**. Then I streamed in **NaijaVoices** ([`naijavoices/naijavoices-dataset`](https://huggingface.co/datasets/naijavoices/naijavoices-dataset)) — a ~600-hour Igbo read-speech corpus. 10k utterances took it to 42.75% validation; scaling to **~25k utterances reached 40.02% validation / 41.95% on the held-out FLEURS test set** — the deployed model. Unlike IgboSynCorp, NaijaVoices is domain-compatible with FLEURS and helped from the first epoch. (A follow-up pass that heavily oversampled FLEURS to chase the last two points just overfit at 40.62% validation, so I kept the 25k model.) Training code: `training/train_full_finetune.py`.

**Hard rules I learned:**

- Never set `forced_decoder_ids` during training. Set it only at inference. Setting it during training corrupted an entire run (115% WER) before I caught it.
- Always verify WER on the full FLEURS test set before pushing any model. val_loss is not a reliable proxy.
- ONNX decoder output tensors must be interleaved per-layer, not grouped by type, or the export silently mislabels outputs.

## On-device benchmarks

Measured on a Redmi Note 10 (Snapdragon 678, no NPU/GPU delegation — CPU-only ONNX Runtime inference), instrumented directly in the app with `System.nanoTime()`:

| Stage                                  | Latency     |
| -------------------------------------- | ----------- |
| Mel extraction (on-device, Kotlin DFT) | ~675–715 ms |
| Encoder inference                      | ~3.8–4.1 s  |
| Cross-attention initialization         | ~0.9 s      |
| Decoder, per token                     | ~99 ms      |
| **Total, short utterance**             | **~6–8 s**  |

The decoder runs at **~9 tokens/second**. It was ~10× slower until I fixed a memory bug: the app had been rebuilding the ~110 MB of cross-attention input tensors on _every single token_. Since those K/V are identical across decode steps, building them once and reusing them both eliminated intermittent out-of-memory crashes and cut per-token time from ~1 s to ~99 ms. The encoder (~3.9 s, fixed) is now the dominant cost.

**Model size:** 319 MB total on-device (93 MB encoder + 54 MB cross-attention + 172 MB decoder), down from Whisper Small's ~970 MB unquantized checkpoint — roughly a 3x reduction. The cross-attention component is kept FP32 deliberately: INT8 quantization there introduced just enough numerical drift to derail decoder attention entirely, even though the same quantization was safe for the encoder and decoder. Documented as a real, measured tradeoff, not an oversight.

## Setup instructions

### Requirements

- Android device, API 26+ (tested on Redmi Note 10, Snapdragon 678)
- Python 3.12 with `torch`, `transformers`, `peft`, `onnxruntime` (only needed to reproduce training/export — not to run the app)

### Running the app

1. Clone this repo
2. Build and install the Android app via Android Studio
3. Hold the record button, speak Igbo, release

Everything runs on-device. No server or network connection required.

### Reproducing the model

Training and evaluation code is in `/training`, and ONNX export/quantization code is in `/export`. The deployed model is [`theelvace/whisper-small-igbo-25k`](https://huggingface.co/theelvace/whisper-small-igbo-25k) (full fine-tune); the original 62.45% LoRA adapter is at [`theelvace/whisper-small-igbo`](https://huggingface.co/theelvace/whisper-small-igbo).

> **Export note:** the ONNX export scripts must run on the pinned stack (`torch==2.2.2`, `transformers==4.46.3`). Newer versions switch `torch.onnx` to a different exporter (writes weights as external data) and reject the legacy KV-cache format the decoder export relies on. `export/test_parity_kvcache.py` verifies the exported INT8 pipeline reproduces PyTorch token-for-token before deployment.

## Limitations, honestly stated

- **41.95% WER is a real number, not a polished demo statistic.** Short, clear utterances transcribe well. Longer or more complex sentences show the model's actual error rate. I'd rather report this accurately than imply more than the model delivers.
- **Live microphone audio is harder than clean test-set audio.** The 41.95% is measured on FLEURS studio recordings; real phone-mic input (background noise, natural pacing) performs worse. On long or complex live sentences the model can fail to emit an end-of-sentence token and ramble until the token cap — short, clear utterances are the reliable sweet spot.
- **The encoder (~3.9 s, fixed) now dominates latency.** After the cross-tensor fix the decoder is fast (~9 tok/s), so the next optimization is NNAPI or GPU/DSP delegation to offload the encoder — which this submission doesn't yet use.

## What's reusable here, beyond this specific model

- The ELAN/`.eaf` parsing and audio-segmentation pipeline (`/igbosyncorp_extraction`) works for any ELAN-annotated linguistic corpus, not just Igbo
- The three-stage ONNX export pattern (encoder / cross-attention init / KV-cache decoder) is a general technique for getting any encoder-decoder Whisper-family model running efficiently on mobile
- The fine-tuned models are public on HuggingFace for anyone building Igbo language tools

## License

The **code** in this repository is MIT licensed.

**Model weights and training data** carry the licenses of their sources — honor these if you reuse the fine-tuned model:

- Base model: OpenAI Whisper (MIT)
- FLEURS (CC-BY-4.0), Common Voice Igbo (CC0)
- NaijaVoices (CC-BY-NC-SA-4.0) — **non-commercial, share-alike**. The deployed model (`theelvace/whisper-small-igbo-25k`) is fine-tuned on NaijaVoices, so it inherits that non-commercial/share-alike restriction. The original 62.45% model (`theelvace/whisper-small-igbo`, FLEURS + Common Voice only) is not subject to it.
