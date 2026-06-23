"""
Evaluate a trained Igbo Whisper adapter on the full FLEURS Igbo test set.

This is the verification step this project treats as non-negotiable:
training/validation loss is not a reliable proxy for word error rate,
and every checkpoint in this project's history was confirmed (or
rejected) by running this script against the full 969-sample FLEURS
test set before being published — never by validation loss alone.

Usage:
    python evaluate_wer.py --adapter theelvace/whisper-small-igbo
    python evaluate_wer.py --adapter ./igbo_lora_output --n-samples 50  # quick check
"""

import argparse
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

YORUBA_TOKEN_ID = 50325
TRANSCRIBE_TOKEN_ID = 50359
NO_TIMESTAMPS_TOKEN_ID = 50363


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="openai/whisper-small")
    parser.add_argument("--adapter", required=True, help="HF repo or local path of the LoRA adapter to evaluate")
    parser.add_argument("--n-samples", type=int, default=None, help="Limit to N test samples for a quick check")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(args.base_model)
    wer_metric = evaluate.load("wer")

    print(f"Loading {args.adapter}...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float32, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model = model.merge_and_unload()
    model.eval()
    model.config.forced_decoder_ids = None

    split = "test" if args.n_samples is None else f"test[:{args.n_samples}]"
    dataset = load_dataset("google/fleurs", "ig_ng", split=split)

    predictions, references = [], []

    print(f"Evaluating {len(dataset)} FLEURS test samples...")
    for idx, sample in enumerate(dataset):
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        feats = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to("cuda")

        with torch.no_grad():
            pred_ids = model.generate(
                feats,
                forced_decoder_ids=[
                    [1, YORUBA_TOKEN_ID],
                    [2, TRANSCRIBE_TOKEN_ID],
                    [3, NO_TIMESTAMPS_TOKEN_ID],
                ],
                max_new_tokens=args.max_new_tokens,
            )
        pred_text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        predictions.append(pred_text.lower())
        references.append(sample["transcription"].lower())

        if idx % 100 == 0:
            print(f"  {idx}/{len(dataset)}")

    wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"\nWER on {len(dataset)} FLEURS test samples: {wer * 100:.2f}%")


if __name__ == "__main__":
    main()
