"""
Evaluate a trained Igbo Whisper adapter on the full FLEURS Igbo test set.

This is the verification step this project treats as non-negotiable:
training/validation loss is not a reliable proxy for word error rate,
and every checkpoint in this project's history was confirmed (or
rejected) by running this script against the full 969-sample FLEURS
test set before being published — never by validation loss alone.

Reports two WER numbers:
  - raw:        predictions and references only lowercased
  - normalized: both passed through normalize_igbo (the same normalization
                used on the training labels). This is the fair comparison,
                since the model is trained to produce normalized text;
                scoring against unnormalized references inflates WER with
                punctuation/symbol mismatches the model was taught to drop.

Usage:
    python evaluate_wer.py --adapter theelvace/whisper-small-igbo
    python evaluate_wer.py --adapter ./igbo_lora_output --n-samples 50  # quick check
    python evaluate_wer.py --adapter theelvace/whisper-small-igbo --num-beams 5
"""

import argparse
import re
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

YORUBA_TOKEN_ID = 50325
TRANSCRIBE_TOKEN_ID = 50359
NO_TIMESTAMPS_TOKEN_ID = 50363


def normalize_igbo(text: str) -> str:
    """Same normalization applied to training labels — strips punctuation/symbols,
    normalizes quotes, keeps Igbo diacritics, lowercases."""
    text = text.strip()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(
        r"[^\w\s\-'àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿ"
        r"ạẹịọụĄąĘęỊịỌọỤụÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖÙÚÛÜÝÞŸ"
        r"ṅṄǹǸ]",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="openai/whisper-small")
    parser.add_argument("--adapter", required=True, help="HF repo or local path of the LoRA adapter to evaluate")
    parser.add_argument("--n-samples", type=int, default=None, help="Limit to N test samples for a quick check")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--num-beams", type=int, default=1, help="Beam width; >1 enables beam search")
    parser.add_argument("--print-samples", type=int, default=10, help="How many (pred, ref) pairs to print")
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

    preds, refs = [], []

    print(f"Evaluating {len(dataset)} FLEURS test samples (num_beams={args.num_beams})...")
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
                num_beams=args.num_beams,
            )
        preds.append(processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True))
        refs.append(sample["transcription"])

        if idx % 100 == 0:
            print(f"  {idx}/{len(dataset)}")

    raw_wer = wer_metric.compute(
        predictions=[p.lower() for p in preds], references=[r.lower() for r in refs]
    )
    norm_wer = wer_metric.compute(
        predictions=[normalize_igbo(p) for p in preds], references=[normalize_igbo(r) for r in refs]
    )

    print(f"\n--- {len(dataset)} FLEURS test samples ---")
    print(f"Raw WER (lowercase only):   {raw_wer * 100:.2f}%")
    print(f"Normalized WER (fair):      {norm_wer * 100:.2f}%")

    if args.print_samples:
        print(f"\n--- first {args.print_samples} (normalized) pairs ---")
        for p, r in list(zip(preds, refs))[: args.print_samples]:
            print(f"  REF : {normalize_igbo(r)}")
            print(f"  PRED: {normalize_igbo(p)}\n")


if __name__ == "__main__":
    main()
