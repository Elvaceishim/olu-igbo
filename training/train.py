"""
LoRA fine-tuning script for Whisper Small on Igbo speech recognition.

Trains on a merged FLEURS Igbo + Nigerian Common Voice (Igbo subset)
dataset. Igbo isn't one of Whisper's native 99 languages, so the
Yoruba language token (<|yo|>, id 50325) is used as a proxy during
both training and inference — the closest available token in Whisper's
existing vocabulary.

This reproduces the training run behind the 62.45% WER checkpoint
published at https://huggingface.co/theelvace/whisper-small-igbo
(part of the Olu Igbo project: https://github.com/Elvaceishim/olu-igbo).

Run on a single GPU (developed against a Kaggle T4). Designed to be run
in one self-contained script/cell rather than split across many — this
project's training sessions on Kaggle were frequently interrupted by
kernel restarts, and a single self-contained run with periodic
checkpointing proved far more reliable than splitting setup and training
across separate cells.

Usage:
    python train.py --base-adapter theelvace/whisper-small-igbo --epochs 2

To continue refining an existing adapter (recommended — this project's
best results came from incrementally refining a working checkpoint
rather than retraining from scratch each time), pass --base-adapter.
To train a fresh LoRA adapter from the base Whisper Small model instead,
omit it.
"""

import re
import os
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from peft import PeftModel, LoraConfig, get_peft_model

# Igbo proxy + control tokens for Whisper's decoder prefix
YORUBA_TOKEN_ID = 50325  # <|yo|> used as an Igbo language proxy
TRANSCRIBE_TOKEN_ID = 50359
NO_TIMESTAMPS_TOKEN_ID = 50363
MAX_LABEL_LEN = 448  # Whisper's max decoder sequence length


def normalize_igbo(text: str) -> str:
    """Lowercase and strip to standard orthographic Igbo, preserving diacritics."""
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


def is_valid_length(example) -> bool:
    return len(example["labels"]) <= MAX_LABEL_LEN


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def load_training_data(processor):
    """Load and prepare FLEURS Igbo + Common Voice Igbo, merged into one training set."""

    def prepare_fleurs(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="np"
        ).input_features[0]
        batch["labels"] = processor.tokenizer(normalize_igbo(batch["transcription"])).input_ids
        return batch

    def prepare_cv(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=16000, return_tensors="np"
        ).input_features[0]
        batch["labels"] = processor.tokenizer(normalize_igbo(batch["sentence"])).input_ids
        return batch

    print("Loading FLEURS Igbo...")
    fleurs = load_dataset("google/fleurs", "ig_ng")
    fleurs_train = fleurs["train"].map(prepare_fleurs, remove_columns=fleurs["train"].column_names)
    fleurs_val = fleurs["validation"].map(prepare_fleurs, remove_columns=fleurs["validation"].column_names)
    print(f"  FLEURS train: {len(fleurs_train)}, val: {len(fleurs_val)}")

    print("Loading Common Voice Igbo...")
    cv = load_dataset("benjaminogbonna/nigerian_common_voice_dataset", name="igbo")
    cv = cv.cast_column("audio", Audio(sampling_rate=16000))
    cv_train = cv["train"].map(prepare_cv, remove_columns=cv["train"].column_names)
    cv_train = cv_train.filter(is_valid_length)  # drops one known mismatched-length row
    print(f"  Common Voice train: {len(cv_train)}")

    train_data = concatenate_datasets([fleurs_train, cv_train])
    val_data = fleurs_val  # validate on FLEURS only — this matters, see note below

    print(f"\nMerged train: {len(train_data)}, FLEURS-only val: {len(val_data)}")
    return train_data, val_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="openai/whisper-small")
    parser.add_argument(
        "--base-adapter",
        default=None,
        help="HuggingFace repo of an existing LoRA adapter to continue refining. "
        "Omit to train a fresh adapter from the base model.",
    )
    parser.add_argument("--output-dir", default="./igbo_lora_output")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Which attention projections LoRA adapts. Widening this (e.g. to include "
        "k_proj and out_proj) trades more trainable capacity for slower training — "
        "worth trying if a narrow adapter has already converged and you're adding a "
        "third, more acoustically distinct data source.",
    )
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(args.base_model)
    train_data, val_data = load_training_data(processor)

    print("\nLoading model...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float32, device_map={"": 0}
    )

    if args.base_adapter:
        print(f"Continuing adapter: {args.base_adapter}")
        model = PeftModel.from_pretrained(base_model, args.base_adapter, is_trainable=True)
    else:
        print("Training a fresh LoRA adapter from the base model.")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()
    model.train()

    # IMPORTANT: never set forced_decoder_ids during training — only at inference.
    # Doing so during training corrupted an early run in this project (115% WER).
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_features = batch["input_features"].to("cuda")
            labels = batch["labels"].to("cuda")

            outputs = model.base_model.model(input_features=input_features, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            if batch_idx % 200 == 0:
                print(f"Epoch {epoch + 1} step {batch_idx}/{len(train_loader)} loss: {loss.item():.4f}")
                # Checkpoint periodically — Kaggle/Colab sessions can be interrupted mid-epoch.
                model.save_pretrained(f"{args.output_dir}_checkpoint")

        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model.base_model.model(
                    input_features=batch["input_features"].to("cuda"),
                    labels=batch["labels"].to("cuda"),
                )
                val_loss += outputs.loss.item()
        avg_val = val_loss / len(val_loader)

        print(f"\nEpoch {epoch + 1} — train: {avg_train:.4f}  FLEURS-val: {avg_val:.4f}\n")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(args.output_dir)
            print(f"Saved best checkpoint — FLEURS-val_loss: {best_val_loss:.4f}")

    print("\nTraining complete.")
    print(
        "IMPORTANT: val_loss is not a reliable proxy for WER. Always verify on the full "
        "FLEURS test set with model.generate() before treating a checkpoint as an improvement — "
        "see evaluate_wer.py."
    )


if __name__ == "__main__":
    main()
