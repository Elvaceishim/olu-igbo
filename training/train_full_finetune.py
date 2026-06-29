"""
Warm-start full fine-tune of Whisper Small for Igbo.

Strategy: the published LoRA adapter already learned real Igbo (~60% WER on
FLEURS), so instead of resetting we MERGE it into the base model and then full
fine-tune the whole network. Inference size stays identical to base whisper-small
(~244M params, on-device-safe), but we get far more capacity than a 3.5M-param
adapter. We add SpecAugment (Whisper's built-in masking) and — critically —
select the checkpoint by FLEURS-test WER, never by val_loss.

Why warm-start rather than a fresh wider adapter: a fresh adapter threw away the
converged starting point and scored 81% after one epoch. Merging the working
adapter first keeps that head start while opening up full capacity.

Run on a single GPU (developed for a Kaggle T4). Self-contained — paste as one
cell. Attach your HF_TOKEN secret if you hit rate limits. On Kaggle, install the
eval deps first: `!pip install evaluate jiwer --quiet`

To keep the adapter/export workflow instead of a full model, see the note near
the model-loading block (attach a fresh LoRA to the merged base).
"""

import os
import re
import numpy as np
import torch
import evaluate
from dataclasses import dataclass
from typing import Any
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio, concatenate_datasets, Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from peft import PeftModel

# On Kaggle, pull HF_TOKEN from secrets if it isn't already in the environment
# (needed for the gated NaijaVoices dataset).
if not os.environ.get("HF_TOKEN"):
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

# ---- config ----
BASE_MODEL = "openai/whisper-small"
WARM_START_ADAPTER = "theelvace/whisper-small-igbo"  # the ~60% model to build on
OUT_DIR = "/kaggle/working/igbo_full_ft"
PUSH_REPO = os.environ.get("PUSH_REPO", "")  # set to a NEW HF repo to auto-upload the
                                             # best checkpoint each epoch (survives a
                                             # mid-run session/quota cutoff)
EPOCHS = int(os.environ.get("EPOCHS", "3"))
BATCH_SIZE = 8
LR = float(os.environ.get("LR", "1e-5"))
EVAL_SAMPLES = None       # per-epoch eval on FLEURS validation (None = full 413)
NUM_BEAMS = 5
YO, TRANSCRIBE, NOTS = 50325, 50359, 50363  # <|yo|> proxy, transcribe, no-timestamps
MAX_LABEL_LEN = 448
NAIJA_DATASET = "naijavoices/naijavoices-dataset"
NAIJA_CONFIG = "igbo-batch-0"
NAIJA_N = int(os.environ.get("NAIJA_N", "25000"))   # NaijaVoices utterances (0 = skip)
FLEURS_OVERSAMPLE = int(os.environ.get("FLEURS_OVERSAMPLE", "2"))  # weight on the eval domain
# Continue from a full fine-tuned model instead of base + adapter (e.g. a domain-adapt pass).
WARM_START_MODEL = os.environ.get("WARM_START_MODEL", "")

processor = WhisperProcessor.from_pretrained(BASE_MODEL)
wer_metric = evaluate.load("wer")


def normalize_igbo(text: str) -> str:
    text = text.strip().replace("’", "'").replace("‘", "'")
    text = re.sub(
        r"[^\w\s\-'àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿ"
        r"ạẹịọụĄąĘęỊịỌọỤụÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖÙÚÛÜÝÞŸ"
        r"ṅṄǹǸ]",
        "",
        text,
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def prepare_fleurs(b):
    a = b["audio"]
    b["input_features"] = processor.feature_extractor(
        a["array"], sampling_rate=a["sampling_rate"], return_tensors="np"
    ).input_features[0]
    b["labels"] = processor.tokenizer(normalize_igbo(b["transcription"])).input_ids
    return b


def prepare_cv(b):
    a = b["audio"]
    b["input_features"] = processor.feature_extractor(
        a["array"], sampling_rate=16000, return_tensors="np"
    ).input_features[0]
    b["labels"] = processor.tokenizer(normalize_igbo(b["sentence"])).input_ids
    return b


def is_valid_length(ex):
    return len(ex["labels"]) <= MAX_LABEL_LEN


print("Loading FLEURS Igbo...")
fleurs = load_dataset("google/fleurs", "ig_ng")
fleurs_train = fleurs["train"].map(prepare_fleurs, remove_columns=fleurs["train"].column_names)

print("Loading Common Voice Igbo...")
cv = load_dataset("benjaminogbonna/nigerian_common_voice_dataset", name="igbo")
cv = cv.cast_column("audio", Audio(sampling_rate=16000))
cv_train = cv["train"].map(prepare_cv, remove_columns=cv["train"].column_names).filter(is_valid_length)


def naijavoices_gen():
    # Stream the gated NaijaVoices Igbo batch (446 GB total — never downloaded whole)
    # and resample 48 kHz -> 16 kHz. from_generator Arrow-caches results to disk.
    stream = load_dataset(
        NAIJA_DATASET, NAIJA_CONFIG, split="train", streaming=True, token=os.environ.get("HF_TOKEN")
    ).cast_column("audio", Audio(sampling_rate=16000))
    n = 0
    for ex in stream:
        if n >= NAIJA_N:
            break
        arr = ex["audio"]["array"]
        if len(arr) > 16000 * 30:  # skip clips beyond Whisper's 30s window
            continue
        labels = processor.tokenizer(normalize_igbo(ex["text"])).input_ids
        if len(labels) > MAX_LABEL_LEN:
            continue
        feats = processor.feature_extractor(arr, sampling_rate=16000, return_tensors="np").input_features[0]
        yield {"input_features": feats, "labels": labels}
        n += 1


# FLEURS is the eval domain (read speech), so oversample it to keep the mix pointed at
# the distribution we're scored on. NAIJA_N=0 skips NaijaVoices (e.g. a FLEURS-focused
# domain-adapt pass that re-points an already-strong model at the test distribution).
parts = [fleurs_train] * FLEURS_OVERSAMPLE + [cv_train]
if NAIJA_N > 0:
    print(f"Streaming NaijaVoices Igbo (up to {NAIJA_N} utterances) — this takes a while...")
    naija_train = Dataset.from_generator(naijavoices_gen)
    print(f"NaijaVoices processed: {len(naija_train)}")
    parts.append(naija_train)

train_data = concatenate_datasets(parts)
fleurs_pct = FLEURS_OVERSAMPLE * len(fleurs_train) / len(train_data) * 100
print(f"Train: {len(train_data)} (FLEURS {fleurs_pct:.0f}% of mix, oversample {FLEURS_OVERSAMPLE}x)")

# Select the best checkpoint on the VALIDATION split, not test — scoring test
# every epoch and reporting that number overfits to it. Final WER is confirmed
# on the held-out test set separately (evaluate_wer.py).
eval_split = "validation" if EVAL_SAMPLES is None else f"validation[:{EVAL_SAMPLES}]"
eval_ds = load_dataset("google/fleurs", "ig_ng", split=eval_split)


@dataclass
class Collator:
    processor: Any

    def __call__(self, features):
        feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(feats, return_tensors="pt")
        labels_in = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(labels_in, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


if WARM_START_MODEL:
    print(f"Warm-starting from full model: {WARM_START_MODEL}")
    model = WhisperForConditionalGeneration.from_pretrained(
        WARM_START_MODEL, torch_dtype=torch.float32, device_map={"": 0}
    )
else:
    print("Warm-starting: merging the existing adapter into the base model...")
    base = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, WARM_START_ADAPTER).merge_and_unload()
# unfreeze the whole network for full fine-tuning (PEFT/merge leaves base params frozen)
for p in model.parameters():
    p.requires_grad_(True)
# To keep the LoRA/export workflow instead, replace the line above with a fresh
# wider adapter on the merged base:
#   merged = PeftModel.from_pretrained(base, WARM_START_ADAPTER).merge_and_unload()
#   model = get_peft_model(merged, LoraConfig(r=64, lora_alpha=128,
#       target_modules=["q_proj","k_proj","v_proj","out_proj"], lora_dropout=0.05, bias="none"))

model.config.forced_decoder_ids = None
model.config.suppress_tokens = None  # None, not [] — an empty list trips save_pretrained's
                                     # "generation params don't belong in config" check
# SpecAugment — applied inside the Whisper encoder during training only.
model.config.apply_spec_augment = True
model.config.mask_time_prob = 0.05
model.config.mask_time_length = 10
model.config.mask_feature_prob = 0.05
model.config.mask_feature_length = 10
model.gradient_checkpointing_enable()
model.config.use_cache = False

collator = Collator(processor)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = EPOCHS * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)
scaler = torch.cuda.amp.GradScaler()


def eval_wer():
    model.config.use_cache = True
    model.eval()
    preds, refs = [], []
    for s in eval_ds:
        feats = processor.feature_extractor(
            np.array(s["audio"]["array"], dtype=np.float32), sampling_rate=16000, return_tensors="pt"
        ).input_features.to("cuda")
        with torch.no_grad():
            ids = model.generate(
                feats,
                forced_decoder_ids=[[1, YO], [2, TRANSCRIBE], [3, NOTS]],
                max_new_tokens=100,
                num_beams=NUM_BEAMS,
            )
        preds.append(processor.tokenizer.decode(ids[0], skip_special_tokens=True))
        refs.append(s["transcription"])
    model.config.use_cache = False
    return wer_metric.compute(
        predictions=[normalize_igbo(p) for p in preds],
        references=[normalize_igbo(r) for r in refs],
    )


best_wer = float("inf")
for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for step, batch in enumerate(train_loader):
        feats = batch["input_features"].to("cuda")
        labels = batch["labels"].to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_features=feats, labels=labels).loss
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running += loss.item()
        if step % 200 == 0:
            print(f"epoch {epoch+1} step {step}/{len(train_loader)} loss {loss.item():.4f}")
            model.save_pretrained(f"{OUT_DIR}_checkpoint")  # Kaggle-interruption insurance

    wer = eval_wer()
    print(f"\nepoch {epoch+1}: train_loss {running/len(train_loader):.4f}  FLEURS-val WER {wer*100:.2f}%\n")
    if wer < best_wer:
        best_wer = wer
        model.save_pretrained(OUT_DIR)
        processor.save_pretrained(OUT_DIR)
        print(f"  new best WER {best_wer*100:.2f}% -> saved to {OUT_DIR}")
        if PUSH_REPO:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=os.environ.get("HF_TOKEN"))
                api.create_repo(PUSH_REPO, exist_ok=True)
                api.upload_folder(folder_path=OUT_DIR, repo_id=PUSH_REPO, repo_type="model")
                print(f"  pushed best to HF: {PUSH_REPO}")
            except Exception as e:
                print(f"  HF push failed (non-fatal): {e}")

print(f"\nDone. Best FLEURS-validation WER: {best_wer*100:.2f}%")
print("Now confirm the real number on the held-out 969-sample TEST set with evaluate_wer.py.")
