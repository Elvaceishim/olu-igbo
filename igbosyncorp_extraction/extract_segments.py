"""
IgboSynCorp ELAN extraction pipeline.

Parses ELAN (.eaf) annotation files from the IgboSynCorp dataset
(Nweya et al., University of Ibadan / Afe-Babalola University, Lacuna Fund,
hosted on Harvard Dataverse: https://doi.org/10.7910/DVN/RXBNCZ and
https://doi.org/10.7910/DVN/YB9FWK) and slices the matching .wav audio
into individual (audio_clip, transcription) segments suitable for ASR
training.

ELAN files store annotations across linked tiers: a "ref" anchor tier
carries the real millisecond timestamps, and a "tx" tier (the orthographic
transcription) references those timestamps indirectly via ANNOTATION_REF.
This script resolves that tier hierarchy via `pympi`, filters out silence
markers and segments outside a sane duration range, and writes one short
.wav clip per valid segment plus a manifest.csv mapping each clip to its
transcription.

This pipeline is general-purpose: it works for any ELAN-annotated speech
corpus with a "tx@<source>" text tier, not just Igbo.

Used in the Olu Igbo project (https://github.com/Elvaceishim/olu-igbo) to
extract ~2,962 training segments from ~40 hours of raw IgboSynCorp oral
narrative recordings. In this project's specific case, adding these
segments to training regressed FLEURS test WER rather than improving it
(see the main README for the full writeup) — but the extraction pipeline
itself is correct and reusable for other low-resource ASR work.

Requires: pympi-ling, soundfile, numpy
"""

import os
import csv
import re
import argparse
import pympi
import soundfile as sf
import numpy as np

# Map .eaf filename -> matching .wav filename. Update this list to match
# whichever IgboSynCorp file pairs you've downloaded (Part I = .eaf
# transcriptions, Part II = .wav audio; both require a free Harvard
# Dataverse guestbook click-through).
FILE_PAIRS = [
    ("Abia_0002_gold.eaf", "Abia_0002.wav"),
    ("Abia_0004_gold.eaf", "Abia_0004.wav"),
    ("Abia_0005_Non_Gold.eaf", "Abia_0005.wav"),
    ("Abia_0010_Non_Gold.eaf", "Abia_0010.WAV"),
    ("Anambra_0002_Non_Gold.eaf", "Anambra_0002.wav"),
    ("Anambra_0010_gold.eaf", "Anambra_0010.WAV"),
    ("Anambra_0011_Non_Gold.eaf", "Anambra_0011.WAV"),
    ("Ebonyi_0011_Non_Gold.eaf", "Ebonyi_0011.wav"),
    ("Ebonyi_0018_Non_Gold.eaf", "Ebonyi_0018.WAV"),
    ("Enugu_0014_gold.eaf", "Enugu_0014.wav"),
    ("Enugu_0025_gold.eaf", "Enugu_0025.WAV"),
    ("Imo_0005_gold.eaf", "Imo_0005.wav"),
    ("Imo_0011_gold.eaf", "Imo_0011.wav"),
]

MIN_DURATION_S = 1.0
MAX_DURATION_S = 20.0


def clean_text(text):
    """Drop ELAN silence/pause markers, normalize whitespace, keep diacritics intact."""
    text = text.strip()
    if text in ("#", "x", ""):
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def get_tx_tier_name(eaf):
    """Find the orthographic text tier dynamically, since tier IDs are file-specific."""
    for name in eaf.get_tier_names():
        if name.startswith("tx@"):
            return name
    return None


def extract_all(eaf_dir, audio_dir, output_dir, manifest_path):
    os.makedirs(output_dir, exist_ok=True)

    manifest_rows = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))
        print(f"Loaded {len(manifest_rows)} existing manifest rows from a previous run.")

    total_segments = 0
    skipped_segments = 0

    for eaf_filename, wav_filename in FILE_PAIRS:
        eaf_path = os.path.join(eaf_dir, eaf_filename)
        wav_path = os.path.join(audio_dir, wav_filename)

        if not os.path.exists(eaf_path) or not os.path.exists(wav_path):
            print(f"SKIP — missing file: {eaf_filename} or {wav_filename}")
            continue

        print(f"Processing {eaf_filename} <-> {wav_filename} ...")

        eaf = pympi.Elan.Eaf(eaf_path)
        tx_tier = get_tx_tier_name(eaf)
        if tx_tier is None:
            print("  No tx@ tier found, skipping.")
            continue

        annotations = eaf.get_annotation_data_for_tier(tx_tier)

        try:
            audio_data, native_sr = sf.read(wav_path, dtype="float32", always_2d=False)
        except Exception as e:
            print(f"  CORRUPTED/UNREADABLE AUDIO — skipping this file entirely: {e}")
            continue
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)  # downmix stereo to mono

        base_name = eaf_filename.replace("_gold.eaf", "").replace("_Non_Gold.eaf", "")

        existing_clips = [f for f in os.listdir(output_dir) if f.startswith(base_name + "_")]
        if existing_clips:
            print(f"  Already processed ({len(existing_clips)} clips found) — skipping.")
            continue

        file_segment_count = 0
        for idx, (start_ms, end_ms, text, _) in enumerate(annotations):
            cleaned = clean_text(text)
            duration_s = (end_ms - start_ms) / 1000.0

            if cleaned is None or duration_s < MIN_DURATION_S or duration_s > MAX_DURATION_S:
                skipped_segments += 1
                continue

            start_sample = int((start_ms / 1000.0) * native_sr)
            end_sample = int((end_ms / 1000.0) * native_sr)
            segment_audio = audio_data[start_sample:end_sample]

            clip_filename = f"{base_name}_{idx:04d}.wav"
            clip_path = os.path.join(output_dir, clip_filename)
            sf.write(clip_path, segment_audio, native_sr, subtype="PCM_16")

            manifest_rows.append({
                "filename": clip_filename,
                "text": cleaned,
                "duration_s": round(duration_s, 2),
                "source_file": base_name,
            })
            file_segment_count += 1
            total_segments += 1

        print(f"  Extracted {file_segment_count} valid segments.")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text", "duration_s", "source_file"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print(f"TOTAL valid segments extracted: {total_segments}")
    print(f"TOTAL skipped (silence/too short/too long): {skipped_segments}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eaf-dir", default=".", help="Directory containing the .eaf files")
    parser.add_argument("--audio-dir", default="./audio", help="Directory containing the .wav files")
    parser.add_argument("--output-dir", default="./clips", help="Where to write extracted clips")
    parser.add_argument("--manifest", default="./manifest.csv", help="Path to the output manifest CSV")
    args = parser.parse_args()

    extract_all(args.eaf_dir, args.audio_dir, args.output_dir, args.manifest)
