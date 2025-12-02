#!/usr/bin/env python3
import csv
import os
from pathlib import Path

def validate_split(split_name, audio_dir, metadata_path):
    print(f"\n=== Validating {split_name.upper()} split ===")
    audio_dir = Path(audio_dir).resolve()
    metadata_path = Path(metadata_path).resolve()

    # ---- Load audio files ----
    audio_files = sorted([p for p in audio_dir.glob("*.wav")])
    audio_stems = {p.stem: p for p in audio_files}

    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    # ---- Load metadata ----
    meta_rows = []
    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            meta_rows.append(r)

    print(f"Found {len(meta_rows)} metadata rows in {metadata_path}")

    # ---- Validate mapping ----
    meta_stems = {}
    missing_audio = []
    for r in meta_rows:
        filepath = r["filepath"]
        caption = r["caption"]

        p = Path(filepath)

        stem = p.stem
        meta_stems[stem] = meta_stems.get(stem, 0) + 1

        if not stem in audio_stems:
            missing_audio.append((stem, filepath))

    extra_audio = [p for p in audio_files if p.stem not in meta_stems]

    # ---- Report duplicates ----
    duplicates = [stem for stem, count in meta_stems.items() if count > 1]

    # ---- Print results ----
    print("\n--- Results ---")

    if missing_audio:
        print(f"❌ Missing audio for {len(missing_audio)} metadata entries:")
        for stem, fp in missing_audio[:10]:
            print(f"  - {fp}")
    else:
        print("✔ All metadata entries have matching audio files")

    if extra_audio:
        print(f"❌ {len(extra_audio)} audio files missing metadata:")
        for p in extra_audio[:10]:
            print(f"  - {p.name}")
    else:
        print("✔ All audio files have matching metadata entries")

    if duplicates:
        print(f"❌ Duplicate stems in metadata ({len(duplicates)}):")
        for stem in duplicates[:10]:
            print(f"  - {stem}")
    else:
        print("✔ No duplicate caption entries")

    if not (missing_audio or extra_audio or duplicates):
        print(f"🎉 {split_name.upper()} split looks PERFECT!")
    else:
        print(f"⚠ {split_name.upper()} split has issues — please fix before training.")


if __name__ == "__main__":
    ROOT = Path("data/dataset")

    validate_split(
        "train",
        ROOT / "train/audio",
        ROOT / "train/metadata.csv",
    )

    validate_split(
        "val",
        ROOT / "val/audio",
        ROOT / "val/metadata.csv",
    )

