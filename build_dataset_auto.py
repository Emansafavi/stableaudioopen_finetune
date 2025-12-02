#!/usr/bin/env python3
import csv
import math
import subprocess
from pathlib import Path
from typing import List, Tuple
import argparse

# ----------------------------- CONFIG ---------------------------------

TARGET_SR = 44100
TARGET_BITS = 16
TARGET_CH = 2
NORMALIZE_DB = -1
SECONDS_PER_CLIP = 4.0

# Light augmentation set (keine extremen Veränderungen)
AUGMENTATIONS: List[Tuple[str, List[str]]] = [
    ("pitchup", ["pitch", "50"]),          # +50 cents
    ("pitchdown", ["pitch", "-50"]),       # -50 cents
    ("tempo98", ["tempo", "-m", "0.98"]),  # -2%
    ("tempo102", ["tempo", "-m", "1.02"]), # +2%
]

CAPTION_TAGS = {
    "pitchup": "(slightly higher pitch)",
    "pitchdown": "(slightly lower pitch)",
    "tempo98": "(slightly slower)",
    "tempo102": "(slightly faster)",
}

# Default: **2 variants per file** (light augmentation)
MAX_VARIANTS_PER_FILE = 2

# ----------------------------------------------------------------------

def run(cmd, logf):
    logf.write(" ".join(cmd) + "\n")
    logf.flush()
    subprocess.run(cmd, check=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sox_convert(in_path: Path, out_path: Path, logf):
    cmd = [
        "sox", str(in_path),
        "-r", str(TARGET_SR),
        "-c", str(TARGET_CH),
        "-b", str(TARGET_BITS),
        str(out_path),
        "gain", "-n", str(NORMALIZE_DB)
    ]
    run(cmd, logf)

def sox_augment(in_path: Path, out_path: Path, effect, logf):
    cmd = ["sox", str(in_path), str(out_path)] + effect + ["gain", "-n", str(NORMALIZE_DB)]
    run(cmd, logf)

def minutes_from_count(n_files: int, seconds_per_clip: float = SECONDS_PER_CLIP) -> float:
    return (n_files * seconds_per_clip) / 60.0

def main():
    ap = argparse.ArgumentParser(description="Light augmentation for Stable Audio finetuning.")
    ap.add_argument("--raw_dir", type=Path, required=True)
    ap.add_argument("--caption_csv", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--target_minutes", type=float, default=120.0)
    ap.add_argument("--max_variants_per_file", type=int, default=MAX_VARIANTS_PER_FILE)
    args = ap.parse_args()

    raw_dir = args.raw_dir
    caption_csv = args.caption_csv
    out_root = args.out_root
    target_minutes = args.target_minutes
    max_variants = args.max_variants_per_file

    audio_out = out_root / "audio"
    logs_out = out_root / "logs"
    ensure_dir(audio_out)
    ensure_dir(logs_out)

    with open(logs_out / "sox_commands.txt", "w") as logf:

        rows = []
        with open(caption_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({"file": r["file"], "caption": r["caption"]})

        raw_wavs = {p.name: p for p in raw_dir.glob("*.wav")}
        existing = [raw_wavs[x["file"]] for x in rows if x["file"] in raw_wavs]

        # Convert originals
        final_rows = []
        for p in existing:
            base = p.stem
            name = f"{base}_44k16.wav"
            outp = audio_out / name
            sox_convert(p, outp, logf)
            src_caption = next(r["caption"] for r in rows if r["file"] == p.name)
            final_rows.append({"file": name, "caption": src_caption})

        base_minutes = minutes_from_count(len(final_rows))
        need_minutes = max(0.0, target_minutes - base_minutes)

        if need_minutes > 0:
            files_needed = math.ceil((need_minutes * 60) / SECONDS_PER_CLIP)
            per_file_variants = min(math.ceil(files_needed / len(final_rows)), max_variants)
        else:
            per_file_variants = 0

        print(f"Target: {target_minutes:.2f} | Base: {base_minutes:.2f}")
        print(f"Generating {per_file_variants} variants per file…")

        # Apply light augmentations
        for row in list(final_rows):
            src = audio_out / row["file"]
            cap = row["caption"]

            for i in range(per_file_variants):
                suffix, effect = AUGMENTATIONS[i % len(AUGMENTATIONS)]
                outname = f"{Path(row['file']).stem}_{suffix}.wav"
                outp = audio_out / outname
                sox_augment(src, outp, effect, logf)

                final_rows.append({
                    "file": outname,
                    "caption": f"{cap} {CAPTION_TAGS[suffix]}"
                })

        # Write metadata
        with open(out_root / "metadata.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "caption"])
            w.writeheader()
            w.writerows(final_rows)

        print(f"Total clips: {len(final_rows)}")
        print(f"Output audio: {audio_out}")
        print(f"Output CSV:   {out_root / 'metadata.csv'}")

if __name__ == "__main__":
    main()

