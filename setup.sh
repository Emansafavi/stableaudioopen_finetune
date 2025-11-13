#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Starting Stable Audio finetune dataset setup…"

# -----------------------------
# Paths
# -----------------------------
ROOT_DIR="$(pwd)"
RAW_DIR="${ROOT_DIR}/data/raw/DCASE"
DEV_AUDIO="${RAW_DIR}/dev/audio"
EVAL_AUDIO="${RAW_DIR}/eval/audio"

OUT_DIR="${ROOT_DIR}/data/dataset"
AUG_DIR="${OUT_DIR}/augmented"
TRAIN_DIR="${OUT_DIR}/train"
VAL_DIR="${OUT_DIR}/val"

mkdir -p "${AUG_DIR}/audio" "${AUG_DIR}/metadata"
mkdir -p "${TRAIN_DIR}/audio" "${VAL_DIR}/audio"

# -----------------------------
# Check raw dataset
# -----------------------------
if [ ! -d "$RAW_DIR" ]; then
    echo "[ERROR] Raw DCASE dataset not found in data/raw/DCASE"
    exit 1
fi

echo "[setup] Found DCASE dataset:"
echo " - dev:  $(ls ${DEV_AUDIO} | wc -l) files"
echo " - eval: $(ls ${EVAL_AUDIO} | wc -l) files"

# -----------------------------
# Step 1: Build CSV with dev captions
# -----------------------------
DEV_CAPTIONS="${RAW_DIR}/dev/caption.csv"
if [ ! -f "$DEV_CAPTIONS" ]; then
    echo "[ERROR] Missing dev caption.csv at $DEV_CAPTIONS"
    exit 1
fi

cp "$DEV_CAPTIONS" "${AUG_DIR}/metadata/metadata_dev.csv"
echo "[setup] Copied dev captions to augmented directory."

# -----------------------------
# Step 2: Augment only dev/
# -----------------------------
echo "[setup] Running audio augmentation (medium level = 10 variants)…"

python3 build_dataset_auto.py \
    --raw_dir "$DEV_AUDIO" \
    --caption_csv "$DEV_CAPTIONS" \
    --out_root "$AUG_DIR" \
    --max_variants_per_file 10

echo "[setup] Augmentation done."
echo " - augmented audio: $(ls ${AUG_DIR}/audio | wc -l)"
echo " - augmented metadata: $(wc -l ${AUG_DIR}/metadata/metadata_all.csv)"

# -----------------------------
# Step 3: Create train/val split (from augmented only!)
# -----------------------------
echo "[setup] Creating train/val split (90% train / 10% val)…"

python3 - << 'PY'
import csv, pathlib, random, shutil

root = pathlib.Path("data/dataset/augmented")
train_dir = pathlib.Path("data/dataset/train/audio")
val_dir   = pathlib.Path("data/dataset/val/audio")

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

meta = root / "metadata/metadata_all.csv"
rows = list(csv.DictReader(open(meta)))

random.seed(42)
random.shuffle(rows)

split = int(len(rows) * 0.9)
train_rows = rows[:split]
val_rows   = rows[split:]

# Save metadata
with open("data/dataset/train/metadata.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(train_rows)

with open("data/dataset/val/metadata.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(val_rows)

# copy audio files
def copy_rows(rows, out_dir):
    for r in rows:
        src = pathlib.Path(r["filepath"])
        dst = out_dir / src.name
        shutil.copy(src, dst)

copy_rows(train_rows, train_dir)
copy_rows(val_rows, val_dir)

print(f"[python] Train files: {len(train_rows)}")
print(f"[python] Val files:   {len(val_rows)}")
PY

echo "[setup] Train/val split completed."
echo " - train: $(ls ${TRAIN_DIR}/audio | wc -l)"
echo " - val:   $(ls ${VAL_DIR}/audio | wc -l)"

# -----------------------------
# Step 4: Write dataset config JSONs
# -----------------------------
echo "[setup] Writing dataset config files…"

cat > configs/dataset_config.train.abs.json << EOF
{
  "audio_dir": "${TRAIN_DIR}/audio",
  "metadata_path": "${TRAIN_DIR}/metadata.csv",
  "sample_rate": 44100,
  "duration": 4.0
}
EOF

cat > configs/dataset_config.val.abs.json << EOF
{
  "audio_dir": "${VAL_DIR}/audio",
  "metadata_path": "${VAL_DIR}/metadata.csv",
  "sample_rate": 44100,
  "duration": 4.0
}
EOF

echo "[setup] Done."

