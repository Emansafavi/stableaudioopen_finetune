#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "[setup] Stable Audio fine-tuning preparation"
echo "============================================"

ROOT_DIR="$(pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
CLOTHO_DIR="${RAW_DIR}/clotho_v2"
AUDIO_DIR="${CLOTHO_DIR}/audio/development"
CAPTIONS_CSV="${CLOTHO_DIR}/captions.csv"

OUT_DIR="${ROOT_DIR}/data/dataset"
TRAIN_DIR="${OUT_DIR}/train"
VAL_DIR="${OUT_DIR}/val"

mkdir -p "${RAW_DIR}" "${CLOTHO_DIR}" "${AUDIO_DIR}"
mkdir -p "${TRAIN_DIR}/audio" "${VAL_DIR}/audio" configs

# ----------------------------------------------------
# 1. Download Clotho v2 audio + captions
# ----------------------------------------------------
echo "[setup] Downloading Clotho v2…"

AUDIO_ZIP="${CLOTHO_DIR}/clotho_audio_development.7z"
CAPTIONS_URL="https://zenodo.org/records/4783391/files/clotho_captions_development.csv?download=1"

# Audio (official mirror)
if [ ! -f "$AUDIO_ZIP" ]; then
    wget -O "$AUDIO_ZIP" \
        "https://zenodo.org/records/4783391/files/clotho_audio_development.7z?download=1"
else
    echo "[setup] Audio archive already exists."
fi

# Captions
if [ ! -f "$CAPTIONS_CSV" ]; then
    wget -O "$CAPTIONS_CSV" "$CAPTIONS_URL"
else
    echo "[setup] Captions CSV already exists."
fi

# ----------------------------------------------------
# 2. Extract audio
# ----------------------------------------------------
echo "[setup] Extracting Clotho audio…"

7z x "$AUDIO_ZIP" -o"$AUDIO_DIR" 

echo "[setup] Extract done."
echo " - audio clips found: $(ls ${AUDIO_DIR}/*.wav | wc -l)"

# ----------------------------------------------------
# 3. Build TRAIN / VAL split
# ----------------------------------------------------
echo "[setup] Building train/val split…"

TRAIN_META="${TRAIN_DIR}/metadata.csv"
VAL_META="${VAL_DIR}/metadata.csv"

echo "filepath,caption" > "$TRAIN_META"
echo "filepath,caption" > "$VAL_META"

total=$(ls ${AUDIO_DIR}/*.wav | wc -l)
val_count=$(python3 - << PY
import math
print(int($total * 0.1))
PY
)

python3 - << PY
import csv, pathlib, random, shutil

audio_dir = pathlib.Path("${AUDIO_DIR}")
captions_path = pathlib.Path("${CAPTIONS_CSV}")
train_dir = pathlib.Path("${TRAIN_DIR}/audio")
val_dir = pathlib.Path("${VAL_DIR}/audio")

# Load captions
rows = list(csv.DictReader(open(captions_path)))
random.seed(42)

# Random shuffle for split
random.shuffle(rows)

val_n = $val_count
val_rows = rows[:val_n]
train_rows = rows[val_n:]

def copy_and_write(rows, out_dir, out_meta):
    with open(out_meta, "a") as meta:
        writer = csv.writer(meta)
        for r in rows:
            # Clotho CSV uses file 'xxx.wav' and 5 captions spread over columns
            # Choose caption_1 (or merge if needed)
            fname = r["file_name"]
            caption = r["caption_1"]
            src = audio_dir / fname
            dst = out_dir / fname
            shutil.copy(src, dst)
            writer.writerow([str(dst), caption])

copy_and_write(train_rows, train_dir, "${TRAIN_META}")
copy_and_write(val_rows, val_dir, "${VAL_META}")

print("[python] Train:", len(train_rows))
print("[python] Val:", len(val_rows))
PY

echo " - train audio: $(ls ${TRAIN_DIR}/audio | wc -l)"
echo " - val audio:   $(ls ${VAL_DIR}/audio | wc -l)"

# ----------------------------------------------------
# 4. Create dataset configs (Stable Audio format)
# ----------------------------------------------------
echo "[setup] Writing dataset configs…"

TRAIN_ABS="$(realpath "$TRAIN_DIR/audio")"
VAL_ABS="$(realpath "$VAL_DIR/audio")"
METADATA_ADAPTER="$(realpath "configs/metadata_adapter.py")"

cat > configs/dataset_config.train.abs.json << EOF
{
  "dataset_type": "audio_dir",
  "random_crop": true,
  "datasets": [
    {
      "id": "train_ds",
      "path": "$TRAIN_ABS",
      "recursive": false,
      "extensions": [".wav"],
      "custom_metadata_module": "$METADATA_ADAPTER"
    }
  ]
}
EOF

cat > configs/dataset_config.val.abs.json << EOF
{
  "dataset_type": "audio_dir",
  "random_crop": false,
  "datasets": [
    {
      "id": "val_ds",
      "path": "$VAL_ABS",
      "recursive": false,
      "extensions": [".wav"],
      "custom_metadata_module": "$METADATA_ADAPTER"
    }
  ]
}
EOF

echo "[setup] DONE ✓"

