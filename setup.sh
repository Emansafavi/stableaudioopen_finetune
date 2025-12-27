#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "[setup] Stable Audio fine-tuning preparation"
echo "============================================"

ROOT_DIR="$(pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
CLOTHO_DIR="${RAW_DIR}/clotho_v2"

# Extract the archive into CLOTHO_DIR/audio, because the .7z contains a top level "development/" folder
AUDIO_PARENT_DIR="${CLOTHO_DIR}/audio"
AUDIO_DIR="${AUDIO_PARENT_DIR}/development"
AUDIO_WM_DIR="${AUDIO_PARENT_DIR}/development_watermarked"

CAPTIONS_CSV="${CLOTHO_DIR}/captions.csv"

OUT_DIR="${ROOT_DIR}/data/dataset"
TRAIN_DIR="${OUT_DIR}/train"
VAL_DIR="${OUT_DIR}/val"

mkdir -p "${RAW_DIR}" "${CLOTHO_DIR}" "${AUDIO_PARENT_DIR}" "${AUDIO_DIR}" "${AUDIO_WM_DIR}"
mkdir -p "${TRAIN_DIR}/audio" "${VAL_DIR}/audio" configs

# ----------------------------------------------------
# 1. Download Clotho v2 audio + captions
# ----------------------------------------------------
echo "[setup] Downloading Clotho v2…"

AUDIO_ZIP="${CLOTHO_DIR}/clotho_audio_development.7z"
AUDIO_URL="https://zenodo.org/records/4783391/files/clotho_audio_development.7z?download=1"
CAPTIONS_URL="https://zenodo.org/records/4783391/files/clotho_captions_development.csv?download=1"

# Audio archive (resume if partial)
if [ ! -f "$AUDIO_ZIP" ]; then
  curl -L -C - -o "$AUDIO_ZIP" "$AUDIO_URL"
else
  echo "[setup] Audio archive already exists."
fi

# Captions CSV
if [ ! -f "$CAPTIONS_CSV" ]; then
  curl -L -o "$CAPTIONS_CSV" "$CAPTIONS_URL"
else
  echo "[setup] Captions CSV already exists."
fi

# ----------------------------------------------------
# 2. Extract audio
# ----------------------------------------------------
echo "[setup] Extracting Clotho audio…"

# Only extract if AUDIO_DIR is empty
if [ "$(ls -1 "${AUDIO_DIR}"/*.wav 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]; then
  # Extract into AUDIO_PARENT_DIR so archive's internal "development/" lands in AUDIO_DIR
  7z x -y "$AUDIO_ZIP" -o"${AUDIO_PARENT_DIR}"
else
  echo "[setup] Audio already extracted."
fi

echo "[setup] Extract done."
echo " - audio clips found: $(ls ${AUDIO_DIR}/*.wav 2>/dev/null | wc -l)"

# ----------------------------------------------------
# 2.5 Watermark stamping (AudioSeal)
# ----------------------------------------------------
echo "[setup] Watermarking audio with AudioSeal…"

if [ "$(ls -1 "${AUDIO_WM_DIR}"/*.wav 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]; then
  python watermark_audioseal.py \
    --in_dir "$AUDIO_DIR" \
    --out_dir "$AUDIO_WM_DIR" \
    --message "test1 24.12.2025" \
    --keep_sr
else
  echo "[setup] Watermarked audio already exists."
fi

echo " - watermarked clips found: $(ls ${AUDIO_WM_DIR}/*.wav 2>/dev/null | wc -l)"

# ----------------------------------------------------
# 3. Build TRAIN / VAL split
# ----------------------------------------------------
echo "[setup] Building train/val split…"

TRAIN_META="${TRAIN_DIR}/metadata.csv"
VAL_META="${VAL_DIR}/metadata.csv"

echo "filepath,caption" > "$TRAIN_META"
echo "filepath,caption" > "$VAL_META"

total=$(ls ${AUDIO_WM_DIR}/*.wav 2>/dev/null | wc -l | tr -d ' ')
val_count=$(python - << PY
print(int($total * 0.1))
PY
)

python - << PY
import csv, pathlib, random, shutil

audio_dir = pathlib.Path("${AUDIO_WM_DIR}")
captions_path = pathlib.Path("${CAPTIONS_CSV}")
train_dir = pathlib.Path("${TRAIN_DIR}/audio")
val_dir = pathlib.Path("${VAL_DIR}/audio")

rows = list(csv.DictReader(open(captions_path)))
random.seed(42)
random.shuffle(rows)

val_n = $val_count
val_rows = rows[:val_n]
train_rows = rows[val_n:]

def copy_and_write(rows, out_dir, out_meta):
    with open(out_meta, "a", newline="") as meta:
        writer = csv.writer(meta)
        for r in rows:
            fname = r["file_name"]
            caption = r["caption_1"]
            src = audio_dir / fname
            dst = out_dir / fname
            if not src.exists():
                continue
            shutil.copy(src, dst)
            writer.writerow([str(dst), caption])

copy_and_write(train_rows, train_dir, "${TRAIN_META}")
copy_and_write(val_rows, val_dir, "${VAL_META}")

print("[python] Train:", len(train_rows))
print("[python] Val:", len(val_rows))
PY

echo " - train audio: $(ls ${TRAIN_DIR}/audio 2>/dev/null | wc -l)"
echo " - val audio:   $(ls ${VAL_DIR}/audio 2>/dev/null | wc -l)"

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