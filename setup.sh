#!/usr/bin/bash
# Safe Bash mode (portable)
set -euo pipefail

echo "[setup] Running in Bash $BASH_VERSION"

# ---------- CONFIG (override via env if needed) ----------
: "${PRJ_DIR:=$(cd "$(dirname "$0")" && pwd)}"
: "${REPO_DIR:=${PRJ_DIR}/../stable-audio-tools}"
: "${HF_MODEL_DIR:=${PRJ_DIR}/../stableaudio/models/stabilityai__stable-audio-open-1.0}"
: "${DATA_ROOT:=${PRJ_DIR}/data}"
: "${VENV_DIR:=${PRJ_DIR}/.venv}"

: "${RAW_ZIP_URL:=https://zenodo.org/records/15630417/files/DCASE-TASK7-2024-Open-Source.zip}"
RAW_DIR="${DATA_ROOT}/raw"
RAW_ZIP="${RAW_DIR}/DCASE-TASK7-2024-Open-Source.zip"
UNZIP_DIR="${RAW_DIR}/DCASE"

AUG_ROOT="${DATA_ROOT}/dataset/output_dataset"
AUG_AUDIO="${AUG_ROOT}/audio"
AUG_CSV="${AUG_ROOT}/metadata.csv"

# Train/val split targets
: "${VAL_FRACTION:=0.1}"
SPLIT_ROOT="${DATA_ROOT}/dataset"
TRAIN_ROOT="${SPLIT_ROOT}/train"
VAL_ROOT="${SPLIT_ROOT}/val"
TRAIN_AUDIO="${TRAIN_ROOT}/audio"
VAL_AUDIO="${VAL_ROOT}/audio"
TRAIN_CSV="${TRAIN_ROOT}/metadata.csv"
VAL_CSV="${VAL_ROOT}/metadata.csv"
ALL_CSV="${SPLIT_ROOT}/metadata_all.csv"

# Project-local configs + adapter (outside the repo so we don’t pollute it)
CONF_DIR="${PRJ_DIR}/configs"
ADAPTER_PY="${CONF_DIR}/metadata_adapter.py"
TRAIN_JSON="${CONF_DIR}/dataset_config.train.abs.json"
VAL_JSON="${CONF_DIR}/dataset_config.val.abs.json"

BUILDER_LOCAL="${PRJ_DIR}/build_dataset_auto.py"
: "${BUILDER_URL:=https://raw.githubusercontent.com/inkuele/stableaudio/main/training_scripts/build_dataset_auto.py}"

: "${TARGET_MINUTES:=120}"
: "${MAX_VARIANTS:=19}"

# ---------- OS packages ----------
if command -v apt-get >/dev/null 2>&1; then
  echo "[setup] apt-get install python3.10, sox, unzip, wget, git"
  sudo apt-get update -y
  sudo importt=python3.10 python3.10-venv sox libsox-fmt-all unzip wget git >/dev/null || true
  sudo apt-get install -y python3.10 python3.10-venv sox libsox-fmt-all unzip wget git
else
  echo "[setup][warn] Not Debian/Ubuntu; ensure python3.10, sox, unzip, wget, git are installed."
fi

# ---------- venv ----------
if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Creating venv at $VENV_DIR"
  python3.10 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -V

# ---------- stable-audio-tools ----------
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "[setup] Cloning stable-audio-tools -> $REPO_DIR"
  git clone https://github.com/Stability-AI/stable-audio-tools.git "$REPO_DIR"
else
  echo "[setup] Found repo at $REPO_DIR"
fi

python -m pip install -U pip setuptools wheel
python -m pip install -e "$REPO_DIR"

# Runtime deps
python - <<'PY'
import sys, subprocess
pkgs = [
  "webdataset", "k-diffusion", "alias-free-torch", "auraloss",
  "transformers==4.43.4", "sentencepiece", "pytorch-lightning==2.2.5",
  "accelerate", "safetensors", "einops", "omegaconf", "torchmetrics",
  "pandas", "tqdm", "matplotlib", "wandb", "prefigure", "soundfile"
]
def ensure(p):
    try:
        __import__(p.split("==")[0].replace("-", "_"))
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", p])
for p in pkgs: ensure(p)
print("[setup] Python deps OK.")
PY

# ---------- download raw + build augmented dataset ----------
mkdir -p "$RAW_DIR"
if [ -d "${UNZIP_DIR}/dev/audio" ] && [ -f "${UNZIP_DIR}/dev/caption.csv" ]; then
  echo "[setup] DCASE already unzipped at ${UNZIP_DIR}"
else
  [ -f "$RAW_ZIP" ] || wget -O "$RAW_ZIP" "$RAW_ZIP_URL"
  mkdir -p "$UNZIP_DIR"
  unzip -o -q "$RAW_ZIP" -d "$UNZIP_DIR"
fi

if [ -f "$BUILDER_LOCAL" ]; then
  echo "[setup] Using local builder $BUILDER_LOCAL"
else
  wget -O "$BUILDER_LOCAL" "$BUILDER_URL"
fi

if [ -f "$AUG_CSV" ] && [ -d "$AUG_AUDIO" ] && [ "$(find "$AUG_AUDIO" -type f | wc -l)" -gt 0 ]; then
  echo "[setup] Augmented dataset exists at $AUG_ROOT"
else
  echo "[setup] Building augmented dataset..."
  mkdir -p "$AUG_ROOT"
  python "$BUILDER_LOCAL" \
    --raw_dir       "${UNZIP_DIR}/dev/audio" \
    --caption_csv   "${UNZIP_DIR}/dev/caption.csv" \
    --out_root      "${AUG_ROOT}" \
    --target_minutes "${TARGET_MINUTES}" \
    --max_variants_per_file "${MAX_VARIANTS}"
fi

# ---------- create train/val split (robust, no prune-in-subshell) ----------
echo "[setup] Creating train/val split (${VAL_FRACTION} val)…"
mkdir -p "$TRAIN_AUDIO" "$VAL_AUDIO"
rm -f "${TRAIN_AUDIO}/"* 2>/dev/null || true
rm -f "${VAL_AUDIO}/"*   2>/dev/null || true

# Build shuffled list of absolute file paths
mapfile -t FILES < <(find "$AUG_AUDIO" -maxdepth 1 -type f \( -iname "*.wav" -o -iname "*.flac" -o -iname "*.mp3" -o -iname "*.ogg" \) -print0 | xargs -0 -I{} realpath "{}" | shuf)
TOTAL=${#FILES[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "[setup][ERROR] No audio files found in ${AUG_AUDIO}"; exit 1
fi
VAL_COUNT=$(python - <<PY
import math; print(max(1, min(${TOTAL}-1, int(round(${VAL_FRACTION}*${TOTAL})) )))
PY
)

# Copy first VAL_COUNT to val, rest to train
for idx in "${!FILES[@]}"; do
  src="${FILES[$idx]}"
  if [ "$idx" -lt "$VAL_COUNT" ]; then
    cp -f -- "$src" "$VAL_AUDIO"/
  else
    cp -f -- "$src" "$TRAIN_AUDIO"/
  fi
done

echo "[setup] Train files: $(find "$TRAIN_AUDIO" -type f | wc -l), Val files: $(find "$VAL_AUDIO" -type f | wc -l)"

# ---------- write split CSVs (+ merged) ----------
export AUG_CSV TRAIN_AUDIO VAL_AUDIO TRAIN_CSV VAL_CSV ALL_CSV  # <-- export BEFORE Python

python - <<'PY'
import csv, os
from pathlib import Path

AUG_CSV = Path(os.environ["AUG_CSV"]).resolve()
TRAIN_AUDIO = Path(os.environ["TRAIN_AUDIO"]).resolve()
VAL_AUDIO   = Path(os.environ["VAL_AUDIO"]).resolve()
TRAIN_CSV   = Path(os.environ["TRAIN_CSV"]).resolve()
VAL_CSV     = Path(os.environ["VAL_CSV"]).resolve()
ALL_CSV     = Path(os.environ["ALL_CSV"]).resolve()

# load original rows into map by stem
rows = []
by_stem = {}
if AUG_CSV.exists():
    with AUG_CSV.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            stem = Path(row.get("filepath") or row.get("file") or "").stem
            if not stem: 
                continue
            prompt = row.get("prompt") or row.get("text","")
            rows.append({"filepath": row.get("filepath") or row.get("file"), "prompt": prompt})
            by_stem[stem] = {"prompt": prompt}

def write_for_dir(audio_dir: Path, out_csv: Path):
    out = []
    for p in sorted(audio_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".wav",".flac",".mp3",".ogg"}:
            stem = p.stem
            base = by_stem.get(stem, {"prompt": stem.replace("_"," ").replace("-"," ")})
            out.append({"filepath": str(p), "prompt": base.get("prompt","")})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filepath","prompt"])
        w.writeheader(); w.writerows(out)

write_for_dir(TRAIN_AUDIO, TRAIN_CSV)
write_for_dir(VAL_AUDIO, VAL_CSV)

# merged
with ALL_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["filepath","prompt"])
    w.writeheader()
    for p in (TRAIN_CSV, VAL_CSV):
        with open(p, "r", encoding="utf-8", newline="") as fin:
            for row in csv.DictReader(fin):
                w.writerow(row)

print(f"[setup] Wrote CSVs:\n - {TRAIN_CSV}\n - {VAL_CSV}\n - {ALL_CSV}")
PY

# ---------- write project-local adapter ----------
mkdir -p "$CONF_DIR"
cat > "$ADAPTER_PY" <<'PY'
import os, csv
from pathlib import Path
from typing import Dict

_CACHE = {}  # cache per-CSV-path: {stem: {"prompt": str}}

def _load(csv_path: str):
    p = Path(csv_path).expanduser().resolve()
    if p in _CACHE:
        return
    mapping: Dict[str, Dict[str,str]] = {}
    if p.exists():
        with p.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                stem = Path(r["filepath"]).stem
                mapping[stem] = {"prompt": r.get("prompt","")}
    _CACHE[p] = mapping

def _meta_for(sample_path, csv_path: str):
    p = Path(str(sample_path))
    cp = Path(csv_path).expanduser().resolve()
    _load(str(cp))
    return _CACHE.get(cp, {}).get(p.stem, {"prompt": ""})

def _default_csv_for(sample_path):
    p = Path(str(sample_path)).resolve()
    split_root = p.parent.parent if p.parent.name == "audio" else p.parent
    guess = split_root / "metadata.csv"
    env = os.environ.get("SA_METADATA_CSV")
    return env or str(guess)

# PATH-style (must return a string)
def get_audio_path(sample_path, *_, **__): return str(sample_path)
def get_path(sample_path, *_, **__):       return str(sample_path)
def resolve_path(sample_path, *_, **__):    return str(sample_path)

# METADATA-style (must return a dict)
def get_custom_metadata(info, audio=None):
    path = (isinstance(info, dict) and (info.get("path") or info.get("relpath") or info.get("filepath"))) or audio or info
    csv_path = _default_csv_for(path)
    return _meta_for(path, csv_path)

def get_metadata(*args, **kwargs):
    try: return get_custom_metadata(*args, **kwargs)
    except TypeError: 
        target = args[0] if args else ""
        return _meta_for(target, _default_csv_for(target))
PY

# ---------- write ABS dataset configs (train & val) ----------
TRAIN_AUDIO_ABS="$(realpath "$TRAIN_AUDIO")"
VAL_AUDIO_ABS="$(realpath "$VAL_AUDIO")"
ADAPTER_ABS="$(realpath "$ADAPTER_PY")"
mkdir -p "$CONF_DIR"

cat > "$TRAIN_JSON" <<JSON
{
  "dataset_type": "audio_dir",
  "random_crop": true,
  "datasets": [
    {
      "id": "train_ds",
      "path": "$TRAIN_AUDIO_ABS",
      "recursive": true,
      "extensions": [".wav", ".flac", ".mp3", ".ogg"],
      "custom_metadata_module": "$ADAPTER_ABS"
    }
  ]
}
JSON

cat > "$VAL_JSON" <<JSON
{
  "dataset_type": "audio_dir",
  "random_crop": false,
  "datasets": [
    {
      "id": "val_ds",
      "path": "$VAL_AUDIO_ABS",
      "recursive": true,
      "extensions": [".wav", ".flac", ".mp3", ".ogg"],
      "custom_metadata_module": "$ADAPTER_ABS"
    }
  ]
}
JSON

echo
echo "✅ Setup complete."
echo "   Repo:              ${REPO_DIR}"
echo "   Venv:              ${VENV_DIR}"
echo "   Model dir:         ${HF_MODEL_DIR}"
echo "   Aug dataset:       ${AUG_ROOT}"
echo "   Split roots:       ${SPLIT_ROOT} (train/val + CSVs)"
echo "   Adapter:           ${ADAPTER_PY}"
echo "   Train config:      ${TRAIN_JSON}"
echo "   Val config:        ${VAL_JSON}"
echo
echo "Example launch:"
echo "  export SA_METADATA_CSV='\${ALL_CSV}'"
echo "  \${VENV_DIR}/bin/python train_run.py \\"
echo "    --repo-dir '\${REPO_DIR}' \\"
echo "    --hf-model-dir '\${HF_MODEL_DIR}' \\"
echo "    --dataset-config '\${TRAIN_JSON}' \\"
echo "    --val-dataset-config '\${VAL_JSON}' \\"
echo "    --save-dir '\${PRJ_DIR}/checkpoints' \\"
echo "    --run-name my_first_ft \\"
echo "    --batch-size 2 --accum-batches 8 --precision 16 \\"
echo "    --extra --checkpoint-every 2000 --seed 42"

