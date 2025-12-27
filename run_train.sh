#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "[train] Stable Audio Open fine-tune (Clotho)"
echo "============================================"

# Resolve repo root (directory of this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SA_FINETUNE_DIR="${ROOT_DIR}"
SA_TOOLS_DIR="${ROOT_DIR}/../stable-audio-tools"

VENV_DIR="${SA_FINETUNE_DIR}/.venv"
CONFIG_TRAIN="${SA_FINETUNE_DIR}/configs/dataset_config.train.abs.json"
CONFIG_VAL="${SA_FINETUNE_DIR}/configs/dataset_config.val.abs.json"
MODEL_DIR="${ROOT_DIR}/../stableaudio/models/stabilityai__stable-audio-open-1.0"
MODEL_CONFIG="${MODEL_DIR}/model_config.json"
MODEL_CKPT="${MODEL_DIR}/model.safetensors"
CHECKPOINT_DIR="${SA_FINETUNE_DIR}/checkpoints"

# Optional: control wandb via env var (default off)
USE_WANDB="${USE_WANDB:-0}"

# ----------------------------------------------------
# Sanity checks
# ----------------------------------------------------
if [ ! -d "$SA_TOOLS_DIR" ]; then
  echo "[error] ../stable-audio-tools not found at: $SA_TOOLS_DIR"
  echo "        Expected layout:"
  echo "          ~/lab/stableaudioopen_finetune/"
  echo "          ~/lab/stable-audio-tools/"
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[error] venv not found at: $VENV_DIR"
  echo "        Create it inside stableaudioopen_finetune first:"
  echo "          cd stableaudioopen_finetune"
  echo "          python3 -m venv .venv"
  echo "          source .venv/bin/activate"
  echo "          pip install -r requirements.txt"
  exit 1
fi

if [ ! -f "$CONFIG_TRAIN" ] || [ ! -f "$CONFIG_VAL" ]; then
  echo "[error] dataset configs not found."
  echo "        Run ./setup.sh first so configs/dataset_config.*.abs.json exist."
  exit 1
fi

if [ ! -f "$MODEL_CONFIG" ] || [ ! -f "$MODEL_CKPT" ]; then
  echo "[error] Stable Audio Open model files not found."
  echo "        Expected:"
  echo "          $MODEL_CONFIG"
  echo "          $MODEL_CKPT"
  echo ""
  echo "        Download with huggingface_hub, for example:"
  echo "          cd .."
  echo "          python -m huggingface_hub download \\"
  echo "            stabilityai/stable-audio-open-1.0 \\"
  echo "            model_config.json \\"
  echo "            --local-dir stableaudio/models/stabilityai__stable-audio-open-1.0"
  echo ""
  echo "          python -m huggingface_hub download \\"
  echo "            stabilityai/stable-audio-open-1.0 \\"
  echo "            model.safetensors \\"
  echo "            --local-dir stableaudio/models/stabilityai__stable-audio-open-1.0"
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR"

# ----------------------------------------------------
# Activate venv
# ----------------------------------------------------
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# Force CPU: hide CUDA and MPS from PyTorch
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_MPS_DISABLE=1

echo "[train] Python: $(python -V)"
python - << 'PY'
import torch
print("[train] torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "mps:", torch.backends.mps.is_available())
PY

# ----------------------------------------------------
# Build training command
# ----------------------------------------------------
CMD=(
  python
  "${SA_FINETUNE_DIR}/train_wrapper.py"
  --config-file "${SA_TOOLS_DIR}/defaults.ini"
  --dataset-config "${CONFIG_TRAIN}"
  --val-dataset-config "${CONFIG_VAL}"
  --model-config "${MODEL_CONFIG}"
  --pretrained-ckpt-path "${MODEL_CKPT}"
  --save-dir "${CHECKPOINT_DIR}"
  --name "clotho_finetune_v1_cpu"
  --precision "32"
  --batch-size 1
  --accum-batches 1
  --num-workers 0
  --checkpoint-every 500
  --val-every 500
)

if [ "$USE_WANDB" = "1" ]; then
  CMD+=(--logger wandb)
else
  echo "[train] WandB disabled (set USE_WANDB=1 to enable)."
fi

echo "[train] Running:"
printf ' %q' "${CMD[@]}"
echo
echo "============================================"

"${CMD[@]}"