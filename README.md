# Stable Audio Open – Fine-tuning Repository

This repository contains scripts and configuration to fine-tune **Stable Audio Open 1.0** models using `stable-audio-tools`, including dataset preparation, training, and checkpoint testing.

The setup assumes **Python 3.10** and a **local clone** of the `stable-audio-tools` repository.

---

## 1. System Requirements

- Linux (tested on Ubuntu)
- NVIDIA GPU with CUDA support
- Python **3.10**
- ffmpeg
- git
- 7zip (p7zip-full)

---

## 2. Python Environment Setup

Create and activate a Python 3.10 virtual environment:

```
python3.10 -m venv .venv  
source .venv/bin/activate  
```

Upgrade pip and build tools:

```
pip install --upgrade pip setuptools wheel  
```

---

## 3. Clone stable-audio-tools

Clone the Stable Audio Tools repository **next to** this fine-tuning repo (recommended layout):

```
cd ..  
git clone https://github.com/Stability-AI/stable-audio-tools.git  
```

Expected directory structure:

```
Documents/  
├── stableaudioopen_finetune/  
└── stable-audio-tools/  
```

Install stable-audio-tools in editable mode:

```
pip install -e ../stable-audio-tools  
```

Important notes:

- stable-audio-tools currently does **not** support Python 3.12
- Python **3.10 is required** due to native dependencies (e.g. PyWavelets)

---

## 4. Download the Base Stable Audio Open Model

Fine-tuning and inference require the **base model weights and configuration**, including `model_config.json`.

Download Stable Audio Open 1.0 from Hugging Face:

```
huggingface-cli download stabilityai/stable-audio-open-1.0 \
  --local-dir ../stableaudio/models/stabilityai__stable-audio-open-1.0 \
  --local-dir-use-symlinks False  
```

After download, you should have:

```
stableaudio/models/stabilityai__stable-audio-open-1.0/  
├── model.safetensors  
├── model_config.json  
├── README.md  
└── ...  
```

The path to `model_config.json` is required when testing fine-tuned checkpoints.

---

## 5. Install Fine-tuning Dependencies

From the fine-tuning repository root:

```
pip install -r requirements.txt  
```

If you encounter build errors:

- Ensure Python version is **3.10**
- If necessary, pin setuptools:

```
pip install "setuptools<81"  
```

---

## 6. Dataset Preparation (Clotho v2 Example)

Run the setup script:

```
bash setup.sh  
```

This will:

- Download Clotho v2 audio and captions
- Extract the audio archive
- Build train / validation splits

Resulting structure:

```
data/  
└── processed/  
    ├── train/  
    └── val/  
```

---

## 7. Training

Training is launched using PyTorch Lightning configuration files.

Example:

```
python3 ../stable-audio-tools/train.py \
  --config-file ../stable-audio-tools/defaults.ini \
  --dataset-config ./configs/dataset_config.train.abs.json \
  --val-dataset-config ./configs/dataset_config.val.abs.json \
  --model-config ./models/stabilityai__stable-audio-open-1.0/model_config.json \
  --pretrained-ckpt-path ./models/stabilityai__stable-audio-open-1.0/model.safetensors \
  --save-dir ./checkpoints \
  --name clotho_finetune_v1 \
  --precision 16-mixed \
  --batch-size 2 \
  --accum-batches 8 \
  --num-workers 4 \
  --checkpoint-every 500 \
  --val-every 500 \
  --logger wandb
```

Checkpoints are written to:

```
checkpoints/  
└── <experiment_name>/  
    └── checkpoints/  
        ├── epoch=*.ckpt  
```

---

## 8. Testing a Fine-tuned Checkpoint

Use `test_checkpoint.py` to run inference with a fine-tuned model.

Example:

```
python test_checkpoint.py \
  --ckpt checkpoints/dcase_finetune_v2/.../epoch=4-step=1000.ckpt \
  --config ../stableaudio/models/stabilityai__stable-audio-open-1.0/model_config.json \
  --prompt "a forest ambience with birds and wind"  
```

The script will:

- Load the base model architecture from `model_config.json`
- Load the fine-tuned checkpoint
- Encode the text prompt using CLAP
- Generate audio and write `out.wav`

---

## 9. Notes on Conditioning and Inference

- Stable Audio Open does **not** expose a `Conditioning` class
- Conditioning is passed internally as **tensor dictionaries**
- Text prompts are embedded using **CLAP**
- Warnings about `flash_attn` can be safely ignored (optional optimization)

---

## 10. Known Issues

- Python 3.12 is **not supported**
- `flash_attn` is optional and disabled if missing
- Missing / unexpected keys when loading checkpoints are expected during fine-tuning

---

## 11. License & Attribution

- Stable Audio Tools © Stability AI
- Dataset licenses apply (e.g. Clotho v2)
- This repository contains **training orchestration only**, not model weights

---

## 12. References

Stable Audio Tools  
https://github.com/Stability-AI/stable-audio-tools  

Stable Audio Open 1.0  
https://huggingface.co/stabilityai/stable-audio-open-1.0  
