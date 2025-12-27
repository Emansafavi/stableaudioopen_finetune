#!/usr/bin/env python3
"""
Batch-generate audio from a Stable Audio Open checkpoint.

Usage:

  python batch_generate_from_ckpt.py \
    --ckpt checkpoints_unwrapped/clotho_finetune_v1_step500_unwrapped.ckpt \
    --config ../stableaudio/models/stabilityai__stable-audio-open-1.0/model_config.json \
    --prompts-dir ./prompts \
    --output-dir ./generated \
    --seconds 8 \
    --batch-size 4

Expected layout:

  prompts/
    prompts.txt   # one text prompt per line

This script:
  - loads your unwrapped fine-tuned checkpoint
  - reads prompts.txt from --prompts-dir
  - generates audio in batches from the checkpoint
  - writes WAVs to --output-dir
"""

import sys
import argparse
import math
from pathlib import Path

import torch
import soundfile as sf

# ---------------------------------------------------------
# Adjust this to your local stable-audio-tools clone
# ---------------------------------------------------------
LOCAL_REPO = "/Users/newpractice/lab/stable-audio-tools"
sys.path.insert(0, LOCAL_REPO)

from stable_audio_tools.models.factory import create_model_from_config_path
from stable_audio_tools.inference.generation import generate_diffusion_cond


def get_device(prefer_mps: bool = True) -> str:
    """Get the best available device."""
    if prefer_mps and torch.backends.mps.is_available():
        print("[device] Using MPS (Metal Performance Shaders)")
        return "mps"
    if torch.cuda.is_available():
        print("[device] Using CUDA")
        return "cuda"
    print("[device] Using CPU")
    return "cpu"


def load_model(ckpt_path: Path, config_path: Path, device: str) -> torch.nn.Module:
    print(f"[load] Model config: {config_path}")
    model = create_model_from_config_path(str(config_path))

    print(f"[load] Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] Missing keys: {len(missing)}  Unexpected keys: {len(unexpected)}")
    if missing:
        print("  first few missing:", missing[:5])
    if unexpected:
        print("  first few unexpected:", unexpected[:5])

    model.eval()

    if device == "mps":
        model.to("mps")
    elif device == "cuda":
        model.cuda()
    else:
        model.cpu()

    return model


def read_prompts(prompts_dir: Path, prompts_file: str = "prompts.txt") -> list[str]:
    p = prompts_dir / prompts_file
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {p}")

    prompts: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(line)

    if not prompts:
        raise RuntimeError(f"No prompts found in {p}")

    print(f"[prompts] Loaded {len(prompts)} prompts from {p}")
    return prompts


def save_batch_wavs(audio_batch: torch.Tensor,
                    sample_rate: int,
                    out_dir: Path,
                    start_index: int,
                    prompts: list[str]):
    """
    Save a batch of audio [B, C, T] to WAV files.
    Names: sample_{global_index:04d}.wav
    """
    audio_batch = audio_batch.detach().cpu()

    bsz = audio_batch.shape[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(bsz):
        idx = start_index + i
        audio = audio_batch[i]  # [C, T]
        audio = audio.to(torch.float32)

        # Normalize to [-1, 1]
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        audio = audio.clamp(-1, 1)

        # [T, C] for soundfile
        audio_np = audio.numpy().T

        fname = f"sample_{idx:04d}.wav"
        path = out_dir / fname
        sf.write(path, audio_np, sample_rate)
        print(f"[save] {fname}  (prompt: {prompts[i]})")


def batch_generate(model,
                   device: str,
                   prompts: list[str],
                   seconds: int,
                   output_dir: Path,
                   batch_size: int = 4,
                   steps: int = 50,
                   cfg_scale: float = 5.0):
    sample_rate = model.sample_rate
    sample_size = seconds * sample_rate
    print(f"[gen] Duration: {seconds}s  sr={sample_rate}  sample_size={sample_size}")
    print(f"[gen] Total prompts: {len(prompts)}  batch_size={batch_size}")

    n = len(prompts)
    num_batches = math.ceil(n / batch_size)

    global_index = 0
    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n)
        batch_prompts = prompts[start:end]

        print(f"\n[gen] Batch {b+1}/{num_batches}  prompts {start}..{end-1}")

        conditioning = []
        for p in batch_prompts:
            conditioning.append({
                "prompt": p,
                "seconds_start": 0,
                "seconds_total": seconds,
            })

        audio = generate_diffusion_cond(
            model=model,
            conditioning=conditioning,
            steps=steps,
            cfg_scale=cfg_scale,
            sample_size=sample_size,
            batch_size=len(batch_prompts),
            seed=-1,   # random seed each call
            device=device,
        )

        save_batch_wavs(
            audio_batch=audio,
            sample_rate=sample_rate,
            out_dir=output_dir,
            start_index=global_index,
            prompts=batch_prompts,
        )

        global_index += len(batch_prompts)


def main():
    ap = argparse.ArgumentParser(
        description="Batch-generate WAVs from a Stable Audio Open checkpoint."
    )
    ap.add_argument("--ckpt", required=True, help="Path to unwrapped checkpoint (.ckpt)")
    ap.add_argument("--config", required=True, help="Path to model_config.json")
    ap.add_argument("--prompts-dir", required=True, help="Directory containing prompts.txt")
    ap.add_argument("--output-dir", required=True, help="Directory to write WAV files")
    ap.add_argument("--seconds", type=int, default=8, help="Duration in seconds per sample")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size for generation")
    ap.add_argument("--steps", type=int, default=50, help="Diffusion steps")
    ap.add_argument("--cfg-scale", type=float, default=5.0, help="CFG scale")
    ap.add_argument("--device", type=str, default=None,
                    choices=["cpu", "cuda", "mps"],
                    help="Force device (default: auto)")

    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    config_path = Path(args.config)
    prompts_dir = Path(args.prompts_dir)
    output_dir = Path(args.output_dir)

    if args.device is None:
        device = get_device(prefer_mps=True)
    else:
        device = args.device
        print(f"[device] Forced device: {device}")

    model = load_model(ckpt_path, config_path, device)
    prompts = read_prompts(prompts_dir)

    batch_generate(
        model=model,
        device=device,
        prompts=prompts,
        seconds=args.seconds,
        output_dir=output_dir,
        batch_size=args.batch_size,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
    )


if __name__ == "__main__":
    main()