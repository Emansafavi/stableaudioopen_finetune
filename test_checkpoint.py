import sys
import argparse
import torch
import soundfile as sf
from pathlib import Path

# ---------------------------------------------------------
# Adjust this to your local repo path
# ---------------------------------------------------------
LOCAL_REPO = "/Users/newpractice/lab/stable-audio-tools"
sys.path.insert(0, LOCAL_REPO)

from stable_audio_tools.models.factory import create_model_from_config_path
from stable_audio_tools.inference.generation import generate_diffusion_cond


def get_device():
    """Get the best available device (MPS for macOS, CPU otherwise)"""
    if torch.backends.mps.is_available():
        device = "mps"
        print("[device] Using MPS (Metal Performance Shaders)")
    else:
        device = "cpu"
        print("[device] Using CPU (MPS not available)")
    return device


def load_model(ckpt_path, config_path, device=None):
    if device is None:
        device = get_device()
    
    print(f"[load] Loading model config: {config_path}")
    model = create_model_from_config_path(config_path)

    print(f"[load] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[load] Missing keys:", len(missing))
    print("[load] Unexpected keys:", len(unexpected))

    model.eval()
    if device == "mps":
        model = model.to("mps")
    elif device == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()
    
    return model, device


def generate(model, prompt, seconds=4, output="output.wav", device="cpu", batch_size=1):
    # Calculate sample size from seconds and sample rate
    sample_rate = model.sample_rate
    sample_size = seconds * sample_rate
    
    # Format conditioning dict (use "prompt" key, not "text_prompt")
    # The conditioner expects a list of dicts (batch format), not a single dict
    conditioning_dict = {"prompt": prompt, "seconds_start": 0, "seconds_total": seconds}
    conditioning = [conditioning_dict] * batch_size
    
    print("[gen] Running inference…")
    print(f"[gen] Prompt: {prompt}")
    print(f"[gen] Duration: {seconds}s, Sample rate: {sample_rate}Hz, Sample size: {sample_size}")

    # Generate audio using the diffusion model
    audio = generate_diffusion_cond(
        model=model,
        conditioning=conditioning,
        steps=50,
        cfg_scale=5.0,
        sample_size=sample_size,
        batch_size=batch_size,
        seed=-1,  # Random seed
        device=device,
    )

    # Audio is returned as a tensor with shape [batch, channels, samples]
    # Normalize and convert to numpy
    audio = audio[0].cpu()  # Take first batch item: [channels, samples]
    audio = audio.to(torch.float32)
    # Normalize to [-1, 1] range
    max_val = torch.max(torch.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    audio = audio.clamp(-1, 1)
    
    # Convert to numpy and transpose to [samples, channels] for soundfile
    audio_np = audio.numpy().T
    sf.write(output, audio_np, sample_rate)
    print(f"[gen] Wrote {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seconds", type=int, default=4)
    args = ap.parse_args()

    model, device = load_model(args.ckpt, args.config)
    generate(model, args.prompt, args.seconds, device=device)


if __name__ == "__main__":
    main()

