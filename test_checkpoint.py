import sys
import argparse
import torch
import soundfile as sf
from pathlib import Path

# ---------------------------------------------------------
# Adjust this to your local repo path
# ---------------------------------------------------------
LOCAL_REPO = "/home/stable-audio/Documents/stable-audio-tools"
sys.path.insert(0, LOCAL_REPO)

from stable_audio_tools.models.factory import create_model_from_config_path
from stable_audio_tools.interface.interfaces.diffusion_cond import DiffusionCondInterface


def load_model(ckpt_path, config_path):
    print(f"[load] Loading model config: {config_path}")
    model = create_model_from_config_path(config_path)

    print(f"[load] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[load] Missing keys:", len(missing))
    print("[load] Unexpected keys:", len(unexpected))

    model.eval().cuda()
    return model


def generate(model, prompt, seconds=4, output="output.wav"):
    interface = DiffusionCondInterface(model)

    cond = {"text_prompt": prompt}
    print("[gen] Running inference…")

    x = interface.generate(
        conditions=[cond],
        num_steps=50,
        cfg_scale=5.0,
        sample_rate=44100,
        num_samples=1,
        seconds=seconds,
    )

    audio = x["audio"][0].cpu().numpy()
    sf.write(output, audio, 44100)
    print(f"[gen] Wrote {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seconds", type=int, default=4)
    args = ap.parse_args()

    model = load_model(args.ckpt, args.config)
    generate(model, args.prompt, args.seconds)


if __name__ == "__main__":
    main()

