#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from audioseal import AudioSeal


def msg_to_16bits(s: str) -> tuple[torch.Tensor, int]:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    val = int.from_bytes(h[:2], "big")  # 0..65535
    bits = [(val >> (15 - i)) & 1 for i in range(16)]
    return torch.tensor(bits, dtype=torch.long), val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--message", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--card", default="audioseal_wm_16bits")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--keep_sr", action="store_true", help="Do not resample, keep original sample rate")
    ap.add_argument("--target_sr", type=int, default=24000, help="Used only if --keep_sr is not set")
    args = ap.parse_args()

    if args.interactive:
        args.message = input("AudioSeal message (will be hashed to 16 bits): ").strip()

    if not args.message:
        args.message = "test1 24.12.2025"

    device = "cpu"  # Force CPU to avoid MPS Metal shader compilation issues

    torch.set_grad_enabled(False)
    gen = AudioSeal.load_generator(args.card).to(device).eval()
    for p in gen.parameters():
        p.requires_grad_(False)

    msg_bits, msg_val = msg_to_16bits(args.message)
    msg_bits = msg_bits.to(device)  # (16,)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write a tiny reproducibility log
    bitstring = "".join(str(int(b)) for b in msg_bits.cpu().tolist())
    (out_dir / "_audioseal_message.txt").write_text(
        f"message: {args.message}\n"
        f"sha256_first16_value: {msg_val}\n"
        f"bits16: {bitstring}\n"
        f"card: {args.card}\n"
        f"strength: {args.strength}\n"
        f"device: {device}\n",
        encoding="utf-8",
    )

    wav_paths = sorted(in_dir.glob("*.wav"))
    if not wav_paths:
        raise SystemExit(f"No wav files found in {in_dir}")

    for p in wav_paths:
        wav, sr = torchaudio.load(str(p))  # (ch, t)

        # optional resample
        if not args.keep_sr and sr != args.target_sr:
            wav = torchaudio.functional.resample(wav, sr, args.target_sr)
            sr = args.target_sr

        # batch it
        wav = wav.unsqueeze(0).to(device)          # (1, ch, t)
        msg = msg_bits.unsqueeze(0)                # (1, 16)

        wm = gen.get_watermark(wav, message=msg)   # (1, ch, t)
        y = wav + (args.strength * wm)

        # avoid uncontrolled clipping during PCM16 conversion
        y = torch.clamp(y, -1.0, 1.0)

        # soundfile wants (t, ch)
        y_np = y.squeeze(0).transpose(0, 1).contiguous().cpu().numpy()
        sf.write(str(out_dir / p.name), y_np, sr, subtype="PCM_16")


if __name__ == "__main__":
    main()