#!/usr/bin/env python3
"""
Decode watermarks from generated WAV files in a directory.

This is intended for *generated outputs* (from Stable Audio, etc),
so it does NOT touch or modify the dataset detection script.

Usage examples:

  # Use expected message taken from your original watermarked dataset
  python decode_watermarks_generated.py \
    generated_samples/ \
    --watermarked-dir data/raw/clotho_v2/audio/development_watermarked

  # Or pass the message text explicitly
  python decode_watermarks_generated.py \
    generated_samples/ \
    --message "THIS_IS_MY_WATERMARK"

You can also pick the device:

  --device cpu        (recommended, matches watermarking script)
  --device cuda       (if you have GPU)
"""

# Fix SSL certificate verification on macOS BEFORE any other imports
# (same pattern as in your dataset detector script)
import ssl
import os

try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
except ImportError:
    pass

ssl._create_default_https_context = ssl._create_unverified_context

import sys
import argparse
import hashlib
from pathlib import Path

import torch
import torchaudio
from audioseal import AudioSeal

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------------------
# Utilities reused from your other script, but copied here to keep
# the training-data detector untouched.
# ---------------------------------------------------------------------

def msg_to_16bits(s: str) -> tuple[torch.Tensor, int, str]:
    """
    Convert a text string to a 16-bit binary message.

    Uses SHA256 hash (first 2 bytes) to create a deterministic 16-bit value.
    Returns:
      bits_tensor:  tensor of shape [16]
      value:       integer 0..65535
      bitstring:   string of 16 chars '0' or '1'
    """
    h = hashlib.sha256(s.encode("utf-8")).digest()
    val = int.from_bytes(h[:2], "big")  # 0..65535
    bits = [(val >> (15 - i)) & 1 for i in range(16)]
    bit_tensor = torch.tensor(bits, dtype=torch.long)
    bitstring = "".join(str(b) for b in bits)
    return bit_tensor, val, bitstring


def message_accuracy(detected: torch.Tensor, expected: torch.Tensor) -> float:
    """Calculate bit accuracy between detected and expected messages."""
    detected_binary = (detected > 0.5).int()
    expected_binary = expected.int()

    matches = (detected_binary == expected_binary).sum().item()
    total = detected_binary.numel()

    return matches / total if total > 0 else 0.0


def load_expected_message_from_dir(watermarked_dir: Path) -> tuple[str, torch.Tensor, int, str]:
    """
    Load expected message from _audioseal_message.txt inside watermarked_dir.

    This lets you reuse the same message that was used to watermark the training set.
    """
    message_file = watermarked_dir / "_audioseal_message.txt"
    if not message_file.exists():
        raise FileNotFoundError(
            f"_audioseal_message.txt not found in {watermarked_dir}.\n"
            f"Expected file: {message_file}"
        )

    message_data = {}
    with open(message_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                message_data[key.strip()] = value.strip()

    expected_message = message_data.get("message", "")
    expected_bitstring = message_data.get("bits16", "")
    expected_value = int(message_data.get("sha256_first16_value", "0"))

    if not expected_message or not expected_bitstring:
        raise ValueError(f"Could not parse message or bits from {message_file}")

    expected_bits = torch.tensor([int(b) for b in expected_bitstring], dtype=torch.long)
    return expected_message, expected_bits, expected_value, expected_bitstring


# ---------------------------------------------------------------------
# Main detection logic
# ---------------------------------------------------------------------

def detect_generated_dir(
    input_dir: str,
    device: str = "cpu",
    expected_message_text: str | None = None,
    watermarked_dir: str | None = None,
    detection_threshold: float = 0.5,
    message_threshold: float = 0.5,
):
    """
    Detect watermarks in generated WAV files.

    Args:
      input_dir: directory with generated WAVs (each file processed)
      device: "cpu" or "cuda"
      expected_message_text: text message used during watermarking
      watermarked_dir: directory containing _audioseal_message.txt
                       (used to derive expected_message and bits)
      detection_threshold: probability threshold for detector
      message_threshold: threshold for bit decoding
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Determine expected message and bits
    if expected_message_text and watermarked_dir:
        print("[config] Both --message and --watermarked-dir given.")
        print("         Using --message as source of expected bits.")

    if expected_message_text:
        print(f"[config] Using explicit message: {expected_message_text!r}")
        expected_bits, expected_value, expected_bitstring = msg_to_16bits(expected_message_text)
        source_desc = "explicit --message"
    elif watermarked_dir:
        w_dir = Path(watermarked_dir)
        print(f"[config] Loading message from watermarked dir: {w_dir}")
        expected_message_text, expected_bits, expected_value, expected_bitstring = (
            load_expected_message_from_dir(w_dir)
        )
        source_desc = "_audioseal_message.txt in watermarked-dir"
    else:
        raise ValueError(
            "You must provide either --message or --watermarked-dir so I know which bits to expect."
        )

    print("=" * 80)
    print("GENERATED AUDIO WATERMARK DECODING")
    print("=" * 80)
    print(f"\nExpected message source: {source_desc}")
    print(f"  message: {expected_message_text!r}")
    print(f"  sha256 first16 value: {expected_value}")
    print(f"  bits16: {expected_bitstring}")
    print()

    # Setup device
    device = device or "cpu"
    print(f"[device] Using: {device}")
    expected_bits = expected_bits.to(device)

    # Load AudioSeal detector
    print("\n[load] Loading AudioSeal detector (16bit)...")
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    detector.eval()
    detector = detector.to(device)
    print("[load] Detector ready")

    # Collect wavs
    wav_files = sorted(input_path.glob("*.wav"))
    if not wav_files:
        print(f"[info] No WAV files found in {input_dir}")
        return []

    print(f"\nFound {len(wav_files)} WAV files in {input_dir}")
    print("-" * 80)

    results = []
    watermarked_count = 0
    message_match_count = 0

    for i, wav_file in enumerate(wav_files, 1):
        try:
            wav, sr = torchaudio.load(str(wav_file))

            # Convert stereo to mono (AudioSeal expects mono)
            if wav.dim() == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            wav_batch = wav.unsqueeze(0).to(device)  # [1, 1, samples]

            with torch.no_grad():
                detection_prob, decoded_message = detector.detect_watermark(
                    wav_batch,
                    message_threshold=message_threshold,
                    detection_threshold=detection_threshold,
                )

            decoded_tensor = decoded_message[0].to(device)
            accuracy = message_accuracy(decoded_tensor, expected_bits)

            decoded_bits_list = (decoded_tensor > 0.5).int().cpu().tolist()
            decoded_bits_str = "".join(map(str, decoded_bits_list))

            is_watermarked = detection_prob.item() > detection_threshold
            message_match = is_watermarked and (accuracy >= 0.9)

            if is_watermarked:
                watermarked_count += 1
            if message_match:
                message_match_count += 1

            status_icon = "✓" if message_match else ("⚠" if is_watermarked else "✗")

            print(f"\n[{i}/{len(wav_files)}] {status_icon} {wav_file.name}")
            print(f"    Detection prob: {detection_prob.item():.4f}")
            print(f"    Message accuracy: {accuracy:.2%}")
            print(f"    Decoded bits: {decoded_bits_str}")
            print(f"    Expected bits: {expected_bitstring}")
            if message_match:
                print(f"    Message match: YES ({expected_message_text!r})")
            elif is_watermarked:
                print(f"    Message match: NO (watermark present, bits differ)")
            else:
                print("    Message match: NO (no watermark detected)")

            results.append(
                {
                    "file": wav_file.name,
                    "watermarked": is_watermarked,
                    "detection_prob": detection_prob.item(),
                    "message_accuracy": accuracy,
                    "decoded_message_bits": decoded_bits_str,
                    "expected_message_bits": expected_bitstring,
                    "message_match": message_match,
                    "status": (
                        "WATERMARKED (match)"
                        if message_match
                        else "WATERMARKED (mismatch)"
                        if is_watermarked
                        else "NOT WATERMARKED"
                    ),
                }
            )

        except Exception as e:
            print(f"\n[{i}/{len(wav_files)}] ✗ {wav_file.name}")
            print(f"    Error: {e}")
            results.append(
                {
                    "file": wav_file.name,
                    "watermarked": False,
                    "detection_prob": 0.0,
                    "message_accuracy": 0.0,
                    "error": str(e),
                    "status": "ERROR",
                }
            )
            continue

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY FOR GENERATED AUDIO")
    print("=" * 80)

    total_count = len(results)
    print(f"\nTotal files checked: {total_count}")
    print(f"Watermarked files: {watermarked_count} ({watermarked_count/total_count*100:.1f}%)")
    print(
        f"Files with matching message: {message_match_count} "
        f"({message_match_count/total_count*100:.1f}%)"
    )

    if watermarked_count > 0:
        avg_prob = sum(
            r["detection_prob"] for r in results if r.get("watermarked")
        ) / watermarked_count
        avg_acc = sum(
            r.get("message_accuracy", 0.0) for r in results if r.get("watermarked")
        ) / watermarked_count
        print(f"\nAverage detection prob (watermarked files): {avg_prob:.4f}")
        print(f"Average message accuracy (watermarked files): {avg_acc:.2%}")

    print("\n" + "-" * 80)
    if watermarked_count == total_count and message_match_count == total_count:
        print("✓ All generated files are watermarked with the correct message.")
    elif watermarked_count == total_count:
        print("⚠ All generated files seem watermarked, but some messages do not match.")
    elif watermarked_count > 0:
        print("⚠ Only some generated files appear watermarked.")
    else:
        print("✗ No watermarks detected in generated files.")
    print("=" * 80)

    # Optional CSV export
    if HAS_PANDAS:
        df = pd.DataFrame(results)
        out_csv = input_path / "watermark_decoding_results.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n[save] Wrote detailed results to {out_csv}")
    else:
        print("\n[info] pandas not installed, skipping CSV export.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode AudioSeal watermarks from generated WAV files."
    )
    parser.add_argument(
        "generated_dir",
        help="Directory containing generated WAV files to check",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for detection (default: cpu)",
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="Expected message text used during watermarking.",
    )
    parser.add_argument(
        "--watermarked-dir",
        type=str,
        default=None,
        help="Directory of original watermarked dataset with _audioseal_message.txt",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.5,
        help="Threshold for watermark detection probability (0-1).",
    )
    parser.add_argument(
        "--message-threshold",
        type=float,
        default=0.5,
        help="Threshold for message bit decoding (0-1).",
    )

    args = parser.parse_args()

    detect_generated_dir(
        input_dir=args.generated_dir,
        device=args.device,
        expected_message_text=args.message,
        watermarked_dir=args.watermarked_dir,
        detection_threshold=args.detection_threshold,
        message_threshold=args.message_threshold,
    )