#!/usr/bin/env python3
"""
Detect watermarks from WAV files in a directory.
This script checks if watermarking was done correctly during setup and detects watermarks.

Usage:
    python detect_watermarks.py [watermarked_dir]
    
If no directory is provided, defaults to: data/raw/clotho_v2/audio/development_watermarked
"""

# Fix SSL certificate verification on macOS BEFORE any other imports
# This is a common issue with Homebrew Python installations
import ssl
import os

try:
    import certifi
    # Use certifi's certificate bundle
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except ImportError:
    pass

# Disable SSL verification as fallback (common fix for macOS Homebrew Python)
# This allows downloads to work even if certificates aren't properly configured
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torchaudio
from pathlib import Path
from audioseal import AudioSeal
import hashlib
import sys
import argparse

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

def msg_to_16bits(s: str) -> tuple[torch.Tensor, int]:
    """
    Convert a text string to a 16-bit binary message.
    Uses SHA256 hash (first 2 bytes) to create a deterministic 16-bit value.
    This matches the encoding used in watermark_audioseal.py.
    """
    h = hashlib.sha256(s.encode("utf-8")).digest()
    val = int.from_bytes(h[:2], "big")  # 0..65535
    bits = [(val >> (15 - i)) & 1 for i in range(16)]
    return torch.tensor(bits, dtype=torch.long), val


def message_accuracy(detected: torch.Tensor, expected: torch.Tensor) -> float:
    """Calculate bit accuracy between detected and expected messages."""
    # Convert to binary (0 or 1)
    detected_binary = (detected > 0.5).int()
    expected_binary = expected.int()
    
    # Count matching bits
    matches = (detected_binary == expected_binary).sum().item()
    total = detected_binary.numel()
    
    return matches / total if total > 0 else 0.0


def load_expected_message(watermarked_dir: Path) -> tuple[str, torch.Tensor, int, str]:
    """
    Load expected message from _audioseal_message.txt file.
    This verifies that watermarking was done correctly during setup.
    
    Returns:
        tuple: (message_text, expected_bits_tensor, expected_value, bitstring)
    """
    message_file = watermarked_dir / "_audioseal_message.txt"
    
    if not message_file.exists():
        raise FileNotFoundError(
            f"Watermarking setup file not found: {message_file}\n"
            f"This file should be created during the watermarking process.\n"
            f"Please run the setup script first to watermark the audio files."
        )
    
    # Parse the message file
    message_data = {}
    with open(message_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                message_data[key.strip()] = value.strip()
    
    expected_message = message_data.get('message', '')
    expected_bitstring = message_data.get('bits16', '')
    expected_value = int(message_data.get('sha256_first16_value', '0'))
    
    if not expected_message:
        raise ValueError(f"Could not parse message from {message_file}")
    
    # Convert bitstring to tensor
    expected_bits = torch.tensor([int(b) for b in expected_bitstring], dtype=torch.long)
    
    return expected_message, expected_bits, expected_value, expected_bitstring


def detect_directory(
    input_dir: str,
    device: str = None,
    detection_threshold: float = 0.5,
    message_threshold: float = 0.5
):
    """
    Detect watermarks in all WAV files in input_dir.
    Automatically loads expected message from _audioseal_message.txt to verify setup.
    
    Args:
        input_dir: Path to directory containing watermarked WAV files to check
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        detection_threshold: Threshold for watermark detection (0-1)
        message_threshold: Threshold for message bit decoding (0-1)
    """
    # Setup paths
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    # Load expected message from setup file
    print("=" * 80)
    print("WATERMARK DETECTION")
    print("=" * 80)
    print(f"\nChecking watermarking setup...")
    
    try:
        expected_message, expected_bits, expected_value, expected_bitstring = load_expected_message(input_path)
        print(f"✓ Found watermarking setup file")
        print(f"  Expected message: '{expected_message}'")
        print(f"  Expected value (SHA256 first 16 bits): {expected_value}")
        print(f"  Expected bitstring: {expected_bitstring}")
    except Exception as e:
        print(f"✗ Error loading watermarking setup: {e}")
        sys.exit(1)
    
    # Setup device
    if device is None:
        device = "cpu"  # Use CPU to match watermarking script
    print(f"\nUsing device: {device}")
    
    expected_bits_tensor = expected_bits.to(device)
    
    # Load detector
    print("\nLoading AudioSeal detector...")
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    detector.eval()
    detector = detector.to(device)
    print("✓ Detector loaded")
    
    # Get all WAV files
    wav_files = sorted(input_path.glob("*.wav"))
    if not wav_files:
        print(f"\n✗ No WAV files found in {input_dir}")
        return []
    
    print(f"\nFound {len(wav_files)} WAV files to check")
    print("-" * 80)
    
    # Store results
    results = []
    watermarked_count = 0
    message_match_count = 0
    
    # Process each file
    for i, wav_file in enumerate(wav_files, 1):
        try:
            # Load audio
            wav, sample_rate = torchaudio.load(str(wav_file))
            
            # Convert to mono if stereo (AudioSeal expects 1 channel)
            if wav.dim() == 2 and wav.shape[0] > 1:
                # Average stereo channels to mono
                wav = wav.mean(dim=0, keepdim=True)  # (1, samples)
            
            # Ensure we have the right shape: (channels, samples)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # (1, samples)
            
            # Add batch dimension for model: (batch, channels, samples)
            wav_batch = wav.unsqueeze(0)  # (1, channels, samples)
            
            # Move to device
            wav_batch = wav_batch.to(device)
            
            # Detect watermark (disable gradients for inference)
            with torch.no_grad():
                detection_prob, decoded_message = detector.detect_watermark(
                    wav_batch,
                    message_threshold=message_threshold,
                    detection_threshold=detection_threshold
                )
            
            # Calculate message accuracy
            decoded_tensor = decoded_message[0].to(device)
            accuracy = message_accuracy(decoded_tensor, expected_bits_tensor)
            
            # Convert to bit-strings for display
            decoded_bits_list = (decoded_tensor > 0.5).int().cpu().tolist()
            decoded_bits_str = ''.join(map(str, decoded_bits_list))
            
            # Determine status
            is_watermarked = detection_prob.item() > detection_threshold
            message_match = accuracy >= 0.9  # Allow 1-2 bit errors (90% accuracy)
            
            if is_watermarked:
                watermarked_count += 1
            if is_watermarked and message_match:
                message_match_count += 1
            
            # Display results
            status_icon = "✓" if (is_watermarked and message_match) else ("⚠" if is_watermarked else "✗")
            print(f"\n[{i}/{len(wav_files)}] {status_icon} {wav_file.name}")
            print(f"    Detection probability: {detection_prob.item():.4f}")
            print(f"    Message accuracy: {accuracy:.2%}")
            print(f"    Encoded message (bits): {decoded_bits_str}")
            if message_match:
                print(f"    Encoded message text: '{expected_message}' ✓")
            else:
                print(f"    Encoded message text: <mismatch>")
            
            # Store result
            results.append({
                "file": wav_file.name,
                "watermarked": is_watermarked,
                "detection_prob": detection_prob.item(),
                "message_accuracy": accuracy,
                "decoded_message_bits": decoded_bits_str,
                "expected_message_bits": expected_bitstring,
                "message_match": message_match,
                "status": ("WATERMARKED (match)" if (is_watermarked and message_match) else 
                          ("WATERMARKED (mismatch)" if is_watermarked else "NOT WATERMARKED"))
            })
            
        except Exception as e:
            print(f"\n[{i}/{len(wav_files)}] ✗ {wav_file.name}")
            print(f"    Error: {e}")
            results.append({
                "file": wav_file.name,
                "watermarked": False,
                "detection_prob": 0.0,
                "message_accuracy": 0.0,
                "error": str(e),
                "status": "ERROR"
            })
            continue
    
    # Print summary
    print("\n" + "=" * 80)
    print("DETECTION SUMMARY")
    print("=" * 80)
    
    total_count = len(results)
    
    print(f"\nTotal files checked: {total_count}")
    print(f"Watermarked files detected: {watermarked_count} ({watermarked_count/total_count*100:.1f}%)")
    print(f"Files with matching message: {message_match_count} ({message_match_count/total_count*100:.1f}%)")
    
    if watermarked_count > 0:
        avg_prob = sum(r["detection_prob"] for r in results if r.get("watermarked")) / watermarked_count
        avg_accuracy = sum(r.get("message_accuracy", 0) for r in results if r.get("watermarked")) / watermarked_count
        print(f"\nAverage detection probability: {avg_prob:.4f}")
        print(f"Average message accuracy: {avg_accuracy:.2%}")
    
    # Overall status
    print("\n" + "-" * 80)
    if watermarked_count == total_count and message_match_count == total_count:
        print("✓ SUCCESS: All files are watermarked with correct message!")
    elif watermarked_count == total_count:
        print("⚠ WARNING: All files are watermarked, but some messages don't match")
    elif watermarked_count > 0:
        print("⚠ WARNING: Some files are not watermarked")
    else:
        print("✗ ERROR: No watermarks detected in any files!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect watermarks in audio files. Verifies watermarking setup was done correctly."
    )
    parser.add_argument(
        "watermarked_dir",
        nargs="?",
        default="data/raw/clotho_v2/audio/development_watermarked",
        help="Directory containing watermarked WAV files (default: data/raw/clotho_v2/audio/development_watermarked)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to use for detection (default: cpu)"
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.5,
        help="Threshold for watermark detection (0-1, default: 0.5)"
    )
    parser.add_argument(
        "--message-threshold",
        type=float,
        default=0.5,
        help="Threshold for message bit decoding (0-1, default: 0.5)"
    )
    
    args = parser.parse_args()
    
    detect_directory(
        input_dir=args.watermarked_dir,
        device=args.device,
        detection_threshold=args.detection_threshold,
        message_threshold=args.message_threshold
    )

