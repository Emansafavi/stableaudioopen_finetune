#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, subprocess, re
from pathlib import Path

def safe_find(p: Path, candidates):
    for name in candidates:
        f = p / name
        if f.exists():
            return f
    return None

def repo_supports_val_cfg(train_py: Path) -> bool:
    """Return True if the repo's train.py exposes a --val-dataset-config arg."""
    try:
        text = train_py.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    patterns = [
        r"--val-dataset-config",                             # explicit flag
        r"add_argument\([^)]*val[_-]dataset[_-]config",      # argparse add_argument
        r"dest\s*=\s*['\"]val_dataset_config['\"]",          # explicit dest
    ]
    return any(re.search(p, text) for p in patterns)

def main():
    ap = argparse.ArgumentParser("Stable Audio Open fine-tune launcher (doc-aligned)")
    ap.add_argument("--repo-dir", type=Path, required=True)
    ap.add_argument("--hf-model-dir", type=Path, required=True,
                    help="Local folder that contains HF files (model.ckpt or model.safetensors, model_config.json)")
    ap.add_argument("--dataset-config", type=Path, required=True)
    ap.add_argument("--val-dataset-config", type=Path, default=None,
                    help="Optional validation dataset config (only passed if repo supports it)")
    ap.add_argument("--save-dir", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum-batches", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--precision", choices=["16","bf16","32"], default="16")
    ap.add_argument("--extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo_dir   = args.repo_dir.resolve()
    defaults   = repo_dir / "defaults.ini"
    train_py   = repo_dir / "train.py"
    if not train_py.exists():
        sys.exit(f"[train] FATAL: {train_py} not found")

    # HF model bits
    hf_dir     = args.hf_model_dir.resolve()
    model_cfg  = safe_find(hf_dir, ["model_config.json"])
    ckpt       = safe_find(hf_dir, ["model.safetensors", "model.ckpt"])
    if not model_cfg:
        sys.exit(f"[train] FATAL: model_config.json not found in {hf_dir}")
    if not ckpt:
        sys.exit(f"[train] FATAL: model.safetensors/.ckpt not found in {hf_dir}")

    # dataset config
    dataset_cfg = args.dataset_config.resolve()
    if not dataset_cfg.exists():
        sys.exit(f"[train] FATAL: dataset config not found: {dataset_cfg}")

    # val dataset config (optional, only pass if supported)
    val_cfg_supported = repo_supports_val_cfg(train_py)
    val_cfg_str = None
    if args.val_dataset_config:
        v = args.val_dataset_config.resolve()
        if not v.exists():
            sys.exit(f"[train] FATAL: val dataset config not found: {v}")
        if val_cfg_supported:
            val_cfg_str = str(v)

    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== Stable Audio Fine-tune ===")
    print(f"repo_dir:   {repo_dir}")
    print(f"defaults:   {defaults}")
    print(f"model_cfg:  {model_cfg}")
    print(f"ckpt:       {ckpt}")
    print(f"dataset:    {dataset_cfg}")
    if args.val_dataset_config:
        if val_cfg_supported:
            print(f"val_set:    {args.val_dataset_config.resolve()}")
        else:
            print("[train] NOTE: repo does not support --val-dataset-config; running without a val loader.")
    print(f"save_dir:   {save_dir}")
    print(f"run_name:   {args.run_name}")
    print(f"precision:  {args.precision}")
    print(f"workers:    {args.num_workers}  batch: {args.batch_size}  accum: {args.accum_batches}")
    print(f"extra:      {(args.extra or '(none)')}")
    print("==============================")

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")

    cmd = [
        sys.executable, "-u", "-m", "train",
        "--config-file", str(defaults),
        "--dataset-config", str(dataset_cfg),
        "--model-config", str(model_cfg),
        "--pretrained-ckpt-path", str(ckpt),
        "--save-dir", str(save_dir),
        "--name", str(args.run_name),
        "--precision", str(args.precision),
        "--num-workers", str(args.num_workers),
        "--batch-size", str(args.batch_size),
        "--accum-batches", str(args.accum_batches),
    ]
    if val_cfg_str:
        cmd += ["--val-dataset-config", val_cfg_str]

    if args.extra:
        # allow passing e.g. --checkpoint-every 2000 --seed 42
        cmd += (args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra)

    subprocess.run(cmd, cwd=repo_dir, env=env, check=True)
    print("\n✅ Training launched via official flags.")

if __name__ == "__main__":
    main()

