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

def repo_supports_flag(train_py: Path, flag_regex: str) -> bool:
    try:
        text = train_py.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return re.search(flag_regex, text) is not None

def repo_supports_val_cfg(train_py: Path) -> bool:
    # explicit flag or argparse wiring
    patterns = [
        r"--val-dataset-config",
        r"add_argument\([^)]*val[_-]dataset[_-]config",
        r"dest\s*=\s*['\"]val_dataset_config['\"]",
    ]
    try:
        text = train_py.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return any(re.search(p, text) for p in patterns)

def convert_lightning_ckpt_to_safetensors(lightning_ckpt: Path) -> Path:
    """
    Best-effort: load a Lightning .ckpt and extract an appropriate state_dict,
    then write a .safetensors file next to it.

    This is only for weights-only restarts when the repo doesn't support true resume.
    """
    import torch
    from safetensors.torch import save_file as safesave

    sd_all = torch.load(str(lightning_ckpt), map_location="cpu")
    # common keys found in SA tools checkpoints
    candidates = [
        "state_dict_ema",   # prefer EMA if present
        "state_dict",
        "model",            # fallback
        "weights",          # generic fallback
    ]
    state_dict = None
    for k in candidates:
        if isinstance(sd_all, dict) and k in sd_all and isinstance(sd_all[k], dict):
            state_dict = sd_all[k]
            break
    if state_dict is None:
        # last resort: if the file already looks like a pure sd
        if isinstance(sd_all, dict):
            state_dict = sd_all
        else:
            raise RuntimeError(f"[resume] Could not find model weights in {lightning_ckpt}")

    # write next to it
    out = lightning_ckpt.with_suffix(".safetensors")
    safesave(state_dict, str(out))
    print(f"[resume] Wrote weights: {out}")
    return out

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

    # NEW: resume options
    ap.add_argument("--ckpt-path", type=Path, default=None,
                    help="True resume: Lightning .ckpt to resume full trainer state (only if repo supports it).")
    ap.add_argument("--resume-lightning-ckpt", type=Path, default=None,
                    help="Weights-only resume: convert this Lightning .ckpt to .safetensors and use as pretrained ckpt.")
    # Optional convenience for keeping the same run folder with WandB
    ap.add_argument("--wandb-id", type=str, default=None,
                    help="If set, exported to WANDB_RUN_ID to keep writing to the same run.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo_dir   = args.repo_dir.resolve()
    defaults   = repo_dir / "defaults.ini"
    train_py   = repo_dir / "train.py"
    if not train_py.exists():
        sys.exit(f"[train] FATAL: {train_py} not found")

    # Detect repo support for optional flags
    val_cfg_supported   = repo_supports_val_cfg(train_py)
    ckpt_flag_supported = repo_supports_flag(train_py, r"--ckpt-path|add_argument\([^)]*ckpt[_-]path|dest\s*=\s*['\"]ckpt_path['\"]")

    # HF model bits
    hf_dir     = args.hf_model_dir.resolve()
    model_cfg  = safe_find(hf_dir, ["model_config.json"])
    ckpt       = safe_find(hf_dir, ["model.safetensors", "model.ckpt"])
    if not model_cfg:
        sys.exit(f"[train] FATAL: model_config.json not found in {hf_dir}")
    if not ckpt and not args.resume_lightning_ckpt:
        sys.exit(f"[train] FATAL: model.safetensors/.ckpt not found in {hf_dir} "
                 f"and no --resume-lightning-ckpt was provided.")

    # dataset config
    dataset_cfg = args.dataset_config.resolve()
    if not dataset_cfg.exists():
        sys.exit(f"[train] FATAL: dataset config not found: {dataset_cfg}")

    # optional val dataset
    val_cfg_str = None
    if args.val_dataset_config:
        v = args.val_dataset_config.resolve()
        if not v.exists():
            sys.exit(f"[train] FATAL: val dataset config not found: {v}")
        if val_cfg_supported:
            val_cfg_str = str(v)

    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare environment (W&B offline by default)
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    if args.wandb_id:
        env["WANDB_RUN_ID"] = args.wandb_id
        env.setdefault("WANDB_RESUME", "allow")

    # Handle resume options
    ckpt_path_forward = None
    # True resume, only if repo supports it
    if args.ckpt_path:
        if ckpt_flag_supported:
            ckpt_path_forward = str(args.ckpt_path.resolve())
        else:
            print("[train] NOTE: repo does not support --ckpt-path (true resume). "
                  "Will attempt weights-only resume instead.")
            # fall through to weights-only if a .ckpt is given
            args.resume_lightning_ckpt = args.ckpt_path

    # Weights-only resume by converting Lightning .ckpt → .safetensors
    if args.resume_lightning_ckpt:
        lightning_ckpt = args.resume_lightning_ckpt.resolve()
        if not lightning_ckpt.exists():
            sys.exit(f"[resume] FATAL: file not found: {lightning_ckpt}")
        try:
            from safetensors.torch import save_file  # check available early
        except Exception as e:
            print("[resume] Missing safetensors package; installing is recommended (pip install safetensors)")
        ckpt = convert_lightning_ckpt_to_safetensors(lightning_ckpt)

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
    if ckpt_path_forward:
        print(f"ckpt_path:  {ckpt_path_forward}  (TRUE resume)")
    elif args.resume_lightning_ckpt:
        print(f"resume:     converted {args.resume_lightning_ckpt} -> {ckpt}  (WEIGHTS-ONLY resume)")
    print(f"extra:      {(args.extra or '(none)')}")
    print("==============================")

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
    if ckpt_path_forward:
        cmd += ["--ckpt-path", ckpt_path_forward]

    if args.extra:
        # allow passing e.g. --checkpoint-every 2000 --seed 42
        cmd += (args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra)

    subprocess.run(cmd, cwd=repo_dir, env=env, check=True)
    print("\n✅ Training launched.")

if __name__ == "__main__":
    main()

