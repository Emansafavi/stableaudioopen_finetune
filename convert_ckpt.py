import torch
from pathlib import Path
from safetensors.torch import save_file

# 1) Point to your Lightning checkpoint
CKPT_IN = Path("checkpoints/my_first_ft/tx26yu7o/checkpoints/epoch=25-step=700.ckpt")

# 2) Load it
all_sd = torch.load(str(CKPT_IN), map_location="cpu")
raw = all_sd.get("state_dict", all_sd)  # lightning puts weights under 'state_dict'

# 3) Prefer EMA if present; else non-EMA
if any(k.startswith("diffusion_ema.") for k in raw):
    pref = "diffusion_ema."
elif any(k.startswith("diffusion.") for k in raw):
    pref = "diffusion."
else:
    pref = ""  # already flat

# 4) Strip the wrapper prefix so keys match the base model
fixed = {}
for k, v in raw.items():
    if pref and k.startswith(pref):
        k = k[len(pref):]
    fixed[k] = v

# 5) Write BOTH .ckpt (torch) and .safetensors
OUT_DIR = Path("models/finetuning")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ckpt_out = OUT_DIR / "model.ckpt"
safe_out = OUT_DIR / "model.safetensors"

# .ckpt as a RAW state_dict (what the GUI expects)
torch.save(fixed, str(ckpt_out))

# optional: .safetensors too
save_file(fixed, str(safe_out))

print(f"[ok] wrote:\n  {ckpt_out}\n  {safe_out}  (keys: {len(fixed)})")

