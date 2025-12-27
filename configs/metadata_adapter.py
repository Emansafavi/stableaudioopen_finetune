import os, csv
from pathlib import Path
from typing import Dict
import functools

_CACHE = {}  # cache per-CSV-path: {stem: {"prompt": str}}

def _load(csv_path: str):
    p = Path(csv_path).expanduser().resolve()
    if p in _CACHE:
        return
    mapping: Dict[str, Dict[str,str]] = {}
    if p.exists():
        with p.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                stem = Path(r["filepath"]).stem
                mapping[stem] = {"prompt": r.get("prompt","")}
    _CACHE[p] = mapping

def _meta_for(sample_path, csv_path: str):
    p = Path(str(sample_path))
    cp = Path(csv_path).expanduser().resolve()
    _load(str(cp))
    return _CACHE.get(cp, {}).get(p.stem, {"prompt": ""})

def _default_csv_for(sample_path):
    p = Path(str(sample_path)).resolve()
    split_root = p.parent.parent if p.parent.name == "audio" else p.parent
    guess = split_root / "metadata.csv"
    env = os.environ.get("SA_METADATA_CSV")
    return env or str(guess)

# PATH-style (must return a string)
def get_audio_path(sample_path, *_, **__): return str(sample_path)
def get_path(sample_path, *_, **__):       return str(sample_path)
def resolve_path(sample_path, *_, **__):    return str(sample_path)

# METADATA-style (must return a dict)
# Make this function pickleable by using a module-level function
def get_custom_metadata(info, audio=None):
    path = (isinstance(info, dict) and (info.get("path") or info.get("relpath") or info.get("filepath"))) or audio or info
    csv_path = _default_csv_for(path)
    return _meta_for(path, csv_path)

def get_metadata(*args, **kwargs):
    try: return get_custom_metadata(*args, **kwargs)
    except TypeError: 
        target = args[0] if args else ""
        return _meta_for(target, _default_csv_for(target))

# Make the function pickleable by ensuring it's at module level
# and can be imported properly
__all__ = ['get_custom_metadata', 'get_metadata', 'get_audio_path', 'get_path', 'resolve_path']
