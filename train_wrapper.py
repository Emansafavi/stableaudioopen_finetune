#!/usr/bin/env python3
"""
Wrapper script to:
1. Fix persistent_workers issue when num_workers=0
2. Force CPU training instead of GPU
"""
import sys
import os
import torch.utils.data
from pathlib import Path

# Force CPU: hide CUDA and MPS from PyTorch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_MPS_DISABLE'] = '1'

# Patch DataLoader BEFORE importing anything that uses it
_original_dataloader_init = torch.utils.data.DataLoader.__init__

def patched_dataloader_init(self, *args, **kwargs):
    """Patched DataLoader.__init__ that fixes persistent_workers with num_workers=0"""
    num_workers = kwargs.get('num_workers', 0)
    if num_workers == 0 and kwargs.get('persistent_workers', False):
        kwargs['persistent_workers'] = False
    return _original_dataloader_init(self, *args, **kwargs)

torch.utils.data.DataLoader.__init__ = patched_dataloader_init

# Add stable-audio-tools to path
tools_dir = Path(__file__).parent.parent / 'stable-audio-tools'
sys.path.insert(0, str(tools_dir))

# Import train module and patch the Trainer creation
import train

# Store original main function
_original_main = train.main

def patched_main():
    """Patched main that forces CPU training"""
    import pytorch_lightning as pl
    
    # Store original Trainer
    _original_trainer = pl.Trainer
    
    class CPUTrainer(pl.Trainer):
        """Trainer that always uses CPU"""
        def __init__(self, *args, **kwargs):
            # Force CPU accelerator and override any GPU settings
            kwargs['accelerator'] = 'cpu'
            kwargs['devices'] = 1
            # Remove any GPU-related strategy
            if 'strategy' in kwargs:
                strategy = str(kwargs['strategy']).lower()
                if 'gpu' in strategy or 'cuda' in strategy or 'ddp' in strategy:
                    kwargs['strategy'] = 'auto'
            return _original_trainer.__init__(self, *args, **kwargs)
    
    # Temporarily replace Trainer
    pl.Trainer = CPUTrainer
    try:
        _original_main()
    finally:
        # Restore original
        pl.Trainer = _original_trainer

# Replace main function
train.main = patched_main

# Now run
if __name__ == '__main__':
    train.main()

