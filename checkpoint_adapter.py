"""
Checkpoint adapter for converting between original and optimized models.

This module provides utilities to:
1. Load checkpoints trained with original modules into optimized models
2. Convert optimized model checkpoints back to original format if needed
"""

import torch
import torch.nn as nn
from collections import OrderedDict


def adapt_checkpoint_original_to_optimized(checkpoint):
    """
    Adapt a checkpoint from original model to work with optimized model.
    
    Since optimized modules maintain parameter compatibility (same names and shapes),
    no adaptation is needed - just return the checkpoint as-is.
    
    Args:
        checkpoint: State dict from original model
        
    Returns:
        State dict compatible with optimized model
    """
    # The optimized modules have the same parameter names and shapes
    # so no conversion is needed
    return checkpoint


def adapt_checkpoint_optimized_to_original(checkpoint):
    """
    Adapt a checkpoint from optimized model to work with original model.
    
    Since optimized modules maintain parameter compatibility (same names and shapes),
    no adaptation is needed - just return the checkpoint as-is.
    
    Args:
        checkpoint: State dict from optimized model
        
    Returns:
        State dict compatible with original model
    """
    # The optimized modules have the same parameter names and shapes
    # so no conversion is needed
    return checkpoint


def replace_modules_with_optimized(model):
    """
    Replace original AF modules with optimized versions in-place.
    
    This function recursively walks through the model and replaces:
    - DownSampleAF -> DownSampleAF_Optimized
    - UpSampleAF -> UpSampleAF_Optimized
    - PolyActPerChannel -> PolyActPerChannel_Optimized
    - LayerNormAF -> LayerNormAF_Optimized
    - UpAct (when up==down) -> FusedUpAct_Optimized
    
    Args:
        model: PyTorch model with original modules
        
    Returns:
        Model with optimized modules (same object, modified in-place)
    """
    from af_ops import DownSampleAF, UpSampleAF, PolyActPerChannel, LayerNormAF, UpAct
    from af_ops_optimized import (
        DownSampleAF_Optimized, UpSampleAF_Optimized, 
        PolyActPerChannel_Optimized, LayerNormAF_Optimized,
        FusedUpAct_Optimized
    )
    
    replacements = []
    
    # Find all modules to replace
    for name, module in model.named_modules():
        if isinstance(module, DownSampleAF):
            new_module = DownSampleAF_Optimized(down=module.down)
            replacements.append((name, new_module))
            
        elif isinstance(module, UpSampleAF):
            new_module = UpSampleAF_Optimized(up=module.up)
            replacements.append((name, new_module))
            
        elif isinstance(module, PolyActPerChannel):
            new_module = PolyActPerChannel_Optimized(
                channels=module.channels,
                data_format=module.data_format,
                degree=module.deg,
                per_channel=module.per_channel
            )
            # Copy parameters
            new_module.coef.data.copy_(module.coef.data)
            if hasattr(module, 'in_scale') and module.in_scale is not None:
                if isinstance(module.in_scale, nn.Parameter):
                    new_module.in_scale.data.copy_(module.in_scale.data)
            if hasattr(module, 'out_scale') and module.out_scale is not None:
                if isinstance(module.out_scale, nn.Parameter):
                    new_module.out_scale.data.copy_(module.out_scale.data)
            replacements.append((name, new_module))
            
        elif isinstance(module, LayerNormAF):
            new_module = LayerNormAF_Optimized(
                dim=module.dim,
                eps=module.eps,
                affine=module.affine
            )
            # Copy parameters if affine
            if module.affine:
                new_module.weight.data.copy_(module.weight.data)
                new_module.bias.data.copy_(module.bias.data)
            replacements.append((name, new_module))
            
        elif isinstance(module, UpAct):
            # Only replace if up == down (can be fused efficiently)
            if module.up == module.down:
                new_module = FusedUpAct_Optimized(
                    act_layer=module.act_layer,
                    up=module.up,
                    down=module.down,
                    data_format=module.data_format
                )
                replacements.append((name, new_module))
    
    # Apply replacements
    for name, new_module in replacements:
        # Navigate to parent and replace
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    print(f"Replaced {len(replacements)} modules with optimized versions")
    return model


def load_checkpoint_with_optimization(model, checkpoint_path, optimize=True):
    """
    Load a checkpoint and optionally optimize the model.
    
    Args:
        model: Model instance (original or optimized)
        checkpoint_path: Path to checkpoint file
        optimize: If True, replace modules with optimized versions
        
    Returns:
        Model with loaded weights (and optionally optimized modules)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    
    # Optionally optimize
    if optimize:
        model = replace_modules_with_optimized(model)
    
    return model


def test_checkpoint_compatibility():
    """
    Test that checkpoints can be loaded into both original and optimized models.
    """
    import tempfile
    import os
    from af_ops import DownSampleAF, PolyActPerChannel
    from af_ops_optimized import DownSampleAF_Optimized, PolyActPerChannel_Optimized
    
    print("Testing checkpoint compatibility...")
    
    # Create a simple model with AF modules
    class SimpleModel(nn.Module):
        def __init__(self, use_optimized=False):
            super().__init__()
            if use_optimized:
                self.down = DownSampleAF_Optimized(down=2)
                self.act = PolyActPerChannel_Optimized(channels=16, data_format='channels_first')
            else:
                self.down = DownSampleAF(down=2)
                self.act = PolyActPerChannel(channels=16, data_format='channels_first')
            self.fc = nn.Linear(16, 10)
        
        def forward(self, x):
            x = self.down(x)
            x = self.act(x)
            B, C, H, W = x.shape
            x = x.mean(dim=(2, 3))
            x = self.fc(x)
            return x
    
    # Create original model and save checkpoint
    model_orig = SimpleModel(use_optimized=False)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
        temp_path = f.name
        torch.save({'model': model_orig.state_dict()}, temp_path)
    
    try:
        # Load into optimized model
        model_opt = SimpleModel(use_optimized=True)
        checkpoint = torch.load(temp_path)
        model_opt.load_state_dict(checkpoint['model'], strict=True)
        
        # Test forward pass produces same output
        x = torch.randn(2, 16, 32, 32)
        with torch.no_grad():
            out_orig = model_orig(x)
            out_opt = model_opt(x)
        
        assert torch.allclose(out_orig, out_opt, rtol=1e-4, atol=1e-5), \
            "Outputs don't match after loading checkpoint!"
        
        print("✓ Checkpoint compatibility test passed")
        
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    test_checkpoint_compatibility()
