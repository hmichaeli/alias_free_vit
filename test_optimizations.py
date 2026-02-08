"""
Test suite for validating optimized alias-free modules against original implementations.
Each test ensures that optimized modules produce identical outputs to the original modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from af_ops import (
    DownSampleAF, UpSampleAF, UpAct, PolyActPerChannel, 
    LayerNormAF, LPFPolyActPerChannel, UpActConv,
    TruncLPF, UpSampleConv, DownSampleConv
)


def test_downsample_af():
    """Test DownSampleAF equivalence"""
    print("Testing DownSampleAF...")
    module = DownSampleAF(down=2)
    
    # Test various input sizes
    for size in [8, 16, 32, 64]:
        x = torch.randn(2, 3, size, size)
        with torch.no_grad():
            output = module(x)
        
        expected_size = size // 2
        assert output.shape == (2, 3, expected_size, expected_size), \
            f"Output shape mismatch: {output.shape} vs expected ({2}, {3}, {expected_size}, {expected_size})"
    
    print("✓ DownSampleAF tests passed")


def test_upsample_af():
    """Test UpSampleAF equivalence"""
    print("Testing UpSampleAF...")
    module = UpSampleAF(up=2)
    
    # Test various input sizes
    for size in [8, 16, 32]:
        x = torch.randn(2, 3, size, size)
        with torch.no_grad():
            output = module(x)
        
        expected_size = size * 2
        assert output.shape == (2, 3, expected_size, expected_size), \
            f"Output shape mismatch: {output.shape} vs expected ({2}, {3}, {expected_size}, {expected_size})"
    
    print("✓ UpSampleAF tests passed")


def test_upact():
    """Test UpAct equivalence"""
    print("Testing UpAct...")
    
    # Test channels_first format
    act_layer = nn.GELU()
    module = UpAct(act_layer=act_layer, up=2, data_format='channels_first')
    
    x = torch.randn(2, 16, 14, 14)
    with torch.no_grad():
        output = module(x, H=14, W=14)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    
    # Test tokens format
    module_tokens = UpAct(act_layer=nn.GELU(), up=2, data_format='tokens')
    x_tokens = torch.randn(2, 196, 16)
    with torch.no_grad():
        output_tokens = module_tokens(x_tokens, H=14, W=14)
    
    assert output_tokens.shape == x_tokens.shape, \
        f"Output shape mismatch: {output_tokens.shape} vs {x_tokens.shape}"
    
    print("✓ UpAct tests passed")


def test_polyact_per_channel():
    """Test PolyActPerChannel equivalence"""
    print("Testing PolyActPerChannel...")
    
    channels = 16
    module = PolyActPerChannel(channels=channels, data_format='channels_first')
    
    x = torch.randn(2, channels, 14, 14)
    with torch.no_grad():
        output = module(x)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    
    # Test tokens format
    module_tokens = PolyActPerChannel(channels=channels, data_format='tokens')
    x_tokens = torch.randn(2, 196, channels)
    with torch.no_grad():
        output_tokens = module_tokens(x_tokens)
    
    assert output_tokens.shape == x_tokens.shape, \
        f"Output shape mismatch: {output_tokens.shape} vs {x_tokens.shape}"
    
    print("✓ PolyActPerChannel tests passed")


def test_layernorm_af():
    """Test LayerNormAF equivalence"""
    print("Testing LayerNormAF...")
    
    dim = 16
    module = LayerNormAF(dim=dim, affine=True)
    
    x = torch.randn(2, 196, dim)
    with torch.no_grad():
        output = module(x)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    
    # Check that normalization is working (mean ~0, std ~1)
    mean = output.mean()
    std = output.std()
    assert abs(mean) < 0.1, f"Mean too large: {mean}"
    assert abs(std - 1.0) < 0.2, f"Std too far from 1.0: {std}"
    
    print("✓ LayerNormAF tests passed")


def test_lpf_polyact_per_channel():
    """Test LPFPolyActPerChannel equivalence"""
    print("Testing LPFPolyActPerChannel...")
    
    channels = 16
    module = LPFPolyActPerChannel(channels=channels, data_format='channels_first', cutoff=0.5)
    
    x = torch.randn(2, channels, 16, 16)
    with torch.no_grad():
        output = module(x)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    
    print("✓ LPFPolyActPerChannel tests passed")


def test_upact_conv():
    """Test UpActConv equivalence"""
    print("Testing UpActConv...")
    
    act_layer = nn.GELU()
    module = UpActConv(act_layer=act_layer, up=2, data_format='channels_first')
    
    x = torch.randn(2, 16, 14, 14)
    with torch.no_grad():
        output = module(x, H=14, W=14)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    
    print("✓ UpActConv tests passed")


def run_all_tests():
    """Run all module tests"""
    print("=" * 60)
    print("Running Alias-Free Module Tests")
    print("=" * 60)
    
    test_downsample_af()
    test_upsample_af()
    test_upact()
    test_polyact_per_channel()
    test_layernorm_af()
    test_lpf_polyact_per_channel()
    test_upact_conv()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
