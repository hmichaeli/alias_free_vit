"""
Comprehensive tests to validate optimized modules produce identical outputs to originals.
"""

import torch
import torch.nn as nn
import numpy as np
from af_ops import (
    DownSampleAF, UpSampleAF, UpAct, PolyActPerChannel, LayerNormAF
)
from af_ops_optimized import (
    DownSampleAF_Optimized, UpSampleAF_Optimized, PolyActPerChannel_Optimized,
    LayerNormAF_Optimized, FusedUpAct_Optimized
)


def compare_tensors(a, b, rtol=1e-5, atol=1e-7, name=""):
    """Compare two tensors for approximate equality"""
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        diff = torch.abs(a - b)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"❌ {name} - Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
        print(f"   Original range: [{a.min():.4f}, {a.max():.4f}]")
        print(f"   Optimized range: [{b.min():.4f}, {b.max():.4f}]")
        return False
    return True


def test_downsample_equivalence():
    """Test DownSampleAF_Optimized produces identical outputs"""
    print("\n" + "="*60)
    print("Testing DownSampleAF Optimization")
    print("="*60)
    
    torch.manual_seed(42)
    passed = True
    
    for down_factor in [2, 4]:
        for size in [8, 16, 32, 64]:
            for batch_size in [1, 2]:
                for channels in [3, 16, 32]:
                    x = torch.randn(batch_size, channels, size, size)
                    
                    # Original
                    orig = DownSampleAF(down=down_factor)
                    with torch.no_grad():
                        out_orig = orig(x)
                    
                    # Optimized
                    opt = DownSampleAF_Optimized(down=down_factor)
                    with torch.no_grad():
                        out_opt = opt(x)
                    
                    if not compare_tensors(
                        out_orig, out_opt, rtol=1e-5, atol=1e-6,
                        name=f"DownSample down={down_factor}, size={size}x{size}, ch={channels}"
                    ):
                        passed = False
    
    if passed:
        print("✓ All DownSampleAF tests passed")
    return passed


def test_upsample_equivalence():
    """Test UpSampleAF_Optimized produces identical outputs"""
    print("\n" + "="*60)
    print("Testing UpSampleAF Optimization")
    print("="*60)
    
    torch.manual_seed(42)
    passed = True
    
    for up_factor in [2, 4]:
        for size in [4, 8, 16, 32]:
            for batch_size in [1, 2]:
                for channels in [3, 16]:
                    x = torch.randn(batch_size, channels, size, size)
                    
                    # Original
                    orig = UpSampleAF(up=up_factor)
                    with torch.no_grad():
                        out_orig = orig(x)
                    
                    # Optimized
                    opt = UpSampleAF_Optimized(up=up_factor)
                    with torch.no_grad():
                        out_opt = opt(x)
                    
                    if not compare_tensors(
                        out_orig, out_opt, rtol=1e-5, atol=1e-6,
                        name=f"UpSample up={up_factor}, size={size}x{size}, ch={channels}"
                    ):
                        passed = False
    
    if passed:
        print("✓ All UpSampleAF tests passed")
    return passed


def test_polyact_equivalence():
    """Test PolyActPerChannel_Optimized produces identical outputs"""
    print("\n" + "="*60)
    print("Testing PolyActPerChannel Optimization")
    print("="*60)
    
    torch.manual_seed(42)
    passed = True
    
    for channels in [8, 16, 32]:
        for data_format in ['channels_first', 'tokens']:
            for degree in [2, 3]:
                # Create both modules with same initialization
                init_coef = [0.1, 0.5, 0.3]
                if degree > 2:
                    init_coef += [0.1] * (degree - 2)
                
                orig = PolyActPerChannel(
                    channels=channels, data_format=data_format, 
                    init_coef=init_coef, degree=degree
                )
                opt = PolyActPerChannel_Optimized(
                    channels=channels, data_format=data_format,
                    init_coef=init_coef, degree=degree
                )
                
                # Copy weights
                opt.coef.data.copy_(orig.coef.data)
                if hasattr(orig, 'in_scale') and orig.in_scale is not None:
                    if isinstance(orig.in_scale, nn.Parameter):
                        opt.in_scale.data.copy_(orig.in_scale.data)
                if hasattr(orig, 'out_scale') and orig.out_scale is not None:
                    if isinstance(orig.out_scale, nn.Parameter):
                        opt.out_scale.data.copy_(orig.out_scale.data)
                
                # Test input
                if data_format == 'channels_first':
                    x = torch.randn(2, channels, 14, 14)
                else:  # tokens
                    x = torch.randn(2, 196, channels)
                
                with torch.no_grad():
                    out_orig = orig(x)
                    out_opt = opt(x)
                
                if not compare_tensors(
                    out_orig, out_opt, rtol=1e-5, atol=1e-6,
                    name=f"PolyAct ch={channels}, format={data_format}, deg={degree}"
                ):
                    passed = False
    
    if passed:
        print("✓ All PolyActPerChannel tests passed")
    return passed


def test_layernorm_equivalence():
    """Test LayerNormAF_Optimized produces identical outputs"""
    print("\n" + "="*60)
    print("Testing LayerNormAF Optimization")
    print("="*60)
    
    torch.manual_seed(42)
    passed = True
    
    for dim in [16, 64, 128]:
        for affine in [False, True]:
            orig = LayerNormAF(dim=dim, affine=affine)
            opt = LayerNormAF_Optimized(dim=dim, affine=affine)
            
            # Copy weights if affine
            if affine:
                opt.weight.data.copy_(orig.weight.data)
                opt.bias.data.copy_(orig.bias.data)
            
            # Test various batch sizes and sequence lengths
            for batch in [1, 4]:
                for seq_len in [49, 196]:
                    x = torch.randn(batch, seq_len, dim)
                    
                    with torch.no_grad():
                        out_orig = orig(x)
                        out_opt = opt(x)
                    
                    if not compare_tensors(
                        out_orig, out_opt, rtol=1e-5, atol=1e-6,
                        name=f"LayerNormAF dim={dim}, affine={affine}, batch={batch}, seq={seq_len}"
                    ):
                        passed = False
    
    if passed:
        print("✓ All LayerNormAF tests passed")
    return passed


def test_fused_upact_equivalence():
    """Test FusedUpAct_Optimized produces identical outputs to UpAct"""
    print("\n" + "="*60)
    print("Testing FusedUpAct Optimization (up == down)")
    print("="*60)
    
    torch.manual_seed(42)
    passed = True
    
    for up_down in [2]:  # When up == down
        for data_format in ['channels_first', 'tokens']:
            act_layer_orig = nn.GELU()
            act_layer_opt = nn.GELU()
            
            orig = UpAct(act_layer=act_layer_orig, up=up_down, down=up_down, data_format=data_format)
            opt = FusedUpAct_Optimized(act_layer=act_layer_opt, up=up_down, down=up_down, data_format=data_format)
            
            if data_format == 'channels_first':
                x = torch.randn(2, 16, 14, 14)
                H, W = 14, 14
            else:  # tokens
                x = torch.randn(2, 196, 16)
                H, W = 14, 14
            
            with torch.no_grad():
                out_orig = orig(x, H, W)
                out_opt = opt(x, H, W)
            
            if not compare_tensors(
                out_orig, out_opt, rtol=1e-4, atol=1e-5,
                name=f"FusedUpAct up={up_down}, format={data_format}"
            ):
                passed = False
    
    if passed:
        print("✓ All FusedUpAct tests passed")
    return passed


def benchmark_module(module, x, num_runs=100, warmup=10, name=""):
    """Benchmark a module's forward pass"""
    import time
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(x) if not hasattr(module, 'requires_input_dimension') else module(x, 14, 14)
    
    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = module(x) if not hasattr(module, 'requires_input_dimension') else module(x, 14, 14)
    end = time.time()
    
    avg_time_ms = (end - start) / num_runs * 1000
    return avg_time_ms


def run_benchmarks():
    """Run performance benchmarks"""
    print("\n" + "="*60)
    print("Performance Benchmarks (CPU)")
    print("="*60)
    
    torch.manual_seed(42)
    
    # DownSampleAF
    x = torch.randn(4, 32, 64, 64)
    orig = DownSampleAF(down=2)
    opt = DownSampleAF_Optimized(down=2)
    
    time_orig = benchmark_module(orig, x, name="DownSampleAF")
    time_opt = benchmark_module(opt, x, name="DownSampleAF_Optimized")
    speedup = time_orig / time_opt
    print(f"DownSampleAF: {time_orig:.3f}ms -> {time_opt:.3f}ms ({speedup:.2f}x speedup)")
    
    # UpSampleAF
    x = torch.randn(4, 32, 32, 32)
    orig = UpSampleAF(up=2)
    opt = UpSampleAF_Optimized(up=2)
    
    time_orig = benchmark_module(orig, x, name="UpSampleAF")
    time_opt = benchmark_module(opt, x, name="UpSampleAF_Optimized")
    speedup = time_orig / time_opt
    print(f"UpSampleAF: {time_orig:.3f}ms -> {time_opt:.3f}ms ({speedup:.2f}x speedup)")
    
    # PolyActPerChannel
    x = torch.randn(4, 16, 32, 32)
    orig = PolyActPerChannel(channels=16, data_format='channels_first')
    opt = PolyActPerChannel_Optimized(channels=16, data_format='channels_first')
    
    time_orig = benchmark_module(orig, x, name="PolyActPerChannel")
    time_opt = benchmark_module(opt, x, name="PolyActPerChannel_Optimized")
    speedup = time_orig / time_opt
    print(f"PolyActPerChannel: {time_orig:.3f}ms -> {time_opt:.3f}ms ({speedup:.2f}x speedup)")
    
    print("\n" + "="*60)


def run_all_tests():
    """Run all equivalence tests"""
    print("\n" + "🔬 " + "="*56 + " 🔬")
    print("   TESTING OPTIMIZED ALIAS-FREE MODULES")
    print("🔬 " + "="*56 + " 🔬")
    
    all_passed = True
    
    all_passed &= test_downsample_equivalence()
    all_passed &= test_upsample_equivalence()
    all_passed &= test_polyact_equivalence()
    all_passed &= test_layernorm_equivalence()
    all_passed &= test_fused_upact_equivalence()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL EQUIVALENCE TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60)
    
    # Run benchmarks
    run_benchmarks()
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
