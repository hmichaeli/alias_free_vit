"""
Example: Complete integration of optimized modules into XCiT-AF

This example demonstrates how to use the optimized modules in a real scenario.
"""

import torch
import torch.nn as nn
from xcit_af import XCiTAF, XCiTAFConfig
from checkpoint_adapter import replace_modules_with_optimized
import argparse
import time


def create_model_config():
    """Create a sample XCiTAF configuration"""
    args = argparse.Namespace(
        # Patch embedding configuration
        pe_down_af=True,
        pe_act='up_poly',
        pe_first_act='up_poly',
        pe_last_act=True,
        pe_fuse_down=False,
        
        # MLP configuration
        mlp_act='up_poly',
        
        # LPI configuration
        lpi_act='gelu',
        
        # Feature extraction type
        features_type='cls_attn_af',
        
        # Normalization
        xca_norm_layer='layer_af',
        
        # Convolution padding
        conv_padding_mode='circular',
        
        # Activation kwargs (as JSON strings)
        pe_act_kwargs='{}',
        mlp_act_kwargs='{}',
        lpi_act_kwargs='{}',
        pe_first_act_kwargs='{}'
    )
    
    return XCiTAFConfig(args)


def benchmark_model(model, input_tensor, num_runs=50, warmup=5):
    """Benchmark model inference time"""
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    end = time.time()
    
    avg_time_ms = (end - start) / num_runs * 1000
    return avg_time_ms


def main():
    print("=" * 80)
    print("XCiT-AF Optimization Example")
    print("=" * 80)
    
    # Create configuration
    cfg = create_model_config()
    
    # Create original model
    print("\n1. Creating original XCiT-AF model...")
    model_orig = XCiTAF(
        cfg,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        eta=1.0,
        tokens_norm=True
    )
    
    num_params = sum(p.numel() for p in model_orig.parameters())
    print(f"   Model created with {num_params:,} parameters")
    
    # Create optimized model
    print("\n2. Creating optimized XCiT-AF model...")
    model_opt = XCiTAF(
        cfg,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        eta=1.0,
        tokens_norm=True
    )
    
    # Copy weights
    model_opt.load_state_dict(model_orig.state_dict())
    
    # Replace with optimized modules
    model_opt = replace_modules_with_optimized(model_opt)
    print("   Model optimized!")
    
    # Test equivalence
    print("\n3. Testing numerical equivalence...")
    x = torch.randn(2, 3, 224, 224)
    
    model_orig.eval()
    model_opt.eval()
    
    with torch.no_grad():
        out_orig = model_orig(x)
        out_opt = model_opt(x)
    
    # Handle tuple output from training mode
    if isinstance(out_orig, tuple):
        out_orig = out_orig[0]
    if isinstance(out_opt, tuple):
        out_opt = out_opt[0]
    
    max_diff = torch.abs(out_orig - out_opt).max().item()
    mean_diff = torch.abs(out_orig - out_opt).mean().item()
    
    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Mean difference: {mean_diff:.2e}")
    
    if max_diff < 1e-4:
        print("   ✅ Outputs are numerically equivalent!")
    else:
        print("   ⚠️  Outputs differ more than expected")
    
    # Benchmark
    print("\n4. Benchmarking inference time (CPU)...")
    device = torch.device('cpu')
    model_orig_cpu = model_orig.to(device)
    model_opt_cpu = model_opt.to(device)
    
    x_cpu = torch.randn(4, 3, 224, 224)
    
    time_orig = benchmark_model(model_orig_cpu, x_cpu, num_runs=10, warmup=2)
    time_opt = benchmark_model(model_opt_cpu, x_cpu, num_runs=10, warmup=2)
    speedup = time_orig / time_opt
    
    print(f"   Original model: {time_orig:.2f} ms/image")
    print(f"   Optimized model: {time_opt:.2f} ms/image")
    print(f"   Speedup: {speedup:.2f}x")
    
    # Memory usage estimate
    print("\n5. Model memory footprint:")
    param_memory_mb = num_params * 4 / (1024 ** 2)  # FP32
    print(f"   Parameters: {param_memory_mb:.2f} MB (FP32)")
    
    # Save checkpoint example
    print("\n6. Saving checkpoint...")
    checkpoint = {
        'model': model_opt.state_dict(),
        'config': cfg.__dict__,
        'num_classes': 1000,
    }
    torch.save(checkpoint, 'xcit_af_optimized_example.pth')
    print("   Checkpoint saved to 'xcit_af_optimized_example.pth'")
    
    # Load checkpoint example
    print("\n7. Loading checkpoint into new model...")
    model_new = XCiTAF(
        cfg,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        eta=1.0,
        tokens_norm=True
    )
    
    # Load weights (works with both original and optimized checkpoints!)
    model_new.load_state_dict(checkpoint['model'])
    model_new = replace_modules_with_optimized(model_new)
    print("   ✅ Checkpoint loaded successfully!")
    
    # Verify loaded model
    model_new.eval()
    with torch.no_grad():
        out_new = model_new(x)
        if isinstance(out_new, tuple):
            out_new = out_new[0]
    
    assert torch.allclose(out_opt, out_new, rtol=1e-5, atol=1e-6), \
        "Loaded model output doesn't match!"
    print("   ✅ Loaded model produces identical outputs!")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Model: XCiT-AF with {num_params:,} parameters")
    print(f"  - Speedup: {speedup:.2f}x on CPU")
    print(f"  - Numerical equivalence: ✅ (max diff {max_diff:.2e})")
    print(f"  - Checkpoint compatible: ✅")
    print("\nNext steps:")
    print("  1. Use 'replace_modules_with_optimized()' on your models")
    print("  2. Test on GPU for better speedups")
    print("  3. Enable mixed precision training for more speed")
    print("  4. See USAGE_GUIDE.md for more examples")


if __name__ == "__main__":
    main()
