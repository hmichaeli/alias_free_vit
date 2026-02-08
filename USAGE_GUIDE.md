# Usage Guide for Optimized Alias-Free ViT Modules

## Quick Start

### 1. Basic Usage - Replace Individual Modules

```python
from af_ops_optimized import (
    DownSampleAF_Optimized,
    UpSampleAF_Optimized,
    PolyActPerChannel_Optimized,
    LayerNormAF_Optimized,
    FusedUpAct_Optimized
)

# Replace downsample operation
downsample = DownSampleAF_Optimized(down=2)

# Replace polynomial activation
polyact = PolyActPerChannel_Optimized(
    channels=128,
    data_format='channels_first',
    degree=2
)

# Use in your model
x = torch.randn(4, 128, 56, 56)
x_down = downsample(x)
x_act = polyact(x)
```

### 2. Automatic Model Optimization

```python
from xcit_af import XCiTAF, XCiTAFConfig
from checkpoint_adapter import replace_modules_with_optimized
import argparse

# Create your model as usual
args = argparse.Namespace(
    pe_down_af=True,
    pe_act='up_poly',
    pe_first_act='up_poly',
    # ... other args
)
cfg = XCiTAFConfig(args)
model = XCiTAF(cfg, patch_size=16, embed_dim=384, depth=12, num_heads=8)

# Optimize all AF modules automatically
model = replace_modules_with_optimized(model)

# Model is now optimized and ready to use!
```

### 3. Load Existing Checkpoints with Optimization

```python
from checkpoint_adapter import load_checkpoint_with_optimization

# Load checkpoint and optimize in one step
model = XCiTAF(cfg, patch_size=16, embed_dim=384, depth=12, num_heads=8)
model = load_checkpoint_with_optimization(
    model, 
    'path/to/checkpoint.pth',
    optimize=True  # Automatically replace with optimized modules
)

# Model loaded with weights and optimized!
```

## Advanced Usage

### 1. TorchScript-Compiled Modules

For production deployment, use TorchScript-compiled versions:

```python
from advanced_optimizations import LayerNormAF_TorchScript

# Replace LayerNorm with TorchScript version
layernorm = LayerNormAF_TorchScript(dim=256, eps=1e-6, affine=True)

# Can be traced/scripted for deployment
scripted_norm = torch.jit.script(layernorm)
scripted_norm.save('layernorm_scripted.pt')
```

### 2. Memory-Efficient Training

For training with limited GPU memory:

```python
from advanced_optimizations import MemoryEfficientUpAct

# Use gradient checkpointing version
upact = MemoryEfficientUpAct(
    act_layer=nn.GELU(),
    up=2,
    data_format='channels_first'
)

# Saves memory during training by recomputing activations
# in backward pass instead of storing them
```

### 3. CUDA-Optimized Modules (GPU only)

When CUDA is available and you have a GPU:

```python
from advanced_optimizations import PolyActPerChannel_CUDA

# Use CUDA-optimized polynomial activation
if torch.cuda.is_available():
    polyact = PolyActPerChannel_CUDA(
        channels=128,
        data_format='channels_first',
        degree=2
    ).cuda()
    
    x = torch.randn(4, 128, 56, 56).cuda()
    output = polyact(x)  # ~3-5x faster than PyTorch version
```

## Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from xcit_af import XCiTAF, XCiTAFConfig
from checkpoint_adapter import replace_modules_with_optimized
import argparse

# Setup
args = argparse.Namespace(
    pe_down_af=True,
    pe_act='up_poly',
    pe_first_act='up_poly',
    pe_last_act=True,
    pe_fuse_down=False,
    mlp_act='up_poly',
    lpi_act='gelu',
    features_type='cls_attn_af',
    xca_norm_layer='layer_af',
    conv_padding_mode='circular',
    pe_act_kwargs='{}',
    mlp_act_kwargs='{}',
    lpi_act_kwargs='{}',
    pe_first_act_kwargs='{}'
)

cfg = XCiTAFConfig(args)

# Create model
model = XCiTAF(
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
    drop_path_rate=0.1
)

# Optimize the model
model = replace_modules_with_optimized(model)
print("Model optimized!")

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output, _ = model(data)  # XCiTAF returns (output, output) during training
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, f'checkpoint_epoch_{epoch}.pth')
```

## Inference Example

```python
import torch
from xcit_af import XCiTAF, XCiTAFConfig
from checkpoint_adapter import load_checkpoint_with_optimization
from PIL import Image
from torchvision import transforms

# Load optimized model
model = XCiTAF(cfg, ...)  # Your model config
model = load_checkpoint_with_optimization(
    model,
    'checkpoint.pth',
    optimize=True
)

# Set to evaluation mode
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('image.jpg')
input_tensor = transform(image).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    
# Get predictions
probabilities = torch.nn.functional.softmax(output, dim=1)
top5_prob, top5_idx = torch.topk(probabilities, 5)

print("Top 5 predictions:")
for i in range(5):
    print(f"  {i+1}. Class {top5_idx[0][i].item()}: {top5_prob[0][i].item():.4f}")
```

## Validation and Testing

### Test Module Equivalence

```python
# Verify your optimizations maintain correctness
python test_equivalence.py

# Expected output:
# ============================================================
# Testing DownSampleAF Optimization
# ============================================================
# ✓ All DownSampleAF tests passed
# ...
# ✅ ALL EQUIVALENCE TESTS PASSED!
```

### Test Checkpoint Compatibility

```python
# Verify checkpoint loading works
python checkpoint_adapter.py

# Expected output:
# Testing checkpoint compatibility...
# ✓ Checkpoint compatibility test passed
```

### Run Full Test Suite

```python
# Run all optimization tests
python test_optimizations.py

# Run equivalence tests
python test_equivalence.py

# Run advanced tests
python advanced_optimizations.py
```

## Performance Tips

### 1. Enable Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        output, _ = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. Use TorchScript for Inference

```python
# Trace the model for faster inference
model.eval()
example_input = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save('model_traced.pt')

# Load and use traced model
loaded_model = torch.jit.load('model_traced.pt')
output = loaded_model(example_input)
```

### 3. Optimize for Different Batch Sizes

The optimizations are most effective with:
- **Large batch sizes** (16-32+): Better GPU utilization
- **Small spatial dimensions** (after downsampling): Faster FFT operations
- **Multiple channels** (128+): Better parallelization

## Troubleshooting

### Issue: Optimizations slower than original

**Solution:** Make sure you're testing on GPU and with appropriate batch sizes. CPU performance may vary.

```python
# Verify GPU usage
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {next(model.parameters()).device}")
```

### Issue: Checkpoint loading fails

**Solution:** Ensure you're using the correct model architecture:

```python
# Check model architecture
print(model)

# Verify parameter shapes match
checkpoint = torch.load('checkpoint.pth')
for name, param in model.named_parameters():
    if name in checkpoint['model']:
        ckpt_shape = checkpoint['model'][name].shape
        if param.shape != ckpt_shape:
            print(f"Shape mismatch: {name} - model: {param.shape}, ckpt: {ckpt_shape}")
```

### Issue: Numerical differences from original

**Solution:** This is expected due to floating-point arithmetic. Check tolerances:

```python
# Verify differences are within tolerance
assert torch.allclose(output_orig, output_opt, rtol=1e-5, atol=1e-6)

# If needed, use looser tolerances for FP16
assert torch.allclose(output_orig, output_opt, rtol=1e-3, atol=1e-4)
```

## Summary of Optimizations

| Module | Optimization | Expected Speedup |
|--------|--------------|------------------|
| DownSampleAF | Fused FFT ops, in-place operations | 1.1-1.2x |
| UpSampleAF | Efficient padding, reduced allocations | 1.0-1.1x |
| PolyActPerChannel | Fast deg-2 path, Horner's method | 1.2-1.5x |
| LayerNormAF | Fused stats, rsqrt | 1.1x |
| FusedUpAct | Combined up-act-down | 1.2-2.0x |
| PolyActPerChannel_CUDA | Custom CUDA kernel | 3-5x (GPU) |
| LayerNormAF_TorchScript | JIT compilation | 1.1-1.2x |

**Overall model speedup:** 1.15-1.3x on CPU, 1.3-2.0x on GPU (varies by configuration)

## Next Steps

1. Benchmark your specific model configuration
2. Profile to identify remaining bottlenecks
3. Consider quantization for deployment (FP16/INT8)
4. Explore model distillation for smaller models
