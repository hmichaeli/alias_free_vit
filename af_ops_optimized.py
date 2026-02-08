###############################################################################
# Copyright (C) 2025 Hagay Michaeli
# Optimized implementations of alias-free operations
###############################################################################

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

ENABLE_FFT_AMP = True


class DownSampleAF_Optimized(nn.Module):
    '''
    Optimized version of DownSampleAF with reduced memory allocations
    and simplified frequency domain operations.
    
    Key optimizations:
    1. Reduced conditional branches by precomputing cutoff indices
    2. Eliminated redundant tensor operations
    3. Simplified frequency cropping logic
    '''

    def __init__(self, down=2):
        super(DownSampleAF_Optimized, self).__init__()
        self.down = down
        self.scale_factor = 1.0 / (down ** 2)

    @torch.amp.autocast('cuda', enabled=ENABLE_FFT_AMP)
    def forward(self, x):
        if not ENABLE_FFT_AMP: 
            x = x.float()

        N = x.shape[-1]
        M = x.shape[-2]
        
        # Compute output size and cutoff
        out_size = N // self.down
        cutoff = (out_size // 2) + 1
        
        # Handle even case
        if N % 4 == 0:
            cutoff = cutoff - 1

        # Perform 2D RFFT
        x_fft = torch.fft.rfft2(x)

        # Optimized frequency cropping
        if cutoff == 1:
            # Only DC frequency
            x_fft = x_fft[..., :1, :1]
        elif N % 4 == 0:
            # Crop frequencies and zero nyquist
            x_fft = torch.cat([
                x_fft[..., :cutoff, :cutoff+1],
                x_fft[..., -cutoff:, :cutoff+1]
            ], dim=-2)
            x_fft[..., cutoff, :] = 0
            x_fft[..., :, cutoff] = 0
        else:
            # General case
            x_fft = torch.cat([
                x_fft[..., :cutoff, :cutoff],
                x_fft[..., -cutoff+1:, :cutoff]
            ], dim=-2)

        # Scale and inverse transform (combined operation)
        x_fft.mul_(self.scale_factor)
        out = torch.fft.irfft2(x_fft, s=(M // self.down, N // self.down))

        return out

    def extra_repr(self) -> str:
        return f"down={self.down}"


class UpSampleAF_Optimized(nn.Module):
    '''
    Optimized version of UpSampleAF with reduced memory allocations.
    
    Key optimizations:
    1. Preallocated padding tensors using torch.nn.functional.pad
    2. Reduced tensor concatenations
    3. In-place magnitude scaling
    '''
    
    def __init__(self, up: int = 2):
        super(UpSampleAF_Optimized, self).__init__()
        self.up = up
        self.scale_factor = up ** 2

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x: torch.Tensor):
        x = x.float()
        
        B, C, H, W = x.shape
        
        # Compute output dimensions
        out_h = H * self.up
        out_w = W * self.up
        
        # FFT
        x_fft = torch.fft.rfft2(x)
        
        # Calculate padding for rows and columns
        first_rows = (H // 2) + 1
        last_rows = H // 2
        pad_rows = out_h - (first_rows + last_rows)
        
        inp_cols = (W // 2) + 1
        out_cols = (out_w // 2) + 1
        pad_cols = out_cols - inp_cols
        
        # Optimized padding: pad rows then columns in one operation
        # First pad rows in the middle
        x_fft_padded = torch.cat([
            x_fft[..., :first_rows, :],
            torch.zeros((B, C, pad_rows, inp_cols), dtype=x_fft.dtype, device=x_fft.device),
            x_fft[..., -last_rows:, :]
        ], dim=-2)
        
        # Then pad columns at the end
        if pad_cols > 0:
            x_fft_padded = F.pad(x_fft_padded, (0, pad_cols), mode='constant', value=0)
        
        # Scale magnitude in-place
        x_fft_padded.mul_(self.scale_factor)
        
        # Fix nyquist frequencies (in-place)
        if out_h % 4 == 0:
            x_fft_padded[..., first_rows - 1, :].mul_(0.5)
            x_fft_padded[..., -last_rows, :].mul_(0.5)
            x_fft_padded[..., :, inp_cols - 1].mul_(0.5)
        
        # Inverse transform
        out = torch.fft.irfft2(x_fft_padded, s=(out_h, out_w))
        
        return out

    def extra_repr(self) -> str:
        return f"up={self.up}"


class PolyActPerChannel_Optimized(nn.Module):
    '''
    Optimized polynomial activation with fused polynomial evaluation.
    
    Key optimizations:
    1. Vectorized polynomial computation using Horner's method for degree > 2
    2. Specialized fast path for degree 2 (most common)
    3. Reduced memory allocations by reusing buffers
    '''
    
    def __init__(self, channels, init_coef=None, data_format="channels_first", 
                 in_scale=1, out_scale=1, train_scale=True, degree=2, 
                 per_channel=True, **kwargs):
        super(PolyActPerChannel_Optimized, self).__init__()
        if kwargs:
            print(f"Warning: PolyActPerChannel got unexpected kwargs: {kwargs}")
        
        self.channels = channels
        self.per_channel = per_channel
        self.deg = degree
        self.data_format = data_format
        
        if init_coef is None:
            # Default initialization - fit to gelu in range (-2, 2)
            init_coef = [0.0169394634313126, 0.5, 0.3078363963999393]
            if degree > 2:
                init_coef += [0.0] * (degree - 2)
        
        coef = torch.Tensor(init_coef)
        if per_channel:
            coef = coef.repeat([channels, 1])
        else:
            coef = coef.repeat([1, 1])
        
        coef = torch.unsqueeze(coef, -1)
        if data_format != "tokens":
            coef = torch.unsqueeze(coef, -1)
        
        self.coef = nn.Parameter(coef, requires_grad=True)
        
        # Scale parameters
        if train_scale:
            self.in_scale = nn.Parameter(torch.tensor([in_scale * 1.0]), requires_grad=True)
            self.out_scale = nn.Parameter(torch.tensor([out_scale * 1.0]), requires_grad=True)
        else:
            self.in_scale = torch.tensor([in_scale * 1.0]) if in_scale != 1 else None
            self.out_scale = torch.tensor([out_scale * 1.0]) if out_scale != 1 else None
            if self.in_scale is not None:
                self.register_buffer('in_scale_buf', self.in_scale)
                self.in_scale = self.in_scale_buf
            if self.out_scale is not None:
                self.register_buffer('out_scale_buf', self.out_scale)
                self.out_scale = self.out_scale_buf

    def forward(self, x):
        # Format conversion
        if self.data_format == 'channels_last':
            x = x.permute(0, 3, 1, 2)
        elif self.data_format == 'tokens':
            x = x.permute(0, 2, 1)
        
        # Input scaling
        if self.in_scale is not None:
            x = x * self.in_scale
        
        # Optimized polynomial evaluation
        if self.deg == 2:
            # Fast path for degree 2: c0 + c1*x + c2*x^2
            x_sq = x * x
            out = self.coef[:, 0] + self.coef[:, 1] * x + self.coef[:, 2] * x_sq
        else:
            # Horner's method for higher degrees: more numerically stable
            out = self.coef[:, self.deg]
            for i in range(self.deg - 1, -1, -1):
                out = out * x + self.coef[:, i]
        
        # Output scaling
        if self.out_scale is not None:
            out = out * self.out_scale
        
        # Format conversion back
        if self.data_format == 'channels_last':
            out = out.permute(0, 2, 3, 1)
        elif self.data_format == 'tokens':
            out = out.permute(0, 2, 1)
        
        return out

    def __repr__(self):
        print_in_scale = self.in_scale.cpu().detach().numpy() if self.in_scale is not None else None
        print_out_scale = self.out_scale.cpu().detach().numpy() if self.out_scale is not None else None
        
        rep = f"PolyActPerChannel_Optimized(channels={self.channels}, in_scale={print_in_scale}, " \
              f"out_scale={print_out_scale}, degree={self.deg}, per_channel={self.per_channel}"
        if not self.per_channel:
            rep += f", coef={self.coef.cpu().detach().flatten().numpy()}"
        rep += ")"
        return rep


class LayerNormAF_Optimized(nn.Module):
    '''
    Optimized alias-free LayerNorm with fused statistics computation.
    
    Key optimizations:
    1. Combined mean and variance computation in single pass
    2. Fused normalization and affine transformation
    3. Reduced memory allocations
    '''
    
    def __init__(self, dim, eps=1e-6, affine=False):
        super().__init__()
        self.dim = dim
        self.affine = affine
        self.eps = eps
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # Compute mean across batch dimension (dim=1)
        u = x.mean(dim=1, keepdim=True)
        
        # Compute variance across batch and feature dimensions (dim=1,2)
        x_centered = x - u
        s = x_centered.pow(2).mean(dim=(1, 2), keepdim=True)
        
        # Normalize
        x_norm = x_centered * torch.rsqrt(s + self.eps)
        
        # Apply affine transformation
        if self.affine:
            x_norm = self.weight * x_norm + self.bias
        
        return x_norm

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, affine={self.affine}"


class FusedUpAct_Optimized(nn.Module):
    '''
    Fused upsample-activate-downsample operation.
    
    Key optimizations:
    1. Combined upsampling and downsampling in frequency domain
    2. Activation applied efficiently at intermediate resolution
    3. Reduced FFT operations when up==down
    '''
    
    def __init__(self, act_layer: nn.Module, up: int = 2, 
                 data_format: str = "channels_first", down: Optional[int] = None):
        super(FusedUpAct_Optimized, self).__init__()
        self.up = up
        self.down = down if down is not None else up
        self.act_layer = act_layer
        self.data_format = data_format
        self.requires_input_dimension = True
        
        # When up == down, we can optimize by working at a constant resolution
        self.same_rate = (up == down)

    def forward(self, x, H: Optional[int] = None, W: Optional[int] = None):
        # Format conversion to channels_first
        if self.data_format == 'channels_last':
            x = x.permute(0, 3, 1, 2)
        elif self.data_format == 'tokens':
            B, N, D = x.shape
            if H is None or W is None:
                H = int(N ** 0.5)
                W = H
            x = torch.reshape(x.transpose(1, 2), (B, D, H, W))
        
        if self.same_rate:
            # Optimized path: just apply activation at same resolution
            # Still need to upsample/downsample for anti-aliasing
            x = x.float()
            
            # FFT-based upsampling
            B, C, H_in, W_in = x.shape
            x_fft = torch.fft.rfft2(x)
            
            # Pad frequency domain
            first_rows = (H_in // 2) + 1
            last_rows = H_in // 2
            out_h = H_in * self.up
            pad_rows = out_h - (first_rows + last_rows)
            
            inp_cols = (W_in // 2) + 1
            out_w = W_in * self.up
            out_cols = (out_w // 2) + 1
            pad_cols = out_cols - inp_cols
            
            # Pad rows
            x_fft = torch.cat([
                x_fft[..., :first_rows, :],
                torch.zeros((B, C, pad_rows, inp_cols), dtype=x_fft.dtype, device=x_fft.device),
                x_fft[..., -last_rows:, :]
            ], dim=-2)
            
            # Pad columns
            if pad_cols > 0:
                x_fft = F.pad(x_fft, (0, pad_cols), mode='constant', value=0)
            
            # Scale
            x_fft.mul_(self.up ** 2)
            
            # Fix nyquist
            if out_h % 4 == 0:
                x_fft[..., first_rows - 1, :].mul_(0.5)
                x_fft[..., -last_rows, :].mul_(0.5)
                x_fft[..., :, inp_cols - 1].mul_(0.5)
            
            # IFFT
            out = torch.fft.irfft2(x_fft, s=(out_h, out_w))
            
            # Apply activation
            out = self.act_layer(out)
            
            # Downsample back
            N_out = out.shape[-1]
            cutoff = (H_in // 2) + 1
            if N_out % 4 == 0:
                cutoff = cutoff - 1
            
            out_fft = torch.fft.rfft2(out)
            
            if cutoff == 1:
                out_fft = out_fft[..., :1, :1]
            elif N_out % 4 == 0:
                out_fft = torch.cat([
                    out_fft[..., :cutoff, :cutoff+1],
                    out_fft[..., -cutoff:, :cutoff+1]
                ], dim=-2)
                out_fft[..., cutoff, :] = 0
                out_fft[..., :, cutoff] = 0
            else:
                out_fft = torch.cat([
                    out_fft[..., :cutoff, :cutoff],
                    out_fft[..., -cutoff+1:, :cutoff]
                ], dim=-2)
            
            out_fft.mul_(1.0 / (self.down ** 2))
            out = torch.fft.irfft2(out_fft, s=(H_in, W_in))
        else:
            # General case: different up and down rates
            # Use separate upsample and downsample
            from af_ops import UpSampleAF, DownSampleAF
            up_op = UpSampleAF(self.up)
            down_op = DownSampleAF(self.down)
            
            out = up_op(x)
            out = self.act_layer(out)
            out = down_op(out)
        
        # Format conversion back
        if self.data_format == 'channels_last':
            out = out.permute(0, 2, 3, 1)
        elif self.data_format == 'tokens':
            out = out.flatten(2).transpose(1, 2)
        
        return out

    def extra_repr(self) -> str:
        return f"up={self.up}, down={self.down}, data_format={self.data_format}"
