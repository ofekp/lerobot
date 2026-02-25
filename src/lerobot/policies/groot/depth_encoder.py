# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Depth branch encoder for GR00T-N1.5.

This module implements a separate depth processing branch that produces patch embeddings
compatible with the RGB SigLIP2 patch embeddings (dim=1152, patch_size=14).

Architecture:
    depth (1ch, meters) 
    → Fourier positional encoding per pixel (dim d, e.g. 64)
    → 2-layer Conv2d CNN (d → 1152, stride=14 overall to match patch_size)
    → zero-initialized 1×1 conv gate (1152 → 1152)
    → element-wise sum with RGB patch embeddings

The zero-init gate ensures that at initialization, the depth branch contributes
nothing, preserving pretrained RGB performance and enabling smooth learning.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierPositionalEncoding(nn.Module):
    """Fourier positional encoding for scalar depth values.

    Maps each scalar depth value z ∈ ℝ to a vector of dimension `encoding_dim`
    using sinusoidal functions with log-linearly spaced frequencies:

        out[2i]   = sin(2π · f_i · z)
        out[2i+1] = cos(2π · f_i · z)

    where f_i = sigma^(i / (n_freqs - 1)) for i = 0, ..., n_freqs - 1.

    Args:
        encoding_dim: Output dimension. Must be even. Half will be sin, half cos.
        min_freq: Minimum frequency (default 1.0).
        max_freq: Maximum frequency (default 50.0). For 0-3m depth, 50 Hz gives
            ~2cm wavelength — fine detail without amplifying sensor noise.
        learnable: If True, frequencies are learnable parameters.
    """

    def __init__(
        self,
        encoding_dim: int = 64,
        min_freq: float = 1.0,
        max_freq: float = 50.0,
        learnable: bool = False,
    ):
        super().__init__()
        assert encoding_dim % 2 == 0, f"encoding_dim must be even, got {encoding_dim}"
        self.encoding_dim = encoding_dim
        n_freqs = encoding_dim // 2  # sin and cos pairs

        # Log-linearly spaced frequencies
        freqs = torch.exp(
            torch.linspace(math.log(min_freq), math.log(max_freq), n_freqs)
        )  # (n_freqs,)

        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: (B, 1, H, W) depth map in meters.

        Returns:
            (B, encoding_dim, H, W) Fourier-encoded depth.
        """
        # depth: (B, 1, H, W) → (B, 1, H, W, 1)
        # freqs: (n_freqs,) → (1, 1, 1, 1, n_freqs)
        d = depth.unsqueeze(-1)  # (B, 1, H, W, 1)
        f = self.freqs.view(1, 1, 1, 1, -1)  # (1, 1, 1, 1, n_freqs)

        # Phase: 2π · f · z
        phase = 2.0 * math.pi * f * d  # (B, 1, H, W, n_freqs)

        # Sin and cos
        sin_enc = torch.sin(phase)  # (B, 1, H, W, n_freqs)
        cos_enc = torch.cos(phase)  # (B, 1, H, W, n_freqs)

        # Concatenate sin and cos: (B, 1, H, W, encoding_dim)
        enc = torch.cat([sin_enc, cos_enc], dim=-1)

        # Reshape to (B, encoding_dim, H, W)
        B, _, H, W, D = enc.shape
        enc = enc.squeeze(1).permute(0, 3, 1, 2)  # (B, encoding_dim, H, W)

        return enc


class DepthPatchEmbedding(nn.Module):
    """Separate CNN branch that converts Fourier-encoded depth maps to patch embeddings.

    This module mirrors the RGB patch embedding's output:
    - Takes (B, fourier_dim, H, W) as input
    - Outputs (N_patches, embed_dim) — one embedding vector per 14×14 patch
    - Uses a 2-layer Conv2d CNN with an overall stride of 14 to match the patch size

    The CNN architecture uses stride-based downsampling to go from pixel-level
    Fourier features to patch-level embeddings:
        Layer 1: 7×7 conv, stride 2  → 2× reduction + spatial context
        Layer 2: 7×7 conv, stride 7  → 7× reduction (total: 2×7 = 14 = patch_size)

    After the CNN, a zero-initialized 1×1 conv (linear projection) gates the output
    so the depth branch starts contributing nothing and smoothly learns.

    Args:
        fourier_dim: Input channels (from Fourier encoding).
        embed_dim: Output embedding dimension (must match RGB patch embed, default 1152).
        patch_size: Patch size to match (default 14).
        hidden_dim: Hidden channels in intermediate CNN layers.
    """

    def __init__(
        self,
        fourier_dim: int = 64,
        embed_dim: int = 1152,
        patch_size: int = 14,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # 2-layer CNN: overall stride = 2 * 7 = 14 = patch_size
        # Layer 1: 7×7 conv, stride 2, captures local spatial context
        # Layer 2: 7×7 conv, stride 7, projects to embed_dim at patch resolution
        self.cnn = nn.Sequential(
            # Layer 1: (B, fourier_dim, H, W) → (B, hidden_dim, H/2, W/2)
            nn.Conv2d(fourier_dim, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.GroupNorm(num_groups=min(32, hidden_dim), num_channels=hidden_dim),
            # Layer 2: (B, hidden_dim, H/2, W/2) → (B, embed_dim, H/14, W/14)
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=7, stride=7, padding=0),
        )

        # Zero-initialized gate: 1×1 conv (acts per-patch)
        # Initialized to zero so depth branch contributes nothing at start.
        # NOTE: HuggingFace's from_pretrained may use meta-device / no_init_weights
        # which can defeat any initialization done in __init__. The authoritative
        # zero-init is done via zero_init_gate() called after model construction.
        self.gate = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, fourier_depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fourier_depth: (B, fourier_dim, H, W) — Fourier-encoded depth map.
                H, W must be divisible by patch_size (14).

        Returns:
            (B, n_patches_h * n_patches_w, embed_dim) — one embedding per patch,
            matching the output format of the standard SiglipVisionEmbeddings.
        """
        B, C, H, W = fourier_depth.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"Input spatial dims ({H}, {W}) must be divisible by patch_size={self.patch_size}"
        )

        # CNN: (B, fourier_dim, H, W) → (B, embed_dim, H/14, W/14)
        x = self.cnn(fourier_depth)

        # Zero-init gate
        x = self.gate(x)

        # Reshape to match SiglipVisionEmbeddings output format:
        # (B, embed_dim, nH, nW) → (B, nH*nW, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        return x


class DepthBranchEncoder(nn.Module):
    """Complete depth branch: Fourier encoding → CNN → zero-init gate.

    This is the top-level module that combines FourierPositionalEncoding
    and DepthPatchEmbedding into a single depth processing branch.

    Args:
        fourier_dim: Dimension of the Fourier positional encoding (default 64).
        embed_dim: Output embedding dimension, must match RGB patch embed (default 1152).
        patch_size: Patch size to match RGB (default 14).
        hidden_dim: Hidden channels in the CNN (default 256).
        min_freq: Minimum Fourier frequency.
        max_freq: Maximum Fourier frequency.
        learnable_freqs: Whether Fourier frequencies are learnable.
    """

    def __init__(
        self,
        fourier_dim: int = 64,
        embed_dim: int = 1152,
        patch_size: int = 14,
        hidden_dim: int = 256,
        min_freq: float = 1.0,
        max_freq: float = 100.0,
        learnable_freqs: bool = False,
    ):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.fourier_encoding = FourierPositionalEncoding(
            encoding_dim=fourier_dim,
            min_freq=min_freq,
            max_freq=max_freq,
            learnable=learnable_freqs,
        )

        self.patch_embedding = DepthPatchEmbedding(
            fourier_dim=fourier_dim,
            embed_dim=embed_dim,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
        )

    def reset_parameters(self):
        """Explicitly initialize all depth branch weights.

        When training, this must be called AFTER from_pretrained / model construction completes,
        because HuggingFace's `no_init_weights()` context replaces all
        `torch.nn.init.*` functions with no-ops, leaving weights as
        uninitialized `torch.empty()` garbage.  This method uses direct
        `.data` tensor operations so it works regardless of that context.

        Includes zero-init of the gate (previously in zero_init_gate).
        """
        # --- CNN layers: Kaiming uniform (same as Conv2d.reset_parameters) ---
        for module in self.patch_embedding.cnn.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    torch.nn.init.uniform_(module.bias, -bound, bound)

        # --- Gate: zero-init so depth contributes nothing at start ---
        self.patch_embedding.gate.weight.data.zero_()
        self.patch_embedding.gate.bias.data.zero_()

        gate_norm = self.patch_embedding.gate.weight.data.norm().item()
        cnn_norm = sum(
            m.weight.data.norm().item()
            for m in self.patch_embedding.cnn.modules()
            if isinstance(m, nn.Conv2d)
        )
        print(
            f"[GROOT] Depth branch reset_parameters: "
            f"CNN weight norm={cnn_norm:.4f}, gate weight norm={gate_norm}"
        )

    def zero_init_gate(self):
        """Zero-initialize the gate weights and biases.

        Must be called AFTER from_pretrained / model construction completes,
        because HuggingFace's from_pretrained uses meta-device / no_init_weights
        contexts that can defeat any initialization done inside __init__.
        """
        self.patch_embedding.gate.weight.data.zero_()
        self.patch_embedding.gate.bias.data.zero_()
        gate_norm = self.patch_embedding.gate.weight.data.norm().item()
        print(f"[GROOT] Depth gate zero-initialized (weight norm: {gate_norm})")

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: (B, 1, H, W) depth map in meters.

        Returns:
            (B, nH * nW, embed_dim) patch embeddings from depth,
            ready to be added to RGB patch embeddings.
        """
        fourier_features = self.fourier_encoding(depth)  # (B, fourier_dim, H, W)
        patch_embeds = self.patch_embedding(fourier_features)  # (B, nH*nW, embed_dim)
        return patch_embeds
