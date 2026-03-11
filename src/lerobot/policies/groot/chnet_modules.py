"""CHNet-style depth processing modules for GR00T.

Implements FastGuide cross-modal guidance from CHNet (arXiv:2401.15902)
adapted for use with a ViT vision backbone. RGB features from intermediate
ViT layers multiplicatively modulate depth features processed by a parallel
CNN encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Basic2d(nn.Module):
    """Conv2d + optional BatchNorm + ReLU."""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,
                 kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, bias=(norm_layer is None))
        self.bn = norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """Standard ResNet basic block (2x conv3x3 + residual)."""

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class FastGuide(nn.Module):
    """Fast Guidance Module from CHNet.

    RGB features multiplicatively modulate depth features via:
    1. Channel-wise: expand RGB to N sub-weights, multiply with depth, sum
    2. Cross-channel: mean across expanded channels as spatial attention
    """

    def __init__(self, channels, expansion_ratio=3):
        super().__init__()
        self.expansion_ratio = expansion_ratio
        # No BN on first conv (per CHNet)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.weight_expansion = Basic2d(channels, channels * expansion_ratio,
                                        kernel_size=1, padding=0)
        self.conv2 = Basic2d(channels, channels, kernel_size=1, padding=0)
        self.conv3 = Basic2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, depth_feat, rgb_feat, return_diagnostics=False):
        """
        Args:
            depth_feat: (B, C, H, W) depth features from CNN encoder
            rgb_feat: (B, C, H, W) RGB features from ViT (projected + resized)
            return_diagnostics: if True, also return spatial attention map
        Returns:
            (B, C, H, W) guided depth features
            If return_diagnostics: tuple of (guided_features, avg_attn)
        """
        weight = self.conv1(rgb_feat)
        weight = self.weight_expansion(weight)  # (B, 3*C, H, W)

        chunks = torch.chunk(weight, self.expansion_ratio, dim=1)
        out = sum(depth_feat * chunk for chunk in chunks)
        out = self.conv2(out)

        avg_attn = weight.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        out = self.conv3(out * avg_attn)
        if return_diagnostics:
            return out, avg_attn
        return out


class ViTFeatureProjector(nn.Module):
    """Projects ViT token features to 2D feature maps for FastGuide."""

    def __init__(self, vit_dim, out_channels, target_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(vit_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.target_size = target_size

    def forward(self, vit_tokens, grid_h, grid_w):
        """
        Args:
            vit_tokens: (B, N, D) where N = grid_h * grid_w
            grid_h, grid_w: spatial grid dimensions
        Returns:
            (B, out_channels, target_size, target_size)
        """
        b, n, d = vit_tokens.shape
        x = vit_tokens.transpose(1, 2).reshape(b, d, grid_h, grid_w)
        x = self.proj(x)
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode='bilinear', align_corners=False)
        return x


def _make_layer(inplanes, planes, blocks=2, stride=1):
    """Build a sequence of BasicBlocks with optional downsampling."""
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )
    layers = [BasicBlock(inplanes, planes, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))
    return nn.Sequential(*layers)


class DepthCNNEncoder(nn.Module):
    """4-stage CNN encoder for depth with FastGuide at each stage."""

    def __init__(self, vit_dim=1024, channels=(64, 128, 256, 256)):
        super().__init__()
        c0, c1, c2, c3 = channels
        stem_ch = c0 // 2  # 32

        # Stem: (1, 224, 224) -> (32, 112, 112)
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_ch, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )

        # Encoder stages
        self.stage1 = _make_layer(stem_ch, c0, blocks=2, stride=2)  # -> (64, 56, 56)
        self.stage2 = _make_layer(c0, c1, blocks=2, stride=2)       # -> (128, 28, 28)
        self.stage3 = _make_layer(c1, c2, blocks=2, stride=2)       # -> (256, 14, 14)
        self.stage4 = _make_layer(c2, c3, blocks=2, stride=2)       # -> (256, 7, 7)

        # FastGuide modules (one per stage)
        self.guide1 = FastGuide(c0)
        self.guide2 = FastGuide(c1)
        self.guide3 = FastGuide(c2)
        self.guide4 = FastGuide(c3)

        # ViT feature projectors (one per stage)
        self.proj1 = ViTFeatureProjector(vit_dim, c0, target_size=56)
        self.proj2 = ViTFeatureProjector(vit_dim, c1, target_size=28)
        self.proj3 = ViTFeatureProjector(vit_dim, c2, target_size=14)
        self.proj4 = ViTFeatureProjector(vit_dim, c3, target_size=7)

        self.out_channels = c3

    def forward(self, depth, vit_features, grid_h, grid_w, return_diagnostics=False):
        """
        Args:
            depth: (B, 1, H, W) normalized depth image
            vit_features: list of 4 tensors, each (B, N, D) from ViT layers
            grid_h, grid_w: ViT patch grid dimensions
            return_diagnostics: if True, also return FastGuide spatial attention maps
        Returns:
            (B, out_channels, 7, 7) depth feature map
            If return_diagnostics: tuple of (depth_features, fastguide_attns)
        """
        fastguide_attns = [] if return_diagnostics else None
        vit_projected = [] if return_diagnostics else None
        depth_stages = [] if return_diagnostics else None
        x = self.stem(depth)

        x = self.stage1(x)
        rgb1 = self.proj1(vit_features[0], grid_h, grid_w)
        if return_diagnostics:
            depth_stages.append(x.clone())
            vit_projected.append(rgb1)
            x, attn = self.guide1(x, rgb1, return_diagnostics=True)
            fastguide_attns.append(attn)
        else:
            x = self.guide1(x, rgb1)

        x = self.stage2(x)
        rgb2 = self.proj2(vit_features[1], grid_h, grid_w)
        if return_diagnostics:
            depth_stages.append(x.clone())
            vit_projected.append(rgb2)
            x, attn = self.guide2(x, rgb2, return_diagnostics=True)
            fastguide_attns.append(attn)
        else:
            x = self.guide2(x, rgb2)

        x = self.stage3(x)
        rgb3 = self.proj3(vit_features[2], grid_h, grid_w)
        if return_diagnostics:
            depth_stages.append(x.clone())
            vit_projected.append(rgb3)
            x, attn = self.guide3(x, rgb3, return_diagnostics=True)
            fastguide_attns.append(attn)
        else:
            x = self.guide3(x, rgb3)

        x = self.stage4(x)
        rgb4 = self.proj4(vit_features[3], grid_h, grid_w)
        if return_diagnostics:
            depth_stages.append(x.clone())
            vit_projected.append(rgb4)
            x, attn = self.guide4(x, rgb4, return_diagnostics=True)
            fastguide_attns.append(attn)
        else:
            x = self.guide4(x, rgb4)

        if return_diagnostics:
            return x, fastguide_attns, vit_projected, depth_stages
        return x


class DepthCrossAttentionFusion(nn.Module):
    """Cross-attention to fuse depth features into ViT token sequence."""

    def __init__(self, depth_channels, hidden_dim, num_heads=8):
        super().__init__()
        self.depth_proj = nn.Linear(depth_channels, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        # Zero-init output projection so depth branch starts as identity.
        # Combined with pre-norm residual (x + norm(attn_out)), this ensures
        # change_ratio starts near 0 and depth signal blends in gradually.
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, eagle_features, depth_features, return_diagnostics=False,
                n_image_tokens=None):
        """
        Args:
            eagle_features: (B, seq_len, hidden_dim) from Eagle model
            depth_features: (N, C, H, W) from DepthCNNEncoder, where N = B * num_views
            return_diagnostics: if True, also return attention weights and depth tokens
            n_image_tokens: if provided and < seq_len, only cross-attend image tokens
                            (text tokens are passed through unchanged)
        Returns:
            (B, seq_len, hidden_dim) enriched features
            If return_diagnostics: tuple of (enriched_features, attn_weights, depth_tokens)
        """
        n, c, h, w = depth_features.shape
        b_eagle = eagle_features.shape[0]
        seq_len = eagle_features.shape[1]
        depth_tokens = depth_features.flatten(2).transpose(1, 2)  # (N, H*W, C)
        depth_tokens = self.depth_proj(depth_tokens)  # (N, H*W, hidden_dim)

        # Handle multi-view: when using multiple cameras, depth has N = B * num_views
        # entries but eagle_features has only B. Merge view tokens so batch dims match.
        if n != b_eagle:
            num_views = n // b_eagle
            tokens_per_view = depth_tokens.shape[1]
            # (B*V, T, D) -> (B, V*T, D) — each batch element attends to all its views
            depth_tokens = depth_tokens.view(b_eagle, num_views * tokens_per_view, -1)

        # Slice image tokens from the full eagle sequence when n_image_tokens is given.
        # Text tokens have no spatial correspondence to depth and produce uniform
        # attention, so we only use the image portion as Q in cross-attention.
        if n_image_tokens is not None and n_image_tokens < seq_len:
            image_tokens = eagle_features[:, :n_image_tokens, :]
            text_tokens = eagle_features[:, n_image_tokens:, :]
        else:
            image_tokens = eagle_features
            text_tokens = None

        # Cross-attention: Q=image_tokens, K=depth, V=depth
        if return_diagnostics:
            attn_out, attn_weights = self.cross_attn(
                query=image_tokens,
                key=depth_tokens,
                value=depth_tokens,
                need_weights=True,
                average_attn_weights=False,  # keep per-head weights
            )
            enriched_image = image_tokens + self.norm(attn_out)
            if text_tokens is not None:
                result = torch.cat([enriched_image, text_tokens], dim=1)
            else:
                result = enriched_image
            return result, attn_weights, depth_tokens
        else:
            attn_out, _ = self.cross_attn(
                query=image_tokens,
                key=depth_tokens,
                value=depth_tokens,
            )
            enriched_image = image_tokens + self.norm(attn_out)
            if text_tokens is not None:
                return torch.cat([enriched_image, text_tokens], dim=1)
            return enriched_image


class CHNetDepthProcessor(nn.Module):
    """Orchestrates the full CHNet depth processing pipeline.

    Combines DepthCNNEncoder + DepthCrossAttentionFusion.
    """

    def __init__(self, vit_dim=1024, hidden_dim=2048, channels=(64, 128, 256, 256),
                 num_heads=8):
        super().__init__()
        self.encoder = DepthCNNEncoder(vit_dim=vit_dim, channels=channels)
        self.fusion = DepthCrossAttentionFusion(
            depth_channels=channels[-1],
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )

    def forward(self, depth, vit_features, grid_h, grid_w, eagle_features,
                return_diagnostics=False, n_image_tokens=None):
        """
        Args:
            depth: (B, 1, H, W) normalized depth
            vit_features: list of 4 tensors from ViT intermediate layers
            grid_h, grid_w: ViT patch grid dimensions
            eagle_features: (B, seq_len, hidden_dim) pre-projection features
            return_diagnostics: if True, also return intermediate tensors for visualization
            n_image_tokens: if provided, only cross-attend image tokens in fusion
        Returns:
            (B, seq_len, hidden_dim) enriched features
            If return_diagnostics: tuple of (enriched_features, diagnostics_dict)
        """
        if return_diagnostics:
            depth_feat, fastguide_attns, vit_projected, depth_stages = self.encoder(
                depth, vit_features, grid_h, grid_w, return_diagnostics=True
            )
            result, attn_weights, depth_tokens = self.fusion(
                eagle_features, depth_feat, return_diagnostics=True,
                n_image_tokens=n_image_tokens,
            )
            return result, {
                "attn_weights": attn_weights,
                "fastguide_attns": fastguide_attns,
                "depth_tokens": depth_tokens,
                "vit_projected": vit_projected,
                "depth_stages": depth_stages,
            }
        else:
            depth_feat = self.encoder(depth, vit_features, grid_h, grid_w)
            return self.fusion(eagle_features, depth_feat, n_image_tokens=n_image_tokens)
