"""Voxel-based 3D encoder for RGB-D observations.

Provides two components:
  - VoxelGrid: back-projects RGB-D images into discretised 3D voxel grids
    (pure tensor ops, no learnable parameters).
  - VoxelEncoder: wraps VoxelGrid with a 3D-CNN that compresses the voxel
    representation into a flat sequence of tokens suitable for transformer
    consumption.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Module-level cache for the most recent voxel grid (used by eval rendering).
# Set by VoxelEncoder.forward(); read by LiberoEnv.render().
_voxel_grid_cache: torch.Tensor | None = None


class VoxelGrid:
    """Back-projects RGB-D images to 3D voxel grids.

    This is a stateless callable (not an ``nn.Module``).  It converts batched
    depth + RGB observations into a dense ``(B, 4, G, G, G)`` tensor where the
    four channels are R, G, B and binary occupancy.

    Args:
        grid_size: Number of voxels along each spatial axis.
        workspace_bounds: Per-axis (min, max) bounds defining the volume of
            interest in world coordinates.  Provided as a tuple of three
            ``(lo, hi)`` pairs for X, Y, Z respectively.
    """

    def __init__(
        self,
        grid_size: int = 40,
        workspace_bounds: tuple[tuple[float, float], ...] = (
            (-0.3, 0.3),
            (-0.3, 0.3),
            (0.6, 1.0),
        ),
    ) -> None:
        self.grid_size = grid_size
        self.workspace_bounds = workspace_bounds

    # ------------------------------------------------------------------
    # Back-projection
    # ------------------------------------------------------------------

    def backproject(
        self,
        depth: torch.Tensor,
        rgb: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unproject depth pixels into world-frame 3D points.

        Args:
            depth: ``(B, 1, H, W)`` depth map in metres.
            rgb: ``(B, 3, H, W)`` colour image in ``[0, 1]``.
            intrinsics: ``(B, 3, 3)`` camera intrinsic matrices.
            extrinsics: ``(B, 4, 4)`` world-to-camera rigid transforms.

        Returns:
            points_world: ``(B, N, 3)`` 3D points in world frame.
            colors: ``(B, N, 3)`` per-point RGB values.
            valid_mask: ``(B, N)`` boolean mask (``True`` where depth is valid).
        """
        B, _, H, W = depth.shape
        device = depth.device

        # Build pixel coordinate grid  --  (H, W) each
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=depth.dtype),
            torch.arange(W, device=device, dtype=depth.dtype),
            indexing="ij",
        )

        # Flatten to (N,) where N = H * W
        u_flat = u_coords.reshape(-1)  # (N,)
        v_flat = v_coords.reshape(-1)  # (N,)
        ones = torch.ones_like(u_flat)  # (N,)

        # Homogeneous pixel coordinates: (3, N)
        pixel_coords = torch.stack([u_flat, v_flat, ones], dim=0)  # (3, N)

        # Depth per pixel: (B, N)
        depth_flat = depth.reshape(B, -1)  # (B, N)

        # Camera-frame 3D points: p_cam = K^{-1} * [u, v, 1]^T * d
        K_inv = torch.inverse(intrinsics)  # (B, 3, 3)
        # (B, 3, 3) @ (3, N) -> (B, 3, N)
        rays = K_inv @ pixel_coords.unsqueeze(0).expand(B, -1, -1)
        points_cam = rays * depth_flat.unsqueeze(1)  # (B, 3, N)

        # Transform to world frame.  extrinsics is world-to-camera, so we
        # need its inverse (camera-to-world).
        cam_to_world = torch.inverse(extrinsics)  # (B, 4, 4)
        R = cam_to_world[:, :3, :3]  # (B, 3, 3)
        t = cam_to_world[:, :3, 3:]  # (B, 3, 1)
        points_world = R @ points_cam + t  # (B, 3, N)
        points_world = points_world.permute(0, 2, 1)  # (B, N, 3)

        # Per-point colours: (B, 3, H, W) -> (B, N, 3)
        colors = rgb.reshape(B, 3, -1).permute(0, 2, 1)  # (B, N, 3)

        # Validity mask: reject depth values outside [0.01, 10.0] metres
        valid_mask = (depth_flat > 0.01) & (depth_flat < 10.0)  # (B, N)

        return points_world, colors, valid_mask

    # ------------------------------------------------------------------
    # Voxelisation
    # ------------------------------------------------------------------

    def voxelize(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter 3D points into a dense voxel grid.

        Args:
            points: ``(B, N, 3)`` world-frame point positions.
            colors: ``(B, N, 3)`` per-point RGB in ``[0, 1]``.
            valid_mask: ``(B, N)`` boolean validity mask.

        Returns:
            voxel_grid: ``(B, 4, G, G, G)`` with channels R, G, B, occupancy.
                RGB channels contain the average colour of all points that
                fall into a given voxel; occupancy is 1 wherever at least one
                valid point lands.
        """
        B, N, _ = points.shape
        G = self.grid_size
        device = points.device
        dtype = points.dtype

        # Workspace limits
        bounds = torch.tensor(self.workspace_bounds, device=device, dtype=dtype)  # (3, 2)
        lo = bounds[:, 0]  # (3,)
        hi = bounds[:, 1]  # (3,)

        # Normalise points to [0, 1] within workspace bounds
        normed = (points - lo) / (hi - lo)  # (B, N, 3)

        # Convert to voxel indices in [0, G-1]
        voxel_idx = (normed * G).long()  # (B, N, 3)

        # Points inside the workspace volume
        in_bounds = (
            (voxel_idx[..., 0] >= 0)
            & (voxel_idx[..., 0] < G)
            & (voxel_idx[..., 1] >= 0)
            & (voxel_idx[..., 1] < G)
            & (voxel_idx[..., 2] >= 0)
            & (voxel_idx[..., 2] < G)
        )  # (B, N)
        mask = valid_mask & in_bounds  # (B, N)

        # Allocate accumulation buffers
        rgb_sum = torch.zeros(B, G, G, G, 3, device=device, dtype=dtype)
        counts = torch.zeros(B, G, G, G, 1, device=device, dtype=dtype)

        # Linearise voxel indices for scatter_add:  flat = x*G*G + y*G + z
        flat_idx = (
            voxel_idx[..., 0] * (G * G)
            + voxel_idx[..., 1] * G
            + voxel_idx[..., 2]
        )  # (B, N)

        # Zero out invalid / out-of-bounds entries so they don't pollute
        flat_idx = flat_idx.clamp(0, G * G * G - 1)  # safety clamp
        masked_colors = colors * mask.unsqueeze(-1).float()  # (B, N, 3)
        masked_ones = mask.unsqueeze(-1).float()  # (B, N, 1)

        # Expand flat_idx for the feature dimension
        flat_idx_rgb = flat_idx.unsqueeze(-1).expand(-1, -1, 3)  # (B, N, 3)
        flat_idx_cnt = flat_idx.unsqueeze(-1)  # (B, N, 1)

        rgb_sum_flat = rgb_sum.reshape(B, G * G * G, 3)
        counts_flat = counts.reshape(B, G * G * G, 1)

        rgb_sum_flat.scatter_add_(1, flat_idx_rgb, masked_colors)
        counts_flat.scatter_add_(1, flat_idx_cnt, masked_ones)

        rgb_sum = rgb_sum_flat.reshape(B, G, G, G, 3)
        counts = counts_flat.reshape(B, G, G, G, 1)

        # Average RGB where counts > 0
        safe_counts = counts.clamp(min=1.0)
        rgb_avg = rgb_sum / safe_counts  # (B, G, G, G, 3)

        # Occupancy: 1 if any point landed in the voxel
        occupancy = (counts > 0).float()  # (B, G, G, G, 1)

        # Stack into (B, G, G, G, 4) then move channels first -> (B, 4, G, G, G)
        voxel_grid = torch.cat([rgb_avg, occupancy], dim=-1)  # (B, G, G, G, 4)
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).contiguous()  # (B, 4, G, G, G)

        return voxel_grid

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __call__(
        self,
        depth: torch.Tensor,
        rgb: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Back-project and voxelise in one step.

        Args:
            depth: ``(B, 1, H, W)`` depth in metres.
            rgb: ``(B, 3, H, W)`` colour in ``[0, 1]``.
            intrinsics: ``(B, 3, 3)`` camera intrinsics.
            extrinsics: ``(B, 4, 4)`` world-to-camera transforms.

        Returns:
            ``(B, 4, G, G, G)`` voxel grid with R, G, B, occupancy channels.
        """
        points_world, colors, valid_mask = self.backproject(
            depth, rgb, intrinsics, extrinsics
        )
        return self.voxelize(points_world, colors, valid_mask)


class VoxelEncoder(nn.Module):
    """3D-CNN encoder that converts RGB-D observations to token sequences.

    Internally uses :class:`VoxelGrid` to produce a dense ``(B, 4, G, G, G)``
    representation which is then processed by a small 3D convolutional network.
    The spatial feature map is flattened to a sequence of tokens and linearly
    projected to the desired hidden dimension.

    With the default ``grid_size=40`` the encoder outputs 125 tokens
    (``(40 / 8)^3``), each of dimension ``hidden_dim``.

    Args:
        grid_size: Number of voxels per spatial axis.
        in_channels: Number of input channels in the voxel grid (default 4:
            R, G, B, occupancy).
        hidden_dim: Dimension of each output token.
        workspace_bounds: Physical workspace limits forwarded to
            :class:`VoxelGrid`.
    """

    def __init__(
        self,
        grid_size: int = 40,
        in_channels: int = 4,
        hidden_dim: int = 1536,
        workspace_bounds: tuple[tuple[float, float], ...] = (
            (-0.3, 0.3),
            (-0.3, 0.3),
            (0.6, 1.0),
        ),
    ) -> None:
        super().__init__()

        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.num_tokens = (grid_size // 8) ** 3  # 125 for grid_size=40

        # Non-learnable voxelisation front-end
        self.voxel_grid = VoxelGrid(grid_size, workspace_bounds)

        # 3D convolutional encoder
        # Spatial downsampling: 40 -> 40 -> 20 -> 10 -> 5
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        # Project from CNN channel dim to transformer hidden dim
        self.projection = nn.Linear(256, hidden_dim)

    def forward(
        self,
        depth: torch.Tensor,
        rgb: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Encode RGB-D observations into a token sequence.

        Args:
            depth: ``(B, 1, H, W)`` depth in metres.
            rgb: ``(B, 3, H, W)`` colour in ``[0, 1]``.
            intrinsics: ``(B, 3, 3)`` camera intrinsics.
            extrinsics: ``(B, 4, 4)`` world-to-camera transforms.

        Returns:
            ``(B, num_tokens, hidden_dim)`` token tensor.  With default
            settings this is ``(B, 125, 1536)``.
        """
        B = depth.shape[0]

        # 1. Build voxel grid  (B, 4, G, G, G)
        voxels = self.voxel_grid(depth, rgb, intrinsics, extrinsics)

        # Cache first sample's voxel grid for visualization (detached, CPU)
        self._last_voxel_grid = voxels[0].detach().cpu()
        # Also store in module-level cache for cross-module access (eval rendering)
        global _voxel_grid_cache
        _voxel_grid_cache = self._last_voxel_grid

        # 2. Encode via 3D CNN  (B, 256, G/8, G/8, G/8)
        features = self.encoder(voxels)

        # 3. Reshape spatial dims into token sequence  (B, num_tokens, 256)
        tokens = features.reshape(B, 256, -1).permute(0, 2, 1)

        # 4. Project to hidden dim  (B, num_tokens, hidden_dim)
        tokens = self.projection(tokens)

        return tokens


def render_voxel_projections(
    voxel_grid: torch.Tensor,
    scale: int = 4,
) -> np.ndarray:
    """Render three orthographic projections of a voxel grid as a single image.

    Takes a ``(4, G, G, G)`` tensor (R, G, B, occupancy) and produces a
    composite image with three views side-by-side:
      - **XY** (top-down, max over Z)
      - **XZ** (front, max over Y)
      - **YZ** (side, max over X)

    Occupied voxels are shown with their RGB colour; empty space is dark grey.

    Args:
        voxel_grid: ``(4, G, G, G)`` tensor on CPU.
        scale: Integer upscale factor for visibility (default 4).

    Returns:
        ``(H, W, 3)`` uint8 numpy array suitable for saving as PNG.
    """
    grid = voxel_grid.float()  # (4, G, G, G)
    rgb = grid[:3]             # (3, G, G, G)
    occ = grid[3:4]            # (1, G, G, G)

    bg = 0.15  # dark grey background value

    def _project(dim: int) -> np.ndarray:
        """Max-project occupancy-weighted RGB along *dim* (spatial axis)."""
        # For each pixel in the projection, pick the voxel with max occupancy
        # along the projection axis. Where multiple voxels are occupied, we
        # average the RGB.
        occ_proj = occ.max(dim=dim + 1).values  # (1, G, G) — +1 for channel dim
        rgb_sum = (rgb * occ).sum(dim=dim + 1)   # (3, G, G)
        occ_count = occ.sum(dim=dim + 1).clamp(min=1.0)  # (1, G, G)
        rgb_avg = rgb_sum / occ_count             # (3, G, G)

        # Blend with background
        img = rgb_avg * occ_proj + bg * (1.0 - occ_proj)  # (3, G, G)
        img = img.clamp(0.0, 1.0).permute(1, 2, 0)  # (G, G, 3)
        return (img.numpy() * 255).astype(np.uint8)

    # Project along Z (top-down XY), Y (front XZ), X (side YZ)
    proj_xy = _project(2)  # collapse Z
    proj_xz = _project(1)  # collapse Y
    proj_yz = _project(0)  # collapse X

    # Add 2-pixel white separator between views
    G = voxel_grid.shape[1]
    sep = np.full((G, 2, 3), 200, dtype=np.uint8)
    composite = np.concatenate([proj_xy, sep, proj_xz, sep, proj_yz], axis=1)

    # Upscale for visibility
    if scale > 1:
        composite = np.repeat(np.repeat(composite, scale, axis=0), scale, axis=1)

    return composite
