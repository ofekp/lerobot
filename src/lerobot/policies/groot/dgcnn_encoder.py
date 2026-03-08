"""DGCNN point cloud encoder for depth observations.

Provides:
  - ``backproject()``: back-projects depth images into world-frame 3D point
    clouds (pure tensor ops, no learnable parameters).
  - ``DGCNNEncoder``: processes point clouds with Dynamic Graph CNN to produce
    a flat sequence of tokens suitable for transformer consumption.

The DGCNN architecture follows Wang et al. "Dynamic Graph CNN for Learning on
Point Clouds" (2019), using iterative EdgeConv layers with dynamic k-NN graph
construction.
"""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Module-level cache for the most recent point cloud (used by eval rendering).
# Set by DGCNNEncoder.forward(); read by LiberoEnv.render().
_point_cloud_cache: torch.Tensor | None = None


# ------------------------------------------------------------------
# Back-projection (standalone function)
# ------------------------------------------------------------------


def backproject(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unproject depth pixels into world-frame 3D points.

    Applies ``K^{-1} * [u, v, 1]^T * d`` to obtain camera-frame points, then
    transforms them to the world frame via the inverse of the extrinsics
    (world-to-camera) matrix.

    Args:
        depth: ``(B, 1, H, W)`` depth map in metres.
        intrinsics: ``(B, 3, 3)`` camera intrinsic matrices.
        extrinsics: ``(B, 4, 4)`` world-to-camera rigid transforms.

    Returns:
        points_world: ``(B, N, 3)`` 3D points in world frame where ``N = H * W``.
        valid_mask: ``(B, N)`` boolean mask (``True`` where depth is in
            ``[0.01, 10.0]`` metres).
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
    # Cast to float32 for inverse (linalg.inv doesn't support bf16/fp16)
    K_inv = torch.inverse(intrinsics.float())  # (B, 3, 3)
    # (B, 3, 3) @ (3, N) -> (B, 3, N)
    rays = K_inv @ pixel_coords.unsqueeze(0).expand(B, -1, -1)
    points_cam = rays * depth_flat.float().unsqueeze(1)  # (B, 3, N)

    # Transform to world frame.  extrinsics is world-to-camera, so we
    # need its inverse (camera-to-world).
    cam_to_world = torch.inverse(extrinsics.float())  # (B, 4, 4)
    R = cam_to_world[:, :3, :3]  # (B, 3, 3)
    t = cam_to_world[:, :3, 3:]  # (B, 3, 1)
    points_world = R @ points_cam + t  # (B, 3, N)
    points_world = points_world.permute(0, 2, 1)  # (B, N, 3)

    # Validity mask: reject depth values outside [0.01, 10.0] metres
    valid_mask = (depth_flat > 0.01) & (depth_flat < 10.0)  # (B, N)

    return points_world, valid_mask


# ------------------------------------------------------------------
# k-Nearest Neighbours
# ------------------------------------------------------------------


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-nearest neighbours in feature space.

    Uses pairwise Euclidean distances via :func:`torch.cdist` and selects the
    ``k`` closest neighbours for each point.

    Args:
        x: ``(B, N, C)`` point features.
        k: Number of neighbours to retrieve.

    Returns:
        idx: ``(B, N, k)`` indices of the k nearest neighbours.
    """
    # (B, N, N) pairwise distances
    dists = torch.cdist(x, x)  # (B, N, N)
    # topk with largest=False gives smallest distances; fetch k+1 and discard
    # index 0 (self) which always has distance 0.
    _, idx = dists.topk(k + 1, dim=-1, largest=False)  # (B, N, k+1)
    idx = idx[:, :, 1:]  # exclude self (always index 0 = distance 0)
    return idx


# ------------------------------------------------------------------
# Farthest Point Sampling
# ------------------------------------------------------------------


def farthest_point_sampling(xyz: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Iterative farthest point sampling.

    Greedily selects ``num_samples`` points from each batch element such that
    each new point is the one farthest from the already-selected set.

    Args:
        xyz: ``(B, N, 3)`` point coordinates.
        num_samples: Number of points to sample.

    Returns:
        idx: ``(B, num_samples)`` indices of selected points.
    """
    B, N, _ = xyz.shape
    device = xyz.device

    idx = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    # Distance from each point to the nearest selected point (initialised to inf)
    distances = torch.full((B, N), float("inf"), device=device)

    # Start from a random point per batch element
    farthest = torch.randint(0, N, (B,), device=device)

    for i in range(num_samples):
        idx[:, i] = farthest
        # (B, 1, 3) — the newly selected point
        centroid = xyz[torch.arange(B, device=device), farthest, :].unsqueeze(1)
        # (B, N) — distance from every point to the new centroid
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        # Update: keep the *minimum* distance to any selected point so far
        distances = torch.min(distances, dist)
        # Next point is the one with the largest minimum distance
        farthest = distances.argmax(dim=-1)

    return idx


# ------------------------------------------------------------------
# EdgeConv — Dynamic Graph Convolution Block
# ------------------------------------------------------------------


class EdgeConv(nn.Module):
    """Dynamic graph convolution block from DGCNN.

    Constructs a k-NN graph in feature space, computes edge features as
    ``[x_i, x_j - x_i]``, applies an MLP (Linear + BatchNorm + LeakyReLU),
    and max-pools over the neighbours.

    Args:
        in_channels: Dimensionality of input point features.
        out_channels: Dimensionality of output point features.
        k: Number of nearest neighbours for graph construction.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 20) -> None:
        super().__init__()
        self.k = k
        # Conv1d layout so BatchNorm1d computes stats over the channel dim
        # correctly (one stat per channel, aggregated over spatial positions).
        self.conv = nn.Conv1d(2 * in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamic graph convolution.

        Args:
            x: ``(B, N, C_in)`` input point features.

        Returns:
            ``(B, N, C_out)`` output point features.
        """
        B, N, C = x.shape
        k = min(self.k, N)  # guard against fewer points than k

        # 1. Build k-NN graph in feature space
        idx = knn(x, k)  # (B, N, k)

        # 2. Gather neighbours
        # Expand idx for gather: (B, N*k, C)
        idx_flat = idx.reshape(B, N * k)  # (B, N*k)
        idx_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, C)  # (B, N*k, C)
        neighbours = torch.gather(x, dim=1, index=idx_expanded)  # (B, N*k, C)
        neighbours = neighbours.reshape(B, N, k, C)  # (B, N, k, C)

        # 3. Edge features: [x_i (repeated), x_j - x_i]
        x_i = x.unsqueeze(2).expand(-1, -1, k, -1)  # (B, N, k, C)
        edge_feat = torch.cat([x_i, neighbours - x_i], dim=-1)  # (B, N, k, 2*C)

        # 4. Apply Conv1d + BN + act on (B, 2C, N*k) layout
        # Reshape: (B, N, k, 2C) -> (B, 2C, N*k)
        edge_feat = edge_feat.permute(0, 3, 1, 2).reshape(B, 2 * C, N * k)
        # Conv1d + BN + act: (B, C_out, N*k)
        edge_feat = self.act(self.bn(self.conv(edge_feat)))
        # Reshape back: (B, C_out, N, k) -> max over k -> (B, C_out, N)
        edge_feat = edge_feat.reshape(B, -1, N, k).max(dim=-1).values  # (B, C_out, N)
        out = edge_feat.permute(0, 2, 1)  # (B, N, C_out)

        return out


# ------------------------------------------------------------------
# DGCNN Encoder — Main Module
# ------------------------------------------------------------------


class DGCNNEncoder(nn.Module):
    """DGCNN-based encoder that converts depth observations to token sequences.

    Back-projects depth images into world-frame point clouds, processes them
    through four :class:`EdgeConv` layers with dynamic graph construction, and
    produces a fixed-length sequence of tokens via farthest point sampling and
    linear projection.

    Args:
        num_points: Number of points to sample from the point cloud.
        k: Number of nearest neighbours for EdgeConv graph construction.
        hidden_dim: Dimension of each output token.
        num_tokens: Number of output tokens (selected via FPS).
        workspace_bounds: Per-axis ``(min, max)`` bounds defining the volume of
            interest in world coordinates.
    """

    def __init__(
        self,
        num_points: int = 2048,
        k: int = 20,
        hidden_dim: int = 1536,
        num_tokens: int = 64,
        workspace_bounds: tuple[tuple[float, float], ...] = (
            (-0.5, 4.5),
            (-1.5, 1.5),
            (1.0, 3.5),
        ),
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.k = k
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.workspace_bounds = workspace_bounds

        # Register workspace bounds as a buffer so they travel with the model
        # (device / dtype transfers, state_dict, etc.)
        bounds_tensor = torch.tensor(workspace_bounds, dtype=torch.float32)  # (3, 2)
        self.register_buffer("_bounds", bounds_tensor)

        # Four EdgeConv layers with increasing feature dimensions
        self.conv1 = EdgeConv(3, 64, k)
        self.conv2 = EdgeConv(64, 128, k)
        self.conv3 = EdgeConv(128, 256, k)
        self.conv4 = EdgeConv(256, 512, k)

        # Concatenated features from all layers: 64 + 128 + 256 + 512 = 960
        self.projection = nn.Linear(960, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # Cache slot for the most recent point cloud (used by eval rendering)
        self._last_point_cloud = None

    def forward(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Encode depth observations into a token sequence.

        Supports both single-view and multi-view inputs.  For multi-view, point
        clouds from all views are concatenated before processing.

        Args:
            depth: ``(B, V, 1, H, W)`` multi-view or ``(B, 1, H, W)``
                single-view depth maps in metres.
            intrinsics: ``(B, V, 3, 3)`` or ``(B, 3, 3)`` camera intrinsics.
            extrinsics: ``(B, V, 4, 4)`` or ``(B, 4, 4)`` world-to-camera
                transforms.

        Returns:
            ``(B, num_tokens, hidden_dim)`` token tensor.
        """
        # ----------------------------------------------------------
        # 1. Back-project depth to world-frame points
        # ----------------------------------------------------------
        if depth.dim() == 5:
            # Multi-view: (B, V, 1, H, W)
            B, V = depth.shape[:2]
            all_points = []
            all_masks = []
            for v in range(V):
                pts, mask = backproject(
                    depth[:, v],          # (B, 1, H, W)
                    intrinsics[:, v],     # (B, 3, 3)
                    extrinsics[:, v],     # (B, 4, 4)
                )
                all_points.append(pts)   # (B, N_v, 3)
                all_masks.append(mask)   # (B, N_v)
            points = torch.cat(all_points, dim=1)  # (B, N_total, 3)
            valid = torch.cat(all_masks, dim=1)     # (B, N_total)
        else:
            # Single-view: (B, 1, H, W)
            B = depth.shape[0]
            points, valid = backproject(depth, intrinsics, extrinsics)

        # ----------------------------------------------------------
        # 2. Filter to workspace bounds
        # ----------------------------------------------------------
        bounds = self._bounds.to(dtype=points.dtype)  # (3, 2)
        lo = bounds[:, 0]  # (3,)
        hi = bounds[:, 1]  # (3,)

        in_bounds = (
            (points[..., 0] >= lo[0]) & (points[..., 0] <= hi[0])
            & (points[..., 1] >= lo[1]) & (points[..., 1] <= hi[1])
            & (points[..., 2] >= lo[2]) & (points[..., 2] <= hi[2])
        )  # (B, N)
        mask = valid & in_bounds  # (B, N)

        assert mask.any(), (
            f"No 3D points survived workspace filtering — check depth values and bounds {self._bounds}"
        )

        # ----------------------------------------------------------
        # 3. Sample to num_points per batch element
        # ----------------------------------------------------------
        sampled_points = torch.zeros(
            B, self.num_points, 3, device=points.device, dtype=points.dtype,
        )

        for b in range(B):
            valid_idx = mask[b].nonzero(as_tuple=False).squeeze(-1)  # (M,)
            M = valid_idx.shape[0]

            if M == 0:
                # No valid points — fill with zeros (degenerate case)
                logger.warning(
                    "DGCNNEncoder: no valid points in batch element %d "
                    "(all depth invalid or outside workspace)", b,
                )
                continue
            elif M >= self.num_points:
                # Enough points — use FPS for uniform coverage
                valid_pts = points[b, valid_idx]  # (M, 3)
                with torch.no_grad():
                    fps_idx = farthest_point_sampling(
                        valid_pts.unsqueeze(0), self.num_points,
                    )  # (1, num_points)
                sampled_points[b] = valid_pts[fps_idx[0]]
            else:
                # Too few points — take all, then oversample with replacement
                valid_pts = points[b, valid_idx]  # (M, 3)
                pad_idx = torch.randint(0, M, (self.num_points - M,), device=points.device)
                sampled_points[b] = torch.cat([valid_pts, valid_pts[pad_idx]], dim=0)

        # ----------------------------------------------------------
        # 4. DGCNN layers — four EdgeConv blocks
        # ----------------------------------------------------------
        f1 = self.conv1(sampled_points)  # (B, num_points, 64)
        f2 = self.conv2(f1)              # (B, num_points, 128)
        f3 = self.conv3(f2)              # (B, num_points, 256)
        f4 = self.conv4(f3)              # (B, num_points, 512)

        # Concatenate features from all layers
        features = torch.cat([f1, f2, f3, f4], dim=-1)  # (B, num_points, 960)

        assert features.abs().sum() > 0, "EdgeConv features are all zeros"

        # ----------------------------------------------------------
        # 5. Pool to num_tokens via FPS on spatial coordinates
        # ----------------------------------------------------------
        with torch.no_grad():
            seed_idx = farthest_point_sampling(
                sampled_points, self.num_tokens,
            )  # (B, num_tokens)

        # Gather features at seed indices
        seed_idx_expanded = seed_idx.unsqueeze(-1).expand(
            -1, -1, features.shape[-1],
        )  # (B, num_tokens, 960)
        tokens = torch.gather(features, dim=1, index=seed_idx_expanded)  # (B, num_tokens, 960)

        # ----------------------------------------------------------
        # 6. Project to hidden_dim
        # ----------------------------------------------------------
        tokens = self.projection(tokens)  # (B, num_tokens, hidden_dim)
        tokens = self.norm(tokens)        # (B, num_tokens, hidden_dim)

        assert tokens.shape == (B, self.num_tokens, self.hidden_dim), (
            f"Output shape mismatch: {tokens.shape} vs expected ({B}, {self.num_tokens}, {self.hidden_dim})"
        )

        # ----------------------------------------------------------
        # 7. Cache point cloud for visualization
        # ----------------------------------------------------------
        self._last_point_cloud = sampled_points[0].detach().cpu()
        global _point_cloud_cache
        _point_cloud_cache = self._last_point_cloud

        return tokens


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------


def render_point_cloud_projections(
    points: torch.Tensor,
    workspace_bounds: tuple[tuple[float, float], ...] = (
        (-0.5, 4.5),
        (-1.5, 1.5),
        (1.0, 3.5),
    ),
    scale: int = 4,
) -> np.ndarray:
    """Render three orthographic projections of a point cloud.

    Produces a composite image with three views side-by-side:
      - **XY** (top-down)
      - **XZ** (front)
      - **YZ** (side)

    Points are coloured by Z-depth (near = red, far = blue).

    Args:
        points: ``(N, 3)`` point cloud on CPU.
        workspace_bounds: Per-axis ``(min, max)`` bounds for normalisation.
        scale: Integer upscale factor for visibility (default 4).

    Returns:
        ``(H, W, 3)`` uint8 numpy array suitable for saving as PNG.
    """
    pts = points.numpy() if isinstance(points, torch.Tensor) else np.asarray(points)

    if pts.shape[0] == 0:
        # Degenerate: return a small grey image
        res = 64
        return np.full((res, res * 3 + 4, 3), 40, dtype=np.uint8)

    bounds = np.array(workspace_bounds)  # (3, 2)
    lo = bounds[:, 0]
    hi = bounds[:, 1]

    # Normalise to [0, 1] within workspace
    span = hi - lo
    span[span == 0] = 1.0  # avoid division by zero
    normed = (pts - lo) / span  # (N, 3)
    normed = np.clip(normed, 0, 1)

    # Colour by Z-depth: near (low Z) = red, far (high Z) = blue
    z_norm = normed[:, 2]  # (N,)
    # Simple red-to-blue colormap: R = (1-z), G = 0, B = z
    colors = np.stack([
        (1.0 - z_norm) * 255,
        np.zeros_like(z_norm),
        z_norm * 255,
    ], axis=-1).astype(np.uint8)  # (N, 3)

    res = 64  # resolution of each projection view

    def _project(ax0: int, ax1: int) -> np.ndarray:
        """Scatter points onto a 2D grid along two axes."""
        img = np.full((res, res, 3), 40, dtype=np.uint8)  # dark grey background
        # Map normalised coordinates to pixel indices
        u = (normed[:, ax0] * (res - 1)).astype(int)
        v = ((1.0 - normed[:, ax1]) * (res - 1)).astype(int)  # flip Y for image coords
        u = np.clip(u, 0, res - 1)
        v = np.clip(v, 0, res - 1)
        # Paint (later points overwrite earlier ones — good enough for viz)
        img[v, u] = colors
        return img

    proj_xy = _project(0, 1)  # top-down: X horizontal, Y vertical
    proj_xz = _project(0, 2)  # front:    X horizontal, Z vertical
    proj_yz = _project(1, 2)  # side:     Y horizontal, Z vertical

    # 2-pixel white separator between views
    sep = np.full((res, 2, 3), 200, dtype=np.uint8)
    composite = np.concatenate([proj_xy, sep, proj_xz, sep, proj_yz], axis=1)

    # Upscale for visibility
    if scale > 1:
        composite = np.repeat(np.repeat(composite, scale, axis=0), scale, axis=1)

    return composite
