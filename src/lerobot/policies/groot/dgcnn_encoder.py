"""DGCNN point cloud encoder for depth observations.

Provides:
  - ``backproject()``: back-projects depth images into world-frame 3D point
    clouds (pure tensor ops, no learnable parameters).  Handles the vertical
    flip (OpenGL framebuffer origin) and OpenCV→MuJoCo axis conversion.
  - ``DGCNNEncoder``: processes point clouds with Dynamic Graph CNN to produce
    a flat sequence of tokens suitable for transformer consumption.

Extrinsics convention: camera-to-world ``[R_cam_to_world | cam_pos]`` where
``R_cam_to_world`` is MuJoCo ``cam_xmat`` and ``cam_pos`` is ``cam_xpos``.

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

    The depth images stored in the dataset (and rendered at eval time) are in
    OpenGL framebuffer order (origin at bottom-left), which is vertically
    flipped relative to the top-left convention assumed by the intrinsics K.
    This function:

    1. Flips the depth vertically so pixel coords match K.
    2. Back-projects via ``K^{-1} * [u, v, 1]^T * d`` → OpenCV camera frame.
    3. Converts OpenCV axes (X right, Y down, Z forward) to MuJoCo/OpenGL
       axes (X right, Y up, Z backward): ``[x, -y, -z]``.
    4. Transforms to world frame using the extrinsics, which are stored as
       ``[R_cam_to_world | cam_pos]`` (camera-to-world, MuJoCo convention).

    Args:
        depth: ``(B, 1, H, W)`` depth map in metres.
        intrinsics: ``(B, 3, 3)`` camera intrinsic matrices.
        extrinsics: ``(B, 4, 4)`` camera-to-world transforms where
            ``E[:3, :3]`` is ``R_cam_to_world`` (MuJoCo ``cam_xmat``) and
            ``E[:3, 3]`` is ``cam_pos`` (MuJoCo ``cam_xpos``).

    Returns:
        points_world: ``(B, N, 3)`` 3D points in world frame where ``N = H * W``.
        valid_mask: ``(B, N)`` boolean mask (``True`` where depth is in
            ``[0.01, 10.0]`` metres).
    """
    B, _, H, W = depth.shape
    device = depth.device

    # Step 1: Flip depth vertically so pixel (0,0) = top-left, matching K.
    # The stored depth is in OpenGL framebuffer order (origin bottom-left).
    depth = torch.flip(depth, dims=[-2])

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

    # Step 2: Back-project → OpenCV camera frame
    # p_cam_opencv = K^{-1} * [u, v, 1]^T * d
    # Cast to float32 for inverse (linalg.inv doesn't support bf16/fp16)
    K_inv = torch.inverse(intrinsics.float())  # (B, 3, 3)
    # (B, 3, 3) @ (3, N) -> (B, 3, N)
    rays = K_inv @ pixel_coords.unsqueeze(0).expand(B, -1, -1)
    points_cam = rays * depth_flat.float().unsqueeze(1)  # (B, 3, N)  — OpenCV axes

    # Step 3-4: Convert OpenCV axes → MuJoCo axes: [x, -y, -z]
    # OpenCV: X right, Y down, Z forward
    # MuJoCo/OpenGL: X right, Y up, Z backward
    points_cam[:, 1, :] = -points_cam[:, 1, :]
    points_cam[:, 2, :] = -points_cam[:, 2, :]

    # Step 5: Camera-to-world using extrinsics directly.
    # E = [R_cam_to_world | cam_pos], so:
    #   points_world = R_cam_to_world @ points_cam_mujoco + cam_pos
    R = extrinsics[:, :3, :3].float()   # R_cam_to_world  (B, 3, 3)
    t = extrinsics[:, :3, 3:].float()   # cam_pos          (B, 3, 1)
    points_world = R @ points_cam + t   # (B, 3, N)
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
        camera_weights: str | None = None,
        debug_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.k = k
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.workspace_bounds = workspace_bounds
        self.debug_dir = debug_dir

        # Parse camera weights: "front:30,wrist:70" -> {"front": 0.3, "wrist": 0.7}
        self.camera_weight_map: dict[str, float] | None = None
        if camera_weights is not None:
            parsed = {}
            for entry in camera_weights.split(","):
                name, pct = entry.strip().split(":")
                parsed[name.strip()] = float(pct.strip())
            total = sum(parsed.values())
            if abs(total - 100.0) > 0.01:
                raise ValueError(
                    f"dgcnn_camera_weights must sum to 100, got {total}: {camera_weights}"
                )
            self.camera_weight_map = {k: v / 100.0 for k, v in parsed.items()}

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

        # Debug visualization state (set externally after construction)
        self._saved_first_batch_train: bool = False
        self._saved_first_batch_eval: bool = False

        # Cache slot for the most recent point cloud (used by eval rendering)
        self._last_point_cloud = None

    def forward(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        camera_names: list[str] | None = None,
    ) -> torch.Tensor:
        """Encode depth observations into a token sequence.

        Supports both single-view and multi-view inputs.  For multi-view, point
        clouds from all views are concatenated before processing.

        Args:
            depth: ``(B, V, 1, H, W)`` multi-view or ``(B, 1, H, W)``
                single-view depth maps in metres.
            intrinsics: ``(B, V, 3, 3)`` or ``(B, 3, 3)`` camera intrinsics.
            extrinsics: ``(B, V, 4, 4)`` or ``(B, 4, 4)`` camera-to-world
                transforms (``[R_cam_to_world | cam_pos]``).
            camera_names: list of camera names matching the view dimension,
                used for per-camera weighted sampling when ``camera_weights``
                was specified at init.

        Returns:
            ``(B, num_tokens, hidden_dim)`` token tensor.
        """
        # ----------------------------------------------------------
        # 1. Back-project depth to world-frame points, track view origin
        # ----------------------------------------------------------
        if depth.dim() == 5:
            # Multi-view: (B, V, 1, H, W)
            B, V = depth.shape[:2]
            all_points = []
            all_masks = []
            all_view_ids = []
            for v in range(V):
                pts, msk = backproject(
                    depth[:, v],          # (B, 1, H, W)
                    intrinsics[:, v],     # (B, 3, 3)
                    extrinsics[:, v],     # (B, 4, 4)
                )
                all_points.append(pts)   # (B, N_v, 3)
                all_masks.append(msk)    # (B, N_v)
                # Tag each point with its view index
                all_view_ids.append(torch.full(
                    (B, pts.shape[1]), v, device=pts.device, dtype=torch.long,
                ))
            points = torch.cat(all_points, dim=1)      # (B, N_total, 3)
            valid = torch.cat(all_masks, dim=1)          # (B, N_total)
            view_ids = torch.cat(all_view_ids, dim=1)    # (B, N_total)
        else:
            # Single-view: (B, 1, H, W)
            B = depth.shape[0]
            V = 1
            points, valid = backproject(depth, intrinsics, extrinsics)
            view_ids = torch.zeros(B, points.shape[1], device=points.device, dtype=torch.long)

        # Validate camera_weights covers all depth cameras
        if self.camera_weight_map is not None:
            if camera_names is None:
                raise ValueError(
                    "dgcnn_camera_weights is set but camera_names was not passed to forward()"
                )
            if len(camera_names) != V:
                raise ValueError(
                    f"camera_names length ({len(camera_names)}) != number of views ({V})"
                )
            missing = set(camera_names) - set(self.camera_weight_map)
            if missing:
                raise ValueError(
                    f"Depth cameras {missing} not listed in dgcnn_camera_weights. "
                    f"Every depth camera must be specified. Got: {self.camera_weight_map}"
                )
            extra = set(self.camera_weight_map) - set(camera_names)
            if extra:
                raise ValueError(
                    f"dgcnn_camera_weights lists cameras {extra} not found in depth data. "
                    f"Depth cameras in dataset: {camera_names}"
                )

        # ----------------------------------------------------------
        # 2. Filter to workspace bounds (DISABLED — using depth validity only)
        # ----------------------------------------------------------
        # TODO: Re-enable workspace bounds filtering once bounds are properly
        # calibrated for the target environment.  For now, relying solely on
        # the depth validity mask (0.01–10.0 m) avoids accidentally discarding
        # all points due to incorrect bounds.
        mask = valid  # (B, N)

        assert mask.any(), (
            "No valid 3D points — all depth values outside [0.01, 10.0] metres"
        )

        # ----------------------------------------------------------
        # 3. Sample to num_points per batch element
        # ----------------------------------------------------------
        sampled_points = torch.zeros(
            B, self.num_points, 3, device=points.device, dtype=points.dtype,
        )

        # Build per-point sampling weights (uniform or camera-weighted)
        if self.camera_weight_map is not None and camera_names is not None:
            weight_per_view = torch.tensor(
                [self.camera_weight_map[c] for c in camera_names],
                device=points.device, dtype=points.dtype,
            )  # (V,)
            point_weights = weight_per_view[view_ids]  # (B, N)
            point_weights = point_weights * mask.float()  # zero out invalid
        else:
            point_weights = mask.float()  # (B, N) — uniform over valid points

        for b in range(B):
            w = point_weights[b]
            if w.sum() == 0:
                logger.warning(
                    "DGCNNEncoder: no valid points in batch element %d "
                    "(all depth invalid or outside workspace)", b,
                )
                continue
            idx = torch.multinomial(w, self.num_points, replacement=True)
            sampled_points[b] = points[b, idx]

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
        # 5. Pool to num_tokens via random sampling on spatial coordinates
        # ----------------------------------------------------------
        seed_idx = torch.stack([
            torch.randperm(self.num_points, device=features.device)[:self.num_tokens]
            for _ in range(B)
        ])  # (B, num_tokens)

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
        # 7. Cache point cloud & save diagnostic HTML (first batch only)
        # ----------------------------------------------------------
        self._last_point_cloud = sampled_points[0].detach().cpu()
        global _point_cloud_cache
        _point_cloud_cache = self._last_point_cloud

        if self.debug_dir is not None:
            # Detect mode from nn.Module.training flag; separate flags for
            # train/eval so each gets its own pair of HTMLs.
            mode = "train" if self.training else "eval"
            already_saved = (
                self._saved_first_batch_train if self.training
                else self._saved_first_batch_eval
            )
            if not already_saved:
                if self.training:
                    self._saved_first_batch_train = True
                else:
                    self._saved_first_batch_eval = True
                try:
                    # Detach everything to CPU for visualization
                    all_points_cpu = points.detach().cpu()      # (B, N_total, 3)
                    view_ids_cpu = view_ids.detach().cpu()       # (B, N_total)
                    mask_cpu = mask.detach().cpu()                # (B, N_total)
                    sampled_cpu = sampled_points.detach().cpu()   # (B, num_points, 3)
                    cam_names = camera_names or [f"cam_{v}" for v in range(V)]
                    bounds_cpu = self._bounds.detach().cpu()

                    # Save for first (idx=0) and last (idx=B-1) batch elements
                    for label, b_idx in [("first", 0), ("last", B - 1)]:
                        save_point_cloud_html(
                            all_points=all_points_cpu[b_idx],
                            view_ids=view_ids_cpu[b_idx],
                            valid_mask=mask_cpu[b_idx],
                            sampled_points=sampled_cpu[b_idx],
                            camera_names=cam_names,
                            workspace_bounds=bounds_cpu,
                            debug_dir=self.debug_dir,
                            mode=mode,
                            tag=label,
                        )
                except Exception as exc:
                    logger.warning("DGCNN point cloud HTML save failed: %s", exc)

        return tokens


# ------------------------------------------------------------------
# Visualization — interactive 3D HTML via Plotly
# ------------------------------------------------------------------


_PLOTLY_COLORS = [
    "rgb(214,39,40)", "rgb(31,119,180)", "rgb(44,160,44)",
    "rgb(255,127,14)", "rgb(148,103,189)", "rgb(23,190,207)",
]


def save_point_cloud_html(
    all_points: torch.Tensor,
    view_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    sampled_points: torch.Tensor,
    camera_names: list[str],
    workspace_bounds: torch.Tensor,
    debug_dir: str,
    mode: str = "train",
    tag: str = "first",
    max_display_points: int = 8000,
) -> None:
    """Save an interactive 3D point cloud visualization as an HTML file.

    Produces a Plotly figure with:
      - One trace per camera showing the full back-projected point cloud
        (filtered to workspace bounds + valid depth).
      - One trace for the final randomly sampled points actually consumed
        by the DGCNN encoder.
      - Workspace bounding box wireframe.
      - World-frame axis indicators.

    Args:
        all_points: ``(N_total, 3)`` all back-projected points (single batch element).
        view_ids: ``(N_total,)`` camera index per point (0, 1, …, V-1).
        valid_mask: ``(N_total,)`` bool mask after workspace + depth filtering.
        sampled_points: ``(num_points, 3)`` the subset actually fed to DGCNN.
        camera_names: list of camera names matching view indices.
        workspace_bounds: ``(3, 2)`` tensor with ``[lo, hi]`` per axis.
        debug_dir: directory to save the HTML file.
        mode: ``"train"`` or ``"eval"``.
        tag: additional label, e.g. ``"first"`` or ``"last"``.
        max_display_points: subsample each trace to at most this many points.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed — skipping point cloud HTML save")
        return

    pts_np = all_points.numpy() if isinstance(all_points, torch.Tensor) else np.asarray(all_points)
    vids_np = view_ids.numpy() if isinstance(view_ids, torch.Tensor) else np.asarray(view_ids)
    mask_np = valid_mask.numpy() if isinstance(valid_mask, torch.Tensor) else np.asarray(valid_mask)
    sampled_np = sampled_points.numpy() if isinstance(sampled_points, torch.Tensor) else np.asarray(sampled_points)
    bounds_np = workspace_bounds.numpy() if isinstance(workspace_bounds, torch.Tensor) else np.asarray(workspace_bounds)

    fig = go.Figure()

    # --- Per-camera point clouds (valid only, ALL points) ---
    V = len(camera_names)
    for v_idx in range(V):
        cam_mask = (vids_np == v_idx) & mask_np
        cam_pts = pts_np[cam_mask]
        if len(cam_pts) == 0:
            continue
        color = _PLOTLY_COLORS[v_idx % len(_PLOTLY_COLORS)]
        fig.add_trace(go.Scatter3d(
            x=cam_pts[:, 0], y=cam_pts[:, 1], z=cam_pts[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=color, opacity=0.5),
            name=f"{camera_names[v_idx]} ({len(cam_pts)} pts)",
        ))

    # --- Sampled points (what DGCNN actually processes, ALL points) ---
    fig.add_trace(go.Scatter3d(
        x=sampled_np[:, 0], y=sampled_np[:, 1], z=sampled_np[:, 2],
        mode="markers",
        marker=dict(size=2.5, color="rgb(0,0,0)", opacity=0.9),
        name=f"sampled ({len(sampled_np)} pts)",
    ))

    # --- Workspace bounding box (transparent solid + wireframe) ---
    lo = bounds_np[:, 0]  # (3,)
    hi = bounds_np[:, 1]  # (3,)
    # 8 corners of the box
    corners = np.array([
        [lo[0], lo[1], lo[2]],  # 0
        [hi[0], lo[1], lo[2]],  # 1
        [hi[0], hi[1], lo[2]],  # 2
        [lo[0], hi[1], lo[2]],  # 3
        [lo[0], lo[1], hi[2]],  # 4
        [hi[0], lo[1], hi[2]],  # 5
        [hi[0], hi[1], hi[2]],  # 6
        [lo[0], hi[1], hi[2]],  # 7
    ])
    # 12 triangles (2 per face) defining the 6 faces of the box
    tri_i = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 3, 3]
    tri_j = [1, 2, 5, 6, 1, 4, 3, 7, 3, 4, 2, 6]
    tri_k = [2, 3, 6, 7, 5, 7, 7, 6, 4, 7, 6, 7]
    fig.add_trace(go.Mesh3d(
        x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
        i=tri_i, j=tri_j, k=tri_k,
        color="lightblue",
        opacity=0.08,
        name="workspace bounds",
        showlegend=True,
    ))
    # Wireframe edges on top for clarity
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[i, 0], corners[j, 0]],
            y=[corners[i, 1], corners[j, 1]],
            z=[corners[i, 2], corners[j, 2]],
            mode="lines",
            line=dict(color="rgba(70,130,180,0.6)", width=3),
            showlegend=False,
        ))

    # --- World origin axes ---
    for ax_i, (color, name) in enumerate(zip(
        ["red", "green", "blue"], ["Xw", "Yw", "Zw"],
    )):
        end = np.zeros(3)
        end[ax_i] = 0.15
        fig.add_trace(go.Scatter3d(
            x=[0, end[0]], y=[0, end[1]], z=[0, end[2]],
            mode="lines+text",
            line=dict(color=color, width=5),
            text=["", name],
            textposition="top center",
            showlegend=False,
        ))

    fig.update_layout(
        title=f"DGCNN Point Cloud — {mode} / {tag}",
        scene=dict(
            xaxis_title="X (world)",
            yaxis_title="Y (world)",
            zaxis_title="Z (world)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    import os
    os.makedirs(debug_dir, exist_ok=True)
    save_path = os.path.join(debug_dir, f"dgcnn_point_cloud_{mode}_{tag}.html")
    fig.write_html(save_path)
    logger.info("Saved DGCNN point cloud HTML (%s/%s) to: %s", mode, tag, save_path)
