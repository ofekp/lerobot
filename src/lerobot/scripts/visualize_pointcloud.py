#!/usr/bin/env python
"""Plotly-based point cloud visualizer for depth datasets.

Loads one frame from a LeRobot dataset, back-projects depth images per camera
into world-frame 3D point clouds, applies workspace bounds filtering, and
renders an interactive plotly Scatter3d HTML with one trace per camera
(toggleable via legend).

Usage:
    python -m lerobot.scripts.visualize_pointcloud \
        --dataset_path /data/libero_spatial_replay \
        --episode 0 --frame 0 \
        --output output/pointcloud_viz.html
"""

import argparse
import logging
from pathlib import Path

import torch

from lerobot.policies.groot.dgcnn_encoder import backproject

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE_BOUNDS = (
    (-0.5, 4.5),
    (-1.5, 1.5),
    (1.0, 3.5),
)


def load_frame(dataset_path: str, episode: int, frame: int) -> dict:
    """Load a single frame from a LeRobot dataset on disk."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(dataset_path)

    # Find the absolute frame index for the requested episode/frame
    ep_starts = ds.meta.episodes["dataset_from_index"]
    ep_ends = ds.meta.episodes["dataset_to_index"]

    if episode >= len(ep_starts):
        raise ValueError(
            f"Episode {episode} out of range (dataset has {len(ep_starts)} episodes)"
        )

    start = ep_starts[episode]
    end = ep_ends[episode]
    start = start.item() if hasattr(start, "item") else int(start)
    end = end.item() if hasattr(end, "item") else int(end)

    abs_idx = start + frame
    if abs_idx >= end:
        raise ValueError(
            f"Frame {frame} out of range for episode {episode} "
            f"(episode has {end - start} frames)"
        )

    return ds[abs_idx]


def extract_cameras(sample: dict) -> list[dict]:
    """Extract per-camera depth, intrinsics, and extrinsics from a sample.

    Returns a list of dicts with keys: name, depth, intrinsics, extrinsics.
    """
    cameras = []

    # Find all depth keys
    depth_keys = sorted(
        k for k in sample if k.startswith("observation.images.") and "depth" in k.lower()
    )

    for dk in depth_keys:
        # e.g. "observation.images.frontview.depth" -> "frontview"
        parts = dk.split(".")
        # Find camera name between "images." and ".depth"
        img_idx = parts.index("images")
        cam_name = ".".join(parts[img_idx + 1 : -1])  # handle nested names

        # Look for corresponding intrinsics and extrinsics
        # They may be under observation.images.{name}.* or observation.camera.{name}.*
        intrinsics_key = None
        extrinsics_key = None
        for prefix in [f"observation.images.{cam_name}", f"observation.camera.{cam_name}"]:
            ik = f"{prefix}.intrinsics"
            ek = f"{prefix}.extrinsics"
            if ik in sample and ek in sample:
                intrinsics_key = ik
                extrinsics_key = ek
                break

        if intrinsics_key is None or extrinsics_key is None:
            logger.warning(
                "Skipping camera %s: missing intrinsics or extrinsics", cam_name
            )
            continue

        depth = sample[dk]
        intrinsics = sample[intrinsics_key]
        extrinsics = sample[extrinsics_key]

        # Ensure correct shapes for backproject: depth (1, 1, H, W), intrinsics (1, 3, 3), extrinsics (1, 4, 4)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(0)  # (1, H, W) -> (1, 1, H, W)

        if intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0)  # (3, 3) -> (1, 3, 3)
        if extrinsics.dim() == 2:
            extrinsics = extrinsics.unsqueeze(0)  # (4, 4) -> (1, 4, 4)

        cameras.append({
            "name": cam_name,
            "depth": depth.float(),
            "intrinsics": intrinsics.float(),
            "extrinsics": extrinsics.float(),
        })

    return cameras


def backproject_cameras(
    cameras: list[dict],
    workspace_bounds: tuple[tuple[float, float], ...] = DEFAULT_WORKSPACE_BOUNDS,
    max_points_per_camera: int = 50000,
) -> list[dict]:
    """Back-project each camera's depth and filter to workspace bounds.

    Returns list of dicts with keys: name, points (N, 3) numpy array.
    """
    bounds = torch.tensor(workspace_bounds, dtype=torch.float32)
    lo = bounds[:, 0]
    hi = bounds[:, 1]

    results = []
    for cam in cameras:
        points, valid = backproject(cam["depth"], cam["intrinsics"], cam["extrinsics"])
        # points: (1, N, 3), valid: (1, N)
        points = points[0]  # (N, 3)
        valid = valid[0]  # (N,)

        # Apply workspace bounds
        in_bounds = (
            (points[:, 0] >= lo[0]) & (points[:, 0] <= hi[0])
            & (points[:, 1] >= lo[1]) & (points[:, 1] <= hi[1])
            & (points[:, 2] >= lo[2]) & (points[:, 2] <= hi[2])
        )
        mask = valid & in_bounds
        filtered = points[mask]

        # Subsample if too many points for plotly performance
        if filtered.shape[0] > max_points_per_camera:
            idx = torch.randperm(filtered.shape[0])[:max_points_per_camera]
            filtered = filtered[idx]

        logger.info(
            "Camera %s: %d valid points (%d after bounds filter, %d displayed)",
            cam["name"],
            valid.sum().item(),
            mask.sum().item(),
            filtered.shape[0],
        )

        results.append({
            "name": cam["name"],
            "points": filtered.numpy(),
        })

    return results


def create_plotly_figure(
    camera_points: list[dict],
    workspace_bounds: tuple[tuple[float, float], ...] = DEFAULT_WORKSPACE_BOUNDS,
):
    """Create an interactive plotly 3D scatter figure with per-camera legend."""
    import plotly.graph_objects as go

    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
    ]

    fig = go.Figure()

    for i, cam in enumerate(camera_points):
        pts = cam["points"]
        if pts.shape[0] == 0:
            continue

        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=color, opacity=0.7),
            name=cam["name"],
            legendgroup=cam["name"],
        ))

    # Add workspace bounds wireframe
    bounds = workspace_bounds
    corners = [
        (bounds[0][i], bounds[1][j], bounds[2][k])
        for i in range(2) for j in range(2) for k in range(2)
    ]
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),  # z-edges
        (0, 2), (1, 3), (4, 6), (5, 7),  # y-edges
        (0, 4), (1, 5), (2, 6), (3, 7),  # x-edges
    ]
    for e0, e1 in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[e0][0], corners[e1][0]],
            y=[corners[e0][1], corners[e1][1]],
            z=[corners[e0][2], corners[e1][2]],
            mode="lines",
            line=dict(color="gray", width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title="Point Cloud Visualization (per camera)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(title="Cameras"),
        width=1200,
        height=800,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize point clouds from a LeRobot depth dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to LeRobot dataset")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--frame", type=int, default=0, help="Frame index within episode")
    parser.add_argument("--output", type=str, default="output/pointcloud_viz.html", help="Output HTML path")
    parser.add_argument(
        "--workspace_bounds",
        type=float,
        nargs=6,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help="Workspace bounds as 6 floats: x_min x_max y_min y_max z_min z_max",
    )
    parser.add_argument("--max_points", type=int, default=50000, help="Max points per camera for display")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.workspace_bounds is not None:
        wb = args.workspace_bounds
        workspace_bounds = ((wb[0], wb[1]), (wb[2], wb[3]), (wb[4], wb[5]))
    else:
        workspace_bounds = DEFAULT_WORKSPACE_BOUNDS

    logger.info("Loading frame: episode=%d, frame=%d from %s", args.episode, args.frame, args.dataset_path)
    sample = load_frame(args.dataset_path, args.episode, args.frame)

    cameras = extract_cameras(sample)
    if not cameras:
        logger.error("No depth cameras found in dataset. Available keys: %s", sorted(sample.keys()))
        return

    logger.info("Found %d depth camera(s): %s", len(cameras), [c["name"] for c in cameras])

    camera_points = backproject_cameras(cameras, workspace_bounds, args.max_points)

    total_points = sum(c["points"].shape[0] for c in camera_points)
    logger.info("Total points to render: %d", total_points)

    fig = create_plotly_figure(camera_points, workspace_bounds)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info("Saved interactive visualization to %s", output_path)


if __name__ == "__main__":
    main()
