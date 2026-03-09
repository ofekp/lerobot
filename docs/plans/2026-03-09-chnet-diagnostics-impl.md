# CHNET Depth Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a diagnostics and interpretability layer for the CHNET depth pipeline that validates correctness, visualizes attention/embeddings, and documents expected values — without clogging logs.

**Architecture:** A single `DepthDiagnostics` class in `chnet_diagnostics.py` that hooks into the existing forward pass. Minimal changes to `chnet_modules.py` (return intermediate tensors when asked), `groot_n1.py` (delegate logging to diagnostics), and training/eval loops (call `report()` at intervals). Visualizations include cross-attention maps, FastGuide attention, embedding arithmetic (depth contribution, proximity direction, PCA, nearest-neighbor pairing).

**Tech Stack:** PyTorch, matplotlib (for visualizations), sklearn.decomposition.PCA (for token space analysis), numpy

**Design doc:** `docs/plans/2026-03-09-chnet-diagnostics-design.md`

---

### Task 1: Create `chnet_diagnostics.py` — Startup Banner

**Files:**
- Create: `src/lerobot/policies/groot/chnet_diagnostics.py`
- Test: Manual (print output verification)

**Step 1: Create the diagnostics module with startup banner**

Create `src/lerobot/policies/groot/chnet_diagnostics.py`:

```python
"""CHNET depth pipeline diagnostics and interpretability.

Provides:
- Startup banner: config validation, param counts, git hash
- Periodic health checks: depth stats, gradient norms, weight norms
- Interpretability visualizations: attention maps, embedding arithmetic, token PCA

Usage:
    diagnostics = DepthDiagnostics(config, model, output_dir)
    diagnostics.print_startup_banner()
    # In forward pass:
    if diagnostics.should_report(step):
        diagnostics.collect(step, ...)
        diagnostics.report(step)
    # After backward:
    diagnostics.log_gradients(step, model)
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _get_git_info():
    """Get current git hash and branch name."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return git_hash, branch
    except Exception:
        return "unknown", "unknown"


def _count_params(model, prefix=""):
    """Count trainable and total parameters, optionally filtered by prefix."""
    trainable = 0
    total = 0
    for name, p in model.named_parameters():
        if prefix and not name.startswith(prefix):
            continue
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


def _format_num(n):
    """Format number with K/M/B suffix."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


class DepthDiagnostics:
    """Diagnostics and interpretability for CHNET depth pipeline.

    Hooks into the CHNET forward pass to collect intermediate tensors
    and produce periodic health checks and visualizations.
    """

    def __init__(self, config, model, output_dir, train_interval=500, eval_interval=1):
        """
        Args:
            config: GrootConfig with depth settings.
            model: EagleBackbone or full model for param counting.
            output_dir: Base directory. Diagnostics saved to {output_dir}/diagnostics/.
            train_interval: Steps between diagnostics during training.
            eval_interval: Episodes between diagnostics during eval.
        """
        self.config = config
        self.model = model
        self.output_dir = Path(output_dir) / "diagnostics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_interval = train_interval
        self.eval_interval = eval_interval

        # Collected data for current step (cleared after report)
        self._collected = {}
        # Track initial weight norms for drift detection
        self._initial_weight_norms = None
        # Eval episode counter
        self._eval_episode = 0

    def print_startup_banner(self, dataset_info=None):
        """Print startup banner with config, git info, and expected values.

        Args:
            dataset_info: Optional dict with 'camera_names' key for camera validation.
        """
        git_hash, branch = _get_git_info()

        # Count parameters
        chnet_train, chnet_total = _count_params(self.model, prefix="chnet")
        total_train, total_total = _count_params(self.model)

        # Get camera names if available
        camera_info = "N/A"
        if dataset_info and "camera_names" in dataset_info:
            names = dataset_info["camera_names"]
            camera_info = f"{names}  ({len(names)} cameras)"

        banner = f"""
{'=' * 60}
 GROOT DEPTH DIAGNOSTICS - Run Summary
{'=' * 60}
 Git hash:             {git_hash:<20s} (verify: matches your intended commit)
 Branch:               {branch}
 use_depth:            {self.config.use_depth:<20} (expect: True for depth experiments)
 depth_scale/mean/std: {self.config.depth_scale}/{self.config.depth_mean}/{self.config.depth_std}  (expect: 1.0/0.0/1.0 for pass-through metric depth)
 chnet_tap_layers:     {self.config.chnet_tap_layers}  (expect: 4 evenly-spaced layers in 24-layer ViT)
 chnet_channels:       {self.config.chnet_channels}
 Cameras:              {camera_info}  (verify: matches your intent)
 CHNet params:         {_format_num(chnet_train)} trainable / {_format_num(chnet_total)} total  (expect: ~1-2M for default channels)
 Total params:         {_format_num(total_total)} total / {_format_num(total_train)} trainable
 Diagnostics interval: every {self.train_interval} train steps, every {self.eval_interval} eval episodes
 Output dir:           {self.output_dir}
{'=' * 60}"""
        print(banner, flush=True)

        # Save to file
        with open(self.output_dir / "startup_banner.txt", "w") as f:
            f.write(banner)

        # Record initial weight norms for drift tracking
        self._initial_weight_norms = self._compute_weight_norms()

    def should_report(self, step, is_training=True):
        """Check if diagnostics should fire at this step."""
        if is_training:
            return step > 0 and step % self.train_interval == 0
        else:
            return self._eval_episode % self.eval_interval == 0

    def _compute_weight_norms(self):
        """Compute per-module weight norms for CHNet."""
        norms = {}
        for name, p in self.model.named_parameters():
            if "chnet" not in name:
                continue
            module = name.split(".")[1] if "." in name else name  # e.g., "encoder" or "fusion"
            if module not in norms:
                norms[module] = 0.0
            norms[module] += p.data.norm().item() ** 2
        return {k: v ** 0.5 for k, v in norms.items()}

    def collect(self, step, depth_input, eagle_features_before, eagle_features_after,
                attn_weights=None, fastguide_attns=None, depth_tokens=None,
                rgb_pixels=None, is_training=True):
        """Collect intermediate tensors for diagnostics.

        All tensors are detached and moved to CPU to avoid holding GPU memory.
        Only call this when should_report() returns True.

        Args:
            step: Current training step or eval episode.
            depth_input: (B, 1, H, W) normalized depth tensor.
            eagle_features_before: (B, seq, D) eagle features before CHNet fusion.
            eagle_features_after: (B, seq, D) eagle features after CHNet fusion.
            attn_weights: Optional (B, num_heads, seq, depth_seq) cross-attention weights.
            fastguide_attns: Optional list of 4 (B, 1, H, W) spatial attention maps.
            depth_tokens: Optional (N, T, D) projected depth tokens before cross-attention.
            rgb_pixels: Optional (N, 3, H, W) RGB pixel values for overlay visualization.
            is_training: Whether in training or eval mode.
        """
        def _detach(t):
            return t.detach().cpu().float() if t is not None else None

        self._collected = {
            "step": step,
            "is_training": is_training,
            "depth_input": _detach(depth_input),
            "eagle_before": _detach(eagle_features_before),
            "eagle_after": _detach(eagle_features_after),
            "attn_weights": _detach(attn_weights),
            "fastguide_attns": [_detach(a) for a in fastguide_attns] if fastguide_attns else None,
            "depth_tokens": _detach(depth_tokens),
            "rgb_pixels": _detach(rgb_pixels),
        }

    def report(self, step, is_training=True):
        """Generate text summary and visualizations.

        Call after collect(). Prints health check to stdout and saves
        visualizations to {output_dir}/step_{N}/ or eval_episode_{N}/.
        """
        if not self._collected:
            return

        mode = "TRAIN" if is_training else "EVAL"
        data = self._collected

        # --- Text health check ---
        self._print_health_check(step, mode, data)

        # --- Visualizations ---
        if is_training:
            viz_dir = self.output_dir / f"step_{step}"
        else:
            viz_dir = self.output_dir / f"eval_episode_{self._eval_episode}"
        viz_dir.mkdir(parents=True, exist_ok=True)

        self._save_visualizations(viz_dir, data, mode, step)
        self._save_index_txt(viz_dir, mode, step)

        # Clear collected data
        self._collected = {}

    def _print_health_check(self, step, mode, data):
        """Print compact health check with expected values."""
        d = data["depth_input"]
        d_range = f"[{d.min().item():.3f}, {d.max().item():.3f}]"
        d_mean = d.mean().item()
        d_zeros = (d == 0).float().mean().item() * 100

        # Feature change ratio
        before = data["eagle_before"]
        after = data["eagle_after"]
        diff_norm = (after - before).norm().item()
        before_norm = before.norm().item()
        change_ratio = diff_norm / before_norm if before_norm > 0 else 0.0

        # Weight norms
        current_norms = self._compute_weight_norms()
        norm_str = " ".join(f"{k}={v:.1f}" for k, v in current_norms.items())

        # Weight drift from init
        drift_str = ""
        if self._initial_weight_norms:
            drifts = {}
            for k in current_norms:
                if k in self._initial_weight_norms and self._initial_weight_norms[k] > 0:
                    pct = abs(current_norms[k] - self._initial_weight_norms[k]) / self._initial_weight_norms[k] * 100
                    drifts[k] = pct
            drift_str = " drift: " + " ".join(f"{k}={v:.1f}%" for k, v in drifts.items())

        print(f"\n[DIAG step={step} {mode}]", flush=True)
        print(f"  depth_input:    range={d_range} mean={d_mean:.3f} zeros={d_zeros:.1f}%"
              f"    (expect: range~[0.01,5.0] for indoor sim, zeros<1%)", flush=True)
        print(f"  weight_norms:   {norm_str}{drift_str}"
              f"    (expect: stable; drift>50% from init=concern)", flush=True)
        print(f"  change_ratio:   {change_ratio:.6f}"
              f"    (expect: 0.001-0.1; <0.0001=depth ignored, >1.0=depth dominates)", flush=True)

    def log_gradients(self, step, model):
        """Log CHNet gradient norms. Call after backward pass.

        Args:
            step: Current training step.
            model: The model (to access .named_parameters()).
        """
        if step % self.train_interval != 0 or step == 0:
            return

        grad_norms = {}
        for name, p in model.named_parameters():
            if "chnet" not in name or p.grad is None:
                continue
            module = name.split(".")[1] if "." in name else name
            if module not in grad_norms:
                grad_norms[module] = 0.0
            grad_norms[module] += p.grad.norm().item() ** 2
        grad_norms = {k: v ** 0.5 for k, v in grad_norms.items()}

        grad_str = " ".join(f"{k}={v:.4f}" for k, v in grad_norms.items())
        print(f"  chnet_grad:     {grad_str}"
              f"    (expect: 0.001-1.0; <0.001=vanishing, >10=exploding)", flush=True)

    def _save_visualizations(self, viz_dir, data, mode, step):
        """Save all visualization PNGs."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
        except ImportError:
            print("[DIAG] matplotlib not available, skipping visualizations", flush=True)
            return

        # 1. Cross-attention map
        if data["attn_weights"] is not None:
            self._viz_cross_attn(viz_dir, data, plt)

        # 2. FastGuide spatial attention
        if data["fastguide_attns"] is not None:
            self._viz_fastguide(viz_dir, data, plt)

        # 3. Embedding similarity
        if data["depth_tokens"] is not None:
            self._viz_embedding_similarity(viz_dir, data, plt)

        # 4. Input overlay
        if data["rgb_pixels"] is not None:
            self._viz_input_overlay(viz_dir, data, plt)

        # 5. Depth contribution map
        self._viz_depth_contribution(viz_dir, data, plt)

        # 6. Proximity direction
        if data["depth_tokens"] is not None and data["depth_input"] is not None:
            self._viz_proximity_direction(viz_dir, data, plt)

        # 7. Token space PCA
        if data["depth_tokens"] is not None:
            self._viz_token_pca(viz_dir, data, plt)

        # 8. Nearest neighbors
        if data["depth_tokens"] is not None:
            self._viz_nearest_neighbors(viz_dir, data, plt)

        plt.close("all")

    def _viz_cross_attn(self, viz_dir, data, plt):
        """Cross-attention heatmap: which depth patches does eagle attend to."""
        attn = data["attn_weights"]  # (B, heads, seq_eagle, seq_depth)
        # Average over batch and heads
        attn_avg = attn[0].mean(dim=0)  # (seq_eagle, seq_depth)
        # Average over eagle query tokens to get per-depth-patch importance
        depth_importance = attn_avg.mean(dim=0)  # (seq_depth,)

        # Try to reshape to spatial grid (7x7 per view for default CHNET)
        n_tokens = depth_importance.shape[0]
        side = int(n_tokens ** 0.5)
        if side * side == n_tokens:
            heatmap = depth_importance.reshape(side, side).numpy()
        else:
            # Multi-view: try 7x7 per view
            tokens_per_view = 49  # 7x7
            n_views = n_tokens // tokens_per_view
            if n_views * tokens_per_view == n_tokens:
                heatmap = depth_importance[:tokens_per_view].reshape(7, 7).numpy()
            else:
                heatmap = depth_importance.unsqueeze(0).numpy()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
        ax.set_title("Cross-Attention: Eagle→Depth\n(bright = high attention)")
        plt.colorbar(im, ax=ax)
        fig.savefig(viz_dir / "cross_attn_map.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _viz_fastguide(self, viz_dir, data, plt):
        """FastGuide spatial attention at each CNN stage."""
        attns = data["fastguide_attns"]
        fig, axes = plt.subplots(1, len(attns), figsize=(4 * len(attns), 4))
        if len(attns) == 1:
            axes = [axes]
        sizes = [56, 28, 14, 7]
        for i, (attn, ax) in enumerate(zip(attns, axes)):
            # attn: (B, 1, H, W) - take first batch element
            heatmap = attn[0, 0].numpy()
            ax.imshow(heatmap, cmap="viridis", interpolation="nearest")
            sz = sizes[i] if i < len(sizes) else "?"
            ax.set_title(f"Stage {i+1} ({sz}x{sz})\nFastGuide spatial attention")
        fig.suptitle("FastGuide: RGB→Depth guidance\n(bright = high guidance weight)")
        fig.savefig(viz_dir / "fastguide_stages.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _viz_embedding_similarity(self, viz_dir, data, plt):
        """Cosine similarity heatmap: depth tokens vs eagle tokens."""
        depth_tok = data["depth_tokens"]  # (N, T_d, D)
        eagle_before = data["eagle_before"]  # (B, T_e, D)
        eagle_after = data["eagle_after"]

        # Use first batch/view element
        dt = depth_tok[0]  # (T_d, D)
        eb = eagle_before[0]  # (T_e, D)
        ea = eagle_after[0]

        # Normalize for cosine similarity
        dt_n = dt / (dt.norm(dim=-1, keepdim=True) + 1e-8)
        eb_n = eb / (eb.norm(dim=-1, keepdim=True) + 1e-8)
        ea_n = ea / (ea.norm(dim=-1, keepdim=True) + 1e-8)

        sim_before = (eb_n @ dt_n.T).numpy()  # (T_e, T_d)
        sim_after = (ea_n @ dt_n.T).numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        vmin = min(sim_before.min(), sim_after.min())
        vmax = max(sim_before.max(), sim_after.max())

        ax1.imshow(sim_before, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax1.set_title("Before fusion")
        ax1.set_xlabel("Depth tokens")
        ax1.set_ylabel("Eagle tokens")

        im = ax2.imshow(sim_after, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax2.set_title("After fusion")
        ax2.set_xlabel("Depth tokens")

        fig.suptitle("Cosine Similarity: Eagle↔Depth tokens\n(expect structured change after fusion)")
        plt.colorbar(im, ax=[ax1, ax2])
        fig.savefig(viz_dir / "embedding_similarity.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _viz_input_overlay(self, viz_dir, data, plt):
        """RGB | Depth (colormap) | Attention overlay."""
        rgb = data["rgb_pixels"][0].permute(1, 2, 0).numpy()  # (H, W, 3)
        depth = data["depth_input"][0, 0].numpy()  # (H, W)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Clamp RGB to [0,1] for display
        rgb_display = np.clip(rgb, 0, 1)
        axes[0].imshow(rgb_display)
        axes[0].set_title("RGB input")

        axes[1].imshow(depth, cmap="viridis")
        axes[1].set_title(f"Depth (meters)\nrange=[{depth.min():.2f}, {depth.max():.2f}]")

        # Overlay: cross-attention on depth if available
        if data["attn_weights"] is not None:
            attn = data["attn_weights"][0].mean(dim=0).mean(dim=0)  # (seq_depth,)
            n_tokens = attn.shape[0]
            side = int(n_tokens ** 0.5)
            if side * side == n_tokens:
                attn_map = attn.reshape(side, side).numpy()
            else:
                tokens_per_view = 49
                attn_map = attn[:tokens_per_view].reshape(7, 7).numpy() if n_tokens >= 49 else attn.unsqueeze(0).numpy()

            from PIL import Image
            # Resize attention to match depth spatial dims
            attn_resized = np.array(Image.fromarray(attn_map).resize(
                (depth.shape[1], depth.shape[0]), Image.BILINEAR
            ))
            axes[2].imshow(depth, cmap="viridis")
            axes[2].imshow(attn_resized, cmap="hot", alpha=0.5)
            axes[2].set_title("Attention overlaid on depth\n(red = high attention)")
        else:
            axes[2].imshow(depth, cmap="viridis")
            axes[2].set_title("(No attention data)")

        fig.suptitle("Input & Attention Overlay")
        fig.savefig(viz_dir / "input_overlay.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _viz_depth_contribution(self, viz_dir, data, plt):
        """Per-token magnitude of eagle_after - eagle_before, shown spatially."""
        before = data["eagle_before"][0]  # (T, D)
        after = data["eagle_after"][0]
        delta_norm = (after - before).norm(dim=-1)  # (T,)

        # Try to reshape to spatial grid
        n_tokens = delta_norm.shape[0]
        side = int(n_tokens ** 0.5)
        if side * side == n_tokens:
            spatial = delta_norm.reshape(side, side).numpy()
        else:
            spatial = delta_norm.unsqueeze(0).numpy()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.imshow(spatial, cmap="magma", interpolation="nearest")
        ax.set_title("Depth contribution per eagle token\n"
                      "(bright = depth changed this token a lot)")
        plt.colorbar(im, ax=ax)
        fig.savefig(viz_dir / "depth_contribution_map.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _viz_proximity_direction(self, viz_dir, data, plt):
        """Project eagle tokens onto near-far depth direction."""
        depth_tok = data["depth_tokens"][0]  # (T_d, D)
        depth_input = data["depth_input"][0, 0]  # (H, W)
        eagle_after = data["eagle_after"][0]  # (T_e, D)

        # Get per-depth-token average depth value
        # depth_tokens correspond to 7x7 spatial grid
        n_depth = depth_tok.shape[0]
        side = int(n_depth ** 0.5)
        if side * side != n_depth:
            # Can't do spatial mapping, skip
            return

        # Downsample depth to match token grid
        from PIL import Image
        depth_small = np.array(Image.fromarray(depth_input.numpy()).resize(
            (side, side), Image.BILINEAR
        ))
        depth_flat = torch.tensor(depth_small.flatten())

        # Near vs far quartiles
        sorted_idx = depth_flat.argsort()
        n_quartile = max(1, n_depth // 4)
        near_idx = sorted_idx[:n_quartile]  # smallest depth = nearest
        far_idx = sorted_idx[-n_quartile:]  # largest depth = farthest

        near_mean = depth_tok[near_idx].mean(dim=0)
        far_mean = depth_tok[far_idx].mean(dim=0)
        direction = near_mean - far_mean
        dir_norm = direction.norm()
        if dir_norm < 1e-8:
            return

        # Project eagle tokens onto this direction
        scores = (eagle_after @ direction) / dir_norm  # (T_e,)
        n_eagle = scores.shape[0]
        side_e = int(n_eagle ** 0.5)
        if side_e * side_e == n_eagle:
            spatial = scores.reshape(side_e, side_e).numpy()
        else:
            spatial = scores.unsqueeze(0).numpy()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.imshow(spatial, cmap="RdBu_r", interpolation="nearest")
        ax.set_title("Proximity direction projection\n"
                      "(red=model thinks 'close', blue='far')\n"
                      "Compare with actual depth image")
        plt.colorbar(im, ax=ax)
        fig.savefig(viz_dir / "proximity_direction.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _viz_token_pca(self, viz_dir, data, plt):
        """PCA of depth + eagle tokens colored by modality and depth value."""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("[DIAG] sklearn not available, skipping PCA visualization", flush=True)
            return

        depth_tok = data["depth_tokens"][0]  # (T_d, D)
        eagle_after = data["eagle_after"][0]  # (T_e, D)

        # Concatenate
        all_tokens = torch.cat([eagle_after, depth_tok], dim=0).numpy()
        n_eagle = eagle_after.shape[0]
        n_depth = depth_tok.shape[0]

        # PCA to 2D
        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_tokens)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Eagle tokens: circles
        ax.scatter(coords[:n_eagle, 0], coords[:n_eagle, 1],
                   c="steelblue", marker="o", alpha=0.4, s=15, label="Eagle tokens")

        # Depth tokens: triangles, colored by depth value
        depth_input = data["depth_input"][0, 0]  # (H, W)
        side = int(n_depth ** 0.5)
        if side * side == n_depth:
            from PIL import Image
            depth_small = np.array(Image.fromarray(depth_input.numpy()).resize(
                (side, side), Image.BILINEAR
            ))
            depth_values = depth_small.flatten()
        else:
            depth_values = np.zeros(n_depth)

        sc = ax.scatter(coords[n_eagle:, 0], coords[n_eagle:, 1],
                        c=depth_values, cmap="viridis", marker="^", alpha=0.7, s=25,
                        label="Depth tokens")
        plt.colorbar(sc, ax=ax, label="Depth value (m)")

        ax.set_title(f"Token space PCA (var explained: {pca.explained_variance_ratio_.sum():.1%})\n"
                      "Circle=eagle, Triangle=depth (color=depth value)")
        ax.legend()
        fig.savefig(viz_dir / "token_space_pca.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _viz_nearest_neighbors(self, viz_dir, data, plt):
        """Arrow diagram: each depth token → its nearest eagle token."""
        depth_tok = data["depth_tokens"][0]  # (T_d, D)
        eagle_after = data["eagle_after"][0]  # (T_e, D)

        # Cosine similarity
        dt_n = depth_tok / (depth_tok.norm(dim=-1, keepdim=True) + 1e-8)
        ea_n = eagle_after / (eagle_after.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (dt_n @ ea_n.T)  # (T_d, T_e)

        # Nearest eagle token for each depth token
        nearest = sim.argmax(dim=1)  # (T_d,)

        n_depth = depth_tok.shape[0]
        n_eagle = eagle_after.shape[0]
        side_d = int(n_depth ** 0.5)
        side_e = int(n_eagle ** 0.5)
        if side_d * side_d != n_depth or side_e * side_e != n_eagle:
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Plot grid points
        for i in range(n_depth):
            dy, dx = divmod(i, side_d)
            # Scale depth coords to [0, 1]
            x_d = dx / max(side_d - 1, 1)
            y_d = dy / max(side_d - 1, 1)

            j = nearest[i].item()
            ey, ex = divmod(j, side_e)
            x_e = ex / max(side_e - 1, 1)
            y_e = ey / max(side_e - 1, 1)

            ax.annotate("", xy=(x_e, y_e), xytext=(x_d, y_d),
                         arrowprops=dict(arrowstyle="->", color="red", alpha=0.3, lw=0.5))

        # Scatter points
        d_coords = np.array([[dx / max(side_d-1,1), dy / max(side_d-1,1)]
                             for dy in range(side_d) for dx in range(side_d)])
        e_coords = np.array([[ex / max(side_e-1,1), ey / max(side_e-1,1)]
                             for ey in range(side_e) for ex in range(side_e)])

        ax.scatter(d_coords[:, 0], d_coords[:, 1], c="orange", marker="^", s=30,
                   label="Depth patches", zorder=5)
        ax.scatter(e_coords[:, 0], e_coords[:, 1], c="steelblue", marker="o", s=20,
                   label="Eagle patches", zorder=5)

        ax.set_title("Nearest neighbor: Depth→Eagle (by cosine sim)\n"
                      "Arrows show cross-modal alignment\n"
                      "(short arrows = good spatial correspondence)")
        ax.legend()
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.invert_yaxis()
        fig.savefig(viz_dir / "nearest_neighbors.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _save_index_txt(self, viz_dir, mode, step):
        """Write index.txt documenting all visualizations."""
        text = f"""=== GROOT Depth Diagnostics - Step {step} ({mode}) ===

1. cross_attn_map.png
   WHAT: Heatmap of cross-attention weights (Q=eagle_tokens, K/V=depth_tokens).
   HOW TO READ: Bright regions = depth locations the model attends to most.
         Reshaped to 7x7 spatial grid per camera view.
   EXPECT: Highlights on object surfaces, edges, and task-relevant regions.
   CONCERN IF: Uniform attention (depth structure ignored) or all-zero.

2. fastguide_stages.png
   WHAT: Spatial attention maps from FastGuide at each of 4 CNN encoder stages.
   HOW TO READ: Bright = high attention. Stage 1-2 are higher resolution (edges).
         Stage 3-4 are lower resolution (semantic regions).
   EXPECT: Structured patterns; early stages show edges, later stages show objects.
   CONCERN IF: Uniform or zero at any stage.

3. embedding_similarity.png
   WHAT: Cosine similarity between depth tokens and eagle tokens (before/after fusion).
   HOW TO READ: Two heatmaps side-by-side. Rows=eagle tokens, cols=depth tokens.
         Bright cells = high similarity.
   EXPECT: After fusion, structured similarity increases at relevant token pairs.
   CONCERN IF: No difference before vs after, or completely uniform.

4. input_overlay.png
   WHAT: RGB | depth (viridis colormap) | attention overlaid on depth.
   HOW TO READ: Third panel shows where the cross-attention focuses spatially.
   EXPECT: Attention on task-relevant objects and surfaces.
   CONCERN IF: Attention on background or spatially uniform.

5. depth_contribution_map.png
   WHAT: Per-token magnitude of (eagle_after_fusion - eagle_before_fusion), shown spatially.
   HOW TO READ: Bright = depth changed this token a lot. Spatial layout matches eagle grid.
   EXPECT: High on objects/surfaces, low on empty background.
   CONCERN IF: Uniform or near-zero everywhere.

6. proximity_direction.png
   WHAT: Eagle tokens projected onto the near-far depth direction.
   HOW TO READ: Red = model thinks "close", blue = model thinks "far". Compare with actual depth.
   EXPECT: Should correlate with real depth values. Red near objects, blue on background.
   CONCERN IF: No correlation with actual depth, or uniform.

7. token_space_pca.png
   WHAT: PCA of depth and eagle tokens in 2D. Circles=eagle, triangles=depth.
   HOW TO READ: Color = depth value (near=warm, far=cool). Check clustering.
   EXPECT: Some modality clustering with overlap at task-relevant regions.
   CONCERN IF: Complete separation (no interaction) or complete overlap (depth redundant).

8. nearest_neighbors.png
   WHAT: Arrows from each depth patch to its most similar eagle patch.
   HOW TO READ: Arrows show cross-modal alignment. Most should be short (same location).
   EXPECT: Spatial correspondence - arrows mostly point to same/nearby locations.
   CONCERN IF: Random directions (no learned correspondence).
"""
        with open(viz_dir / "index.txt", "w") as f:
            f.write(text)

    def increment_eval_episode(self):
        """Call after each eval episode completes."""
        self._eval_episode += 1
```

**Step 2: Verify the module imports cleanly**

Run: `cd /home/ofekpear/depth_vla_chnet/lerobot && python -c "from lerobot.policies.groot.chnet_diagnostics import DepthDiagnostics; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py
git commit -m "feat: add DepthDiagnostics class with startup banner and visualizations"
```

---

### Task 2: Add `return_diagnostics` to `chnet_modules.py`

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_modules.py:50-88` (FastGuide)
- Modify: `src/lerobot/policies/groot/chnet_modules.py:134-196` (DepthCNNEncoder)
- Modify: `src/lerobot/policies/groot/chnet_modules.py:199-237` (DepthCrossAttentionFusion)
- Modify: `src/lerobot/policies/groot/chnet_modules.py:240-268` (CHNetDepthProcessor)

**Step 1: Modify FastGuide.forward to optionally return spatial attention**

In `chnet_modules.py`, change the `FastGuide.forward` method:

```python
def forward(self, depth_feat, rgb_feat, return_diagnostics=False):
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
```

**Step 2: Modify DepthCNNEncoder.forward to pass through and collect FastGuide attentions**

```python
def forward(self, depth, vit_features, grid_h, grid_w, return_diagnostics=False):
    fastguide_attns = []
    x = self.stem(depth)

    x = self.stage1(x)
    rgb1 = self.proj1(vit_features[0], grid_h, grid_w)
    if return_diagnostics:
        x, attn1 = self.guide1(x, rgb1, return_diagnostics=True)
        fastguide_attns.append(attn1)
    else:
        x = self.guide1(x, rgb1)

    x = self.stage2(x)
    rgb2 = self.proj2(vit_features[1], grid_h, grid_w)
    if return_diagnostics:
        x, attn2 = self.guide2(x, rgb2, return_diagnostics=True)
        fastguide_attns.append(attn2)
    else:
        x = self.guide2(x, rgb2)

    x = self.stage3(x)
    rgb3 = self.proj3(vit_features[2], grid_h, grid_w)
    if return_diagnostics:
        x, attn3 = self.guide3(x, rgb3, return_diagnostics=True)
        fastguide_attns.append(attn3)
    else:
        x = self.guide3(x, rgb3)

    x = self.stage4(x)
    rgb4 = self.proj4(vit_features[3], grid_h, grid_w)
    if return_diagnostics:
        x, attn4 = self.guide4(x, rgb4, return_diagnostics=True)
        fastguide_attns.append(attn4)
    else:
        x = self.guide4(x, rgb4)

    if return_diagnostics:
        return x, fastguide_attns
    return x
```

**Step 3: Modify DepthCrossAttentionFusion.forward to optionally return attention weights and projected depth tokens**

```python
def forward(self, eagle_features, depth_features, return_diagnostics=False):
    n, c, h, w = depth_features.shape
    b_eagle = eagle_features.shape[0]
    depth_tokens = depth_features.flatten(2).transpose(1, 2)  # (N, H*W, C)
    depth_tokens = self.depth_proj(depth_tokens)  # (N, H*W, hidden_dim)

    if n != b_eagle:
        num_views = n // b_eagle
        tokens_per_view = depth_tokens.shape[1]
        depth_tokens = depth_tokens.view(b_eagle, num_views * tokens_per_view, -1)

    # Cross-attention: Q=eagle, K=depth, V=depth
    if return_diagnostics:
        attn_out, attn_weights = self.cross_attn(
            query=eagle_features,
            key=depth_tokens,
            value=depth_tokens,
            need_weights=True,
            average_attn_weights=False,  # keep per-head weights
        )
        result = self.norm(eagle_features + attn_out)
        return result, attn_weights, depth_tokens
    else:
        attn_out, _ = self.cross_attn(
            query=eagle_features,
            key=depth_tokens,
            value=depth_tokens,
        )
        return self.norm(eagle_features + attn_out)
```

**Step 4: Modify CHNetDepthProcessor.forward to orchestrate diagnostics return**

```python
def forward(self, depth, vit_features, grid_h, grid_w, eagle_features, return_diagnostics=False):
    if return_diagnostics:
        depth_feat, fastguide_attns = self.encoder(depth, vit_features, grid_h, grid_w, return_diagnostics=True)
        result, attn_weights, depth_tokens = self.fusion(eagle_features, depth_feat, return_diagnostics=True)
        return result, {
            "attn_weights": attn_weights,
            "fastguide_attns": fastguide_attns,
            "depth_tokens": depth_tokens,
        }
    else:
        depth_feat = self.encoder(depth, vit_features, grid_h, grid_w)
        return self.fusion(eagle_features, depth_feat)
```

**Step 5: Verify module still works**

Run: `cd /home/ofekpear/depth_vla_chnet/lerobot && python -c "
from lerobot.policies.groot.chnet_modules import CHNetDepthProcessor
import torch
proc = CHNetDepthProcessor(vit_dim=1152, hidden_dim=2048)
depth = torch.randn(2, 1, 224, 224)
vit_feats = [torch.randn(2, 256, 1152)] * 4
eagle = torch.randn(2, 256, 2048)
# Without diagnostics
out = proc(depth, vit_feats, 16, 16, eagle)
print('Without diagnostics:', out.shape)
# With diagnostics
out2, diag = proc(depth, vit_feats, 16, 16, eagle, return_diagnostics=True)
print('With diagnostics:', out2.shape, list(diag.keys()))
print('OK')
"`
Expected: Both calls succeed, shapes match, diag has expected keys.

**Step 6: Commit**

```bash
git add src/lerobot/policies/groot/chnet_modules.py
git commit -m "feat: add return_diagnostics flag to CHNET modules for interpretability"
```

---

### Task 3: Integrate diagnostics into `groot_n1.py` forward_eagle

**Files:**
- Modify: `src/lerobot/policies/groot/groot_n1.py:443-535` (forward_eagle)
- Modify: `src/lerobot/policies/groot/groot_n1.py:58-75` (EagleBackbone.__init__)

**Step 1: Add diagnostics attribute to EagleBackbone.__init__**

After line 139 (after CHNet initialization), add:

```python
# Diagnostics module (set externally after model creation)
self._diagnostics = None
```

**Step 2: Modify forward_eagle to use diagnostics**

Replace the existing CHNet section (lines 478-532) with diagnostics-aware code. The key changes:

1. When `self._diagnostics` is set and `should_report()` returns True, call CHNet with `return_diagnostics=True`
2. Call `diagnostics.collect()` with all intermediate tensors
3. Replace the existing inline print statements with diagnostics calls
4. Keep the existing RuntimeError checks (all-zero and unchanged features) — these are critical safety checks, not diagnostics

```python
# In forward_eagle, replace lines 478-532:
if depth_normalized is not None:
    vit_feats = [self._vit_hook_features[idx] for idx in self._chnet_tap_layers]

    pixel_values = eagle_input.get("pixel_values")
    if pixel_values is not None:
        _, _, h_pv, w_pv = pixel_values.shape
        patch_size = 14
        grid_h = h_pv // patch_size
        grid_w = w_pv // patch_size
    else:
        grid_h = grid_w = 16

    if depth_normalized.shape[-1] != 224 or depth_normalized.shape[-2] != 224:
        depth_normalized = F.interpolate(depth_normalized, size=(224, 224), mode='nearest')

    eagle_features_before = eagle_features

    # Check if diagnostics should collect data this step
    do_diag = (self._diagnostics is not None
               and self._diagnostics.should_report(self._depth_fwd_count, is_training=self.training))

    if do_diag:
        eagle_features, diag_data = self.chnet(
            depth=depth_normalized,
            vit_features=vit_feats,
            grid_h=grid_h,
            grid_w=grid_w,
            eagle_features=eagle_features,
            return_diagnostics=True,
        )
        self._diagnostics.collect(
            step=self._depth_fwd_count,
            depth_input=depth_normalized,
            eagle_features_before=eagle_features_before,
            eagle_features_after=eagle_features,
            attn_weights=diag_data["attn_weights"],
            fastguide_attns=diag_data["fastguide_attns"],
            depth_tokens=diag_data["depth_tokens"],
            rgb_pixels=pixel_values,
            is_training=self.training,
        )
        self._diagnostics.report(self._depth_fwd_count, is_training=self.training)
    else:
        eagle_features = self.chnet(
            depth=depth_normalized,
            vit_features=vit_feats,
            grid_h=grid_h,
            grid_w=grid_w,
            eagle_features=eagle_features,
        )

    # Safety checks (always active, regardless of diagnostics)
    diff = (eagle_features - eagle_features_before).abs()
    diff_norm = diff.norm().item()
    if eagle_features.abs().max().item() == 0.0:
        raise RuntimeError(
            "[GROOT] CHNet output is all zeros! Cross-attention fusion produced empty features."
        )
    if diff_norm == 0.0:
        raise RuntimeError(
            "[GROOT] CHNet did not change eagle_features at all! "
            "Cross-attention residual is zero — depth signal is not being fused."
        )
```

**Step 3: Remove old inline print statements**

Remove the old logging at lines 464-470 and 524-532 (the ones that print depth shape/range and change_ratio every 5000 steps). These are now handled by the diagnostics module. Keep the RuntimeError checks.

**Step 4: Commit**

```bash
git add src/lerobot/policies/groot/groot_n1.py
git commit -m "feat: integrate DepthDiagnostics into EagleBackbone forward pass"
```

---

### Task 4: Wire diagnostics into `modeling_groot.py`

**Files:**
- Modify: `src/lerobot/policies/groot/modeling_groot.py:52-61` (__init__)
- Modify: `src/lerobot/policies/groot/modeling_groot.py:63-102` (_create_groot_model)

**Step 1: Initialize diagnostics in GrootPolicy.__init__**

After `self._groot_model = self._create_groot_model()` (line 59), add:

```python
# Initialize diagnostics if depth is enabled
self._diagnostics = None
if self.config.use_depth:
    from lerobot.policies.groot.chnet_diagnostics import DepthDiagnostics
    # output_dir will be set externally before training starts
    self._diagnostics_cls = DepthDiagnostics
```

**Step 2: Add a method to initialize diagnostics with output_dir**

After the `reset()` method, add:

```python
def init_diagnostics(self, output_dir, dataset_info=None):
    """Initialize depth diagnostics. Call before training/eval starts.

    Args:
        output_dir: Path to output directory (e.g., cfg.output_dir).
        dataset_info: Optional dict with 'camera_names' for validation.
    """
    if not self.config.use_depth:
        return
    from lerobot.policies.groot.chnet_diagnostics import DepthDiagnostics
    backbone = self._groot_model.backbone
    self._diagnostics = DepthDiagnostics(
        config=self.config,
        model=backbone,
        output_dir=output_dir,
    )
    backbone._diagnostics = self._diagnostics
    self._diagnostics.print_startup_banner(dataset_info=dataset_info)
```

**Step 3: Commit**

```bash
git add src/lerobot/policies/groot/modeling_groot.py
git commit -m "feat: wire DepthDiagnostics initialization into GrootPolicy"
```

---

### Task 5: Hook diagnostics into training loop

**Files:**
- Modify: `src/lerobot/scripts/lerobot_train.py:405-430` (after policy creation, before training loop)
- Modify: `src/lerobot/scripts/lerobot_train.py:494-507` (after update_policy, for gradient logging)

**Step 1: Initialize diagnostics after policy creation**

After line 428 (`logging.info(f"{num_total_params=}...")`), add:

```python
# Initialize depth diagnostics if policy supports it
unwrapped_policy = policy if not hasattr(policy, "module") else policy.module
if hasattr(unwrapped_policy, "init_diagnostics"):
    camera_names = [k for k in dataset.meta.features if "image" in k and "depth" not in k]
    unwrapped_policy.init_diagnostics(
        output_dir=cfg.output_dir,
        dataset_info={"camera_names": camera_names},
    )
```

**Step 2: Add gradient logging after backward pass**

After line 507 (`step += 1`), add:

```python
# Log CHNet gradient norms at diagnostic intervals
unwrapped = accelerator.unwrap_model(policy)
if hasattr(unwrapped, "_diagnostics") and unwrapped._diagnostics is not None:
    unwrapped._diagnostics.log_gradients(step, unwrapped._groot_model.backbone)
```

**Step 3: Commit**

```bash
git add src/lerobot/scripts/lerobot_train.py
git commit -m "feat: hook depth diagnostics into training loop"
```

---

### Task 6: Hook diagnostics into eval loop

**Files:**
- Modify: `src/lerobot/scripts/lerobot_eval.py:252-300` (eval_policy setup)

**Step 1: Add eval episode counting**

In the `rollout()` function, after the episode loop completes (around line 226), we need to increment the eval episode counter. However, the diagnostics are triggered from within the model's forward pass (same as training), so we mainly need to ensure the diagnostics object is initialized before eval.

The diagnostics are already triggered via `should_report()` in `forward_eagle()`. For eval, we just need to make sure `init_diagnostics` is called. In the training loop flow, eval happens after training has started, so diagnostics are already initialized.

For standalone eval (running `lerobot_eval.py` directly), add initialization in `eval_policy()`:

After `policy.eval()` (around line 298 of `lerobot_eval.py`), add:

```python
# Initialize depth diagnostics for standalone eval if not already done
if hasattr(policy, "init_diagnostics") and not hasattr(policy, "_diagnostics") or policy._diagnostics is None:
    eval_output_dir = videos_dir.parent if videos_dir else Path("./eval_diagnostics")
    policy.init_diagnostics(output_dir=eval_output_dir)
```

**Step 2: Commit**

```bash
git add src/lerobot/scripts/lerobot_eval.py
git commit -m "feat: hook depth diagnostics into eval loop"
```

---

### Task 7: End-to-end verification

**Step 1: Verify import chain**

Run: `cd /home/ofekpear/depth_vla_chnet/lerobot && python -c "
from lerobot.policies.groot.chnet_diagnostics import DepthDiagnostics
from lerobot.policies.groot.chnet_modules import CHNetDepthProcessor
print('All imports OK')
"`

**Step 2: Verify CHNet diagnostics return path**

Run: `cd /home/ofekpear/depth_vla_chnet/lerobot && python -c "
import torch
from lerobot.policies.groot.chnet_modules import CHNetDepthProcessor

proc = CHNetDepthProcessor(vit_dim=1152, hidden_dim=2048)
depth = torch.randn(2, 1, 224, 224)
vit_feats = [torch.randn(2, 256, 1152)] * 4
eagle = torch.randn(2, 256, 2048)

# Normal path
out = proc(depth, vit_feats, 16, 16, eagle)
print(f'Normal: {out.shape}')

# Diagnostics path
out2, diag = proc(depth, vit_feats, 16, 16, eagle, return_diagnostics=True)
print(f'Diag: {out2.shape}')
print(f'  attn_weights: {diag[\"attn_weights\"].shape}')
print(f'  fastguide_attns: {len(diag[\"fastguide_attns\"])} x {diag[\"fastguide_attns\"][0].shape}')
print(f'  depth_tokens: {diag[\"depth_tokens\"].shape}')
print('OK')
"`

**Step 3: Verify DepthDiagnostics standalone**

Run: `cd /home/ofekpear/depth_vla_chnet/lerobot && python -c "
import torch
from lerobot.policies.groot.chnet_diagnostics import DepthDiagnostics
from lerobot.policies.groot.chnet_modules import CHNetDepthProcessor

proc = CHNetDepthProcessor(vit_dim=1152, hidden_dim=2048)

# Mock config
class MockConfig:
    use_depth = True
    depth_scale = 1.0
    depth_mean = 0.0
    depth_std = 1.0
    chnet_tap_layers = (5, 11, 17, 23)
    chnet_channels = (64, 128, 256, 256)

diag = DepthDiagnostics(MockConfig(), proc, '/tmp/test_diag', train_interval=1)
diag.print_startup_banner()

# Simulate data
depth = torch.randn(2, 1, 224, 224).abs()  # positive depth values
eagle_before = torch.randn(2, 256, 2048)
eagle_after = eagle_before + 0.01 * torch.randn_like(eagle_before)
attn_weights = torch.randn(2, 8, 256, 49).softmax(dim=-1)
fastguide_attns = [torch.randn(2, 1, s, s).abs() for s in [56, 28, 14, 7]]
depth_tokens = torch.randn(2, 49, 2048)
rgb = torch.randn(2, 3, 224, 224).clamp(0, 1)

diag.collect(1, depth, eagle_before, eagle_after,
             attn_weights, fastguide_attns, depth_tokens, rgb)
diag.report(1)
print('Check /tmp/test_diag for output files')
"`

**Step 4: Check output files were created**

Run: `ls -la /tmp/test_diag/step_1/`
Expected: `index.txt`, `cross_attn_map.png`, `fastguide_stages.png`, `embedding_similarity.png`, `input_overlay.png`, `depth_contribution_map.png`, `proximity_direction.png`, `token_space_pca.png`, `nearest_neighbors.png`

**Step 5: Commit final state**

```bash
git add -A
git commit -m "feat: complete CHNET depth diagnostics and interpretability layer"
```

---

### Task Summary

| Task | What | Files |
|------|------|-------|
| 1 | DepthDiagnostics class (banner, health checks, all visualizations) | New: `chnet_diagnostics.py` |
| 2 | `return_diagnostics` flag in CHNET modules | Modify: `chnet_modules.py` |
| 3 | Integrate into `forward_eagle`, replace inline prints | Modify: `groot_n1.py` |
| 4 | Wire init into GrootPolicy | Modify: `modeling_groot.py` |
| 5 | Hook into training loop (init + gradient logging) | Modify: `lerobot_train.py` |
| 6 | Hook into eval loop | Modify: `lerobot_eval.py` |
| 7 | End-to-end verification | Verification scripts |
