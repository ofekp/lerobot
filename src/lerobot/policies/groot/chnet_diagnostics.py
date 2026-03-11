"""CHNet Depth Diagnostics for GR00T.

Provides startup banner, periodic health checks, gradient logging,
and 8 visualization methods for understanding the CHNet depth pipeline.
"""

import datetime
import os
import subprocess
import textwrap
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Matplotlib with non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

# Optional dependencies
try:
    from sklearn.decomposition import PCA as SklearnPCA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _git_info():
    """Return (short_hash, branch) or ('unknown', 'unknown')."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha, branch
    except Exception:
        return "unknown", "unknown"


def _safe_detach(t):
    """Detach, move to CPU, convert to float32. Returns None if input is None."""
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float()
    return t


def _count_params(module):
    """Return (total_params, trainable_params) for an nn.Module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _viridis_colormap(values, vmin=None, vmax=None):
    """Apply viridis colormap to a 2D numpy array, return (H, W, 3) uint8."""
    cm = plt.cm.viridis
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    mapped = cm(norm(values))[:, :, :3]  # drop alpha
    return (mapped * 255).astype(np.uint8)


class DepthDiagnostics:
    """Diagnostics for the CHNet depth processing pipeline.

    Provides:
      - Startup banner with config, git info, param counts, camera info
      - Periodic health checks (depth stats, weight norms, drift, change ratio)
      - Gradient logging per CHNet sub-module
      - 8 visualization methods with annotated index.txt
    """

    def __init__(self, config, model, output_dir, train_interval=500, eval_interval=1):
        """
        Args:
            config: GrootConfig instance with depth-related fields.
            model: The GR00TN15 model (has .backbone with EagleBackbone).
            output_dir: Root directory for diagnostics output.
            train_interval: Report every N training steps.
            eval_interval: Report every N eval episodes.
        """
        self.config = config
        self.model = model
        self.output_dir = Path(output_dir) / "diagnostics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_interval = train_interval
        self.eval_interval = eval_interval

        # State
        self._collected = {}
        self._initial_weights = {}
        self._eval_episode = 0
        self._startup_printed = False
        self._hooks_verified = False
        self._prev_change_ratio = None

        # Snapshot initial CHNet weight norms for drift tracking
        self._snapshot_initial_weights()

    # ------------------------------------------------------------------
    # Weight snapshot for drift tracking
    # ------------------------------------------------------------------
    def _snapshot_initial_weights(self):
        """Snapshot L2 norms of CHNet sub-module weights at init."""
        backbone = self.model
        if backbone.chnet is None:
            return
        for name, param in backbone.chnet.named_parameters():
            self._initial_weights[name] = param.detach().cpu().float().norm().item()

    # ------------------------------------------------------------------
    # Startup banner
    # ------------------------------------------------------------------
    def print_startup_banner(self, dataset_info=None):
        """Print a detailed startup banner and save to file."""
        if self._startup_printed:
            return
        self._startup_printed = True

        git_hash, git_branch = _git_info()
        backbone = self.model
        chnet = backbone.chnet

        lines = []
        lines.append("=" * 72)
        lines.append("  CHNet Depth Diagnostics — Startup Banner")
        lines.append("=" * 72)
        lines.append(f"  Timestamp       : {datetime.datetime.now().isoformat()}")
        lines.append(f"  Git hash        : {git_hash}")
        lines.append(f"  Git branch      : {git_branch}")
        lines.append("")
        lines.append("  --- Config ---")
        lines.append(f"  use_depth       : {self.config.use_depth}")
        lines.append(f"  depth_scale     : {self.config.depth_scale}  (expected: 0.001-1.0)")
        lines.append(f"  depth_mean      : {self.config.depth_mean}  (expected: 0.0-2.0)")
        lines.append(f"  depth_std       : {self.config.depth_std}  (expected: 0.1-2.0)")
        lines.append(f"  chnet_tap_layers: {self.config.chnet_tap_layers}")
        lines.append(f"  chnet_channels  : {self.config.chnet_channels}")
        lines.append(f"  train_interval  : {self.train_interval}")
        lines.append(f"  eval_interval   : {self.eval_interval}")
        lines.append("")

        lines.append("  --- Parameter Counts ---")
        if chnet is not None:
            total, trainable = _count_params(chnet)
            lines.append(f"  CHNet total     : {total:,}  (expected: 5M-20M)")
            lines.append(f"  CHNet trainable : {trainable:,}")

            # Sub-module counts
            for sub_name in ["encoder", "fusion"]:
                sub = getattr(chnet, sub_name, None)
                if sub is not None:
                    st, sr = _count_params(sub)
                    lines.append(f"    {sub_name:14s}: {st:>10,} total, {sr:>10,} trainable")

            # Encoder sub-modules
            enc = chnet.encoder
            for stage_name in ["stem", "stage1", "stage2", "stage3", "stage4"]:
                stage = getattr(enc, stage_name, None)
                if stage is not None:
                    st, _ = _count_params(stage)
                    lines.append(f"      {stage_name:12s}: {st:>10,}")
            for g_name in ["guide1", "guide2", "guide3", "guide4"]:
                guide = getattr(enc, g_name, None)
                if guide is not None:
                    st, _ = _count_params(guide)
                    lines.append(f"      {g_name:12s}: {st:>10,}")
            for p_name in ["proj1", "proj2", "proj3", "proj4"]:
                proj = getattr(enc, p_name, None)
                if proj is not None:
                    st, _ = _count_params(proj)
                    lines.append(f"      {p_name:12s}: {st:>10,}")
        else:
            lines.append("  CHNet           : DISABLED")

        lines.append("")
        model_total, model_train = _count_params(self.model)
        lines.append(f"  Full model total    : {model_total:,}")
        lines.append(f"  Full model trainable: {model_train:,}")

        # Camera info from config features
        lines.append("")
        lines.append("  --- Camera Info ---")
        if hasattr(self.config, "input_features"):
            for k, feat in self.config.input_features.items():
                if hasattr(feat, "type") and str(feat.type).endswith("VISUAL"):
                    lines.append(f"  {k}: shape={feat.shape}")

        # Dataset info
        if dataset_info is not None:
            lines.append("")
            lines.append("  --- Dataset Info ---")
            for k, v in dataset_info.items():
                lines.append(f"  {k}: {v}")

        # Trainability audit
        lines.append("")
        lines.append("  --- Trainability Audit ---")
        if chnet is not None:
            frozen_submods = []
            for sub_name in ["encoder", "fusion"]:
                sub = getattr(chnet, sub_name, None)
                if sub is not None:
                    n_trainable = sum(1 for p in sub.parameters() if p.requires_grad)
                    n_total = sum(1 for p in sub.parameters())
                    if n_trainable == 0:
                        frozen_submods.append(sub_name)
                    lines.append(f"  {sub_name:14s}: {n_trainable}/{n_total} params trainable")
            if frozen_submods:
                lines.append(f"  !! WARNING: {frozen_submods} have NO trainable params — depth won't learn!")

        # BatchNorm mode check
        lines.append("")
        lines.append("  --- BatchNorm Status ---")
        if chnet is not None:
            bn_count = 0
            bn_eval = 0
            for name, mod in chnet.named_modules():
                if isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    bn_count += 1
                    if not mod.training:
                        bn_eval += 1
            lines.append(f"  BN layers: {bn_count} total, {bn_eval} in eval mode")
            if bn_eval > 0:
                lines.append(f"  !! WARNING: {bn_eval} BatchNorm layers stuck in eval mode — stats won't update!")

        # Expected values summary
        lines.append("")
        lines.append("  --- Expected Values During Training ---")
        lines.append("  depth_input range   : [0.0, ~5.0] meters  (after scale/mean/std)")
        lines.append("  change_ratio        : 0.001-0.05  (how much CHNet modifies eagle features)")
        lines.append("  CHNet weight norm   : 1.0-50.0 per sub-module")
        lines.append("  CHNet grad norm     : 1e-5 to 1e-1 per sub-module")
        lines.append("  weight drift        : <5% in first 1000 steps, growing slowly")
        lines.append("=" * 72)

        banner = "\n".join(lines)
        print(banner, flush=True)

        # Save to file
        banner_path = self.output_dir / "startup_banner.txt"
        banner_path.write_text(banner + "\n")

    # ------------------------------------------------------------------
    # ViT hooks verification (call once after first forward pass)
    # ------------------------------------------------------------------
    def verify_hooks(self):
        """Verify ViT hooks captured features correctly. Call after first forward."""
        if self._hooks_verified:
            return
        self._hooks_verified = True

        backbone = self.model
        hook_feats = getattr(backbone, "_vit_hook_features", {})
        tap_layers = getattr(backbone, "_chnet_tap_layers", ())

        lines = ["[CHNet Hook Verification]"]
        if not tap_layers:
            lines.append("  !! WARNING: No tap layers configured — CHNet won't receive ViT features!")
            print("\n".join(lines), flush=True)
            return

        missing = [idx for idx in tap_layers if idx not in hook_feats]
        if missing:
            lines.append(f"  !! CRITICAL: ViT hooks MISSING for layers {missing} — hooks didn't fire!")
            lines.append(f"  Captured layers: {sorted(hook_feats.keys())}")
        else:
            lines.append(f"  All {len(tap_layers)} ViT hooks fired: layers {list(tap_layers)}")

        for idx in tap_layers:
            if idx in hook_feats:
                feat = hook_feats[idx]
                shape = tuple(feat.shape)
                lines.append(f"    layer {idx}: shape={shape}")
                if feat.abs().max().item() < 1e-8:
                    lines.append(f"    !! WARNING: layer {idx} features are all zeros!")
                if torch.isnan(feat).any():
                    lines.append(f"    !! CRITICAL: layer {idx} features contain NaN!")

        print("\n".join(lines), flush=True)

    # ------------------------------------------------------------------
    # Reporting schedule
    # ------------------------------------------------------------------
    def should_report(self, step, is_training=True):
        """Whether to produce diagnostics at this step."""
        if is_training:
            return step > 0 and step % self.train_interval == 0
        else:
            return self._eval_episode % self.eval_interval == 0

    def increment_eval_episode(self):
        """Call at the start of each eval episode."""
        self._eval_episode += 1

    # ------------------------------------------------------------------
    # Collect tensors
    # ------------------------------------------------------------------
    def collect(self, step, depth_input, eagle_features_before, eagle_features_after,
                attn_weights=None, fastguide_attns=None, depth_tokens=None,
                rgb_pixels=None, is_training=True,
                grid_h=None, grid_w=None, num_views=1, n_image_tokens=None,
                vit_projected=None, depth_stages=None):
        """Collect tensors for diagnostics. All are detached to CPU float32.

        Args:
            step: Current training step or eval step.
            depth_input: (B, 1, H, W) normalized depth.
            eagle_features_before: (B, seq, D) eagle features before CHNet.
            eagle_features_after: (B, seq, D) eagle features after CHNet.
            attn_weights: Optional (B, heads, seq_q, seq_kv) cross-attention weights.
            fastguide_attns: Optional list of 4 (B, 1, H, W) spatial attention maps.
            depth_tokens: Optional (B, N, D) depth tokens fed to cross-attention.
            rgb_pixels: Optional (B, 3, H, W) RGB input pixels.
            is_training: Whether we are in training mode.
            grid_h: ViT patch grid height.
            grid_w: ViT patch grid width.
            num_views: Number of camera views.
            n_image_tokens: Number of image tokens in eagle sequence.
            vit_projected: Optional list of 4 (B, C, H, W) projected ViT feature maps.
            depth_stages: Optional list of 4 (B, C, H, W) depth CNN stage features
                          before FastGuide.
        """
        self._collected = {
            "step": step,
            "is_training": is_training,
            "depth_input": _safe_detach(depth_input),
            "eagle_before": _safe_detach(eagle_features_before),
            "eagle_after": _safe_detach(eagle_features_after),
            "attn_weights": _safe_detach(attn_weights),
            "fastguide_attns": [_safe_detach(a) for a in fastguide_attns] if fastguide_attns else None,
            "depth_tokens": _safe_detach(depth_tokens),
            "rgb_pixels": _safe_detach(rgb_pixels),
            "grid_h": grid_h,
            "grid_w": grid_w,
            "num_views": num_views,
            "n_image_tokens": n_image_tokens,
            "vit_projected": [_safe_detach(v) for v in vit_projected] if vit_projected else None,
            "depth_stages": [_safe_detach(d) for d in depth_stages] if depth_stages else None,
        }

    # ------------------------------------------------------------------
    # Report: health checks + visualizations
    # ------------------------------------------------------------------
    def report(self, step, is_training=True):
        """Run health checks and produce visualizations."""
        if not self._collected:
            return

        # Determine output directory
        if is_training:
            step_dir = self.output_dir / f"step_{step}"
        else:
            step_dir = self.output_dir / f"eval_episode_{self._eval_episode}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Health checks (printed + saved)
        health_lines = self._health_checks(step)
        health_path = step_dir / "health_check.txt"
        health_text = "\n".join(health_lines)
        health_path.write_text(health_text + "\n")
        print(health_text, flush=True)

        # Visualizations
        index_entries = []

        viz_methods = [
            ("cross_attn_map", self._viz_cross_attn_map),
            ("fastguide_stages", self._viz_fastguide_stages),
            ("embedding_similarity", self._viz_embedding_similarity),
            ("input_overlay", self._viz_input_overlay),
            ("depth_contribution_map", self._viz_depth_contribution_map),
            ("proximity_direction", self._viz_proximity_direction),
            ("token_space_pca", self._viz_token_space_pca),
            ("nearest_neighbors", self._viz_nearest_neighbors),
            ("eagle_token_heatmap", self._viz_eagle_token_heatmap),
            ("depth_token_heatmap", self._viz_depth_token_heatmap),
            ("fastguide_pyramid", self._viz_fastguide_pyramid),
            ("reshape_verification", self._viz_reshape_verification),
        ]

        for name, method in viz_methods:
            try:
                doc = method(step_dir)
                if doc is not None:
                    index_entries.append(doc)
                else:
                    print(f"[CHNet Diag] Visualization '{name}' returned None (missing data)", flush=True)
            except Exception as e:
                msg = f"[CHNet Diag] Visualization '{name}' failed: {e}"
                print(msg, flush=True)
                index_entries.append({
                    "file": f"{name}.png",
                    "what": name,
                    "error": str(e),
                })

        # Write index.txt
        self._write_index(step_dir, index_entries, step)

        # Clear collected data
        self._collected = {}

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------
    def _health_checks(self, step):
        """Run depth input stats, weight norms, weight drift, change ratio."""
        lines = []
        lines.append(f"[CHNet Health @ step {step}]")

        # 1. Depth input stats + integrity
        depth = self._collected.get("depth_input")
        eagle_before = self._collected.get("eagle_before")
        eagle_after = self._collected.get("eagle_after")

        if depth is not None:
            lines.append(f"  depth_input shape : {tuple(depth.shape)}")
            lines.append(f"  depth_input min   : {depth.min().item():.4f}  (expected: >= -1.0)")
            lines.append(f"  depth_input max   : {depth.max().item():.4f}  (expected: <= 10.0)")
            lines.append(f"  depth_input mean  : {depth.mean().item():.4f}  (expected: 0.5-3.0)")
            lines.append(f"  depth_input std   : {depth.std().item():.4f}  (expected: 0.1-2.0)")
            zero_frac = (depth.abs() < 1e-6).float().mean().item()
            lines.append(f"  depth_input zero% : {zero_frac * 100:.1f}%  (expected: <30%)")

            # NaN/Inf check
            if torch.isnan(depth).any():
                lines.append("  !! CRITICAL: depth_input contains NaN — data pipeline is broken!")
            if torch.isinf(depth).any():
                lines.append("  !! CRITICAL: depth_input contains Inf — normalization is broken!")

            # Constant depth (broken sensor / bad data)
            depth_std = depth.std().item()
            if depth_std < 1e-6:
                lines.append("  !! WARNING: depth_input is constant (std < 1e-6) — no geometry signal!")
        else:
            lines.append("  depth_input       : NOT PROVIDED")

        # NaN/Inf in eagle features
        if eagle_before is not None and torch.isnan(eagle_before).any():
            lines.append("  !! CRITICAL: eagle_features BEFORE CHNet contain NaN!")
        if eagle_after is not None:
            if torch.isnan(eagle_after).any():
                lines.append("  !! CRITICAL: eagle_features AFTER CHNet contain NaN — CHNet is producing NaN!")
            if torch.isinf(eagle_after).any():
                lines.append("  !! CRITICAL: eagle_features AFTER CHNet contain Inf — CHNet is exploding!")

        # 2. Cross-attention entropy (is depth being used or ignored?)
        attn = self._collected.get("attn_weights")
        if attn is not None:
            # attn: (B, heads, seq_q, seq_kv) — compute entropy over depth tokens (last dim)
            attn_b0 = attn[0]  # (heads, seq_q, seq_kv)
            # Clamp for numerical safety
            attn_probs = attn_b0.clamp(min=1e-8)
            entropy = -(attn_probs * attn_probs.log()).sum(dim=-1).mean().item()
            max_entropy = np.log(attn_b0.shape[-1])
            entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0
            n_image_tokens = self._collected.get("n_image_tokens")
            num_views_h = self._collected.get("num_views", 1)
            lines.append(
                f"  attn_entropy      : {entropy:.3f} / {max_entropy:.3f} "
                f"(ratio={entropy_ratio:.3f}, Q={attn_b0.shape[1]} img tokens, "
                f"K={attn_b0.shape[2]} depth tokens = {num_views_h}x7x7)"
            )
            if entropy_ratio > 0.95:
                lines.append("  !! WARNING: attention is near-uniform — depth tokens not discriminated!")
            elif entropy_ratio < 0.1:
                lines.append("  !! WARNING: attention is collapsed to few tokens — possible degenerate fusion!")

        # 3. Weight norms
        backbone = self.model
        chnet = backbone.chnet
        if chnet is not None:
            lines.append("  --- CHNet Weight Norms ---")
            for sub_name in ["encoder", "fusion"]:
                sub = getattr(chnet, sub_name, None)
                if sub is not None:
                    norm_val = sum(p.detach().float().norm().item() ** 2 for p in sub.parameters()) ** 0.5
                    lines.append(f"    {sub_name:14s} L2 norm: {norm_val:.4f}  (expected: 1.0-50.0)")

            # 3. Weight drift from initial
            if self._collected.get("is_training", True):
                lines.append("  --- Weight Drift (vs init) ---")
                for name, param in chnet.named_parameters():
                    if name in self._initial_weights:
                        current_norm = param.detach().cpu().float().norm().item()
                        init_norm = self._initial_weights[name]
                        if init_norm > 1e-8:
                            drift_pct = abs(current_norm - init_norm) / init_norm * 100
                        else:
                            drift_pct = 0.0 if abs(current_norm) < 1e-8 else float("inf")
                        # Print top-level summary + out_proj (critical for zero-init verification)
                        if name.count(".") <= 2 or "out_proj" in name:
                            lines.append(
                                f"    {name:40s}: {drift_pct:6.2f}% drift  "
                                f"(init={init_norm:.4f}, now={current_norm:.4f})  "
                                f"(expected: <5% early, <20% late)"
                            )
            else:
                lines.append("  --- Weight Drift ---")
                lines.append("  (skipped during eval — init snapshot is post-checkpoint)")

        # 4. Change ratio + trend
        if eagle_before is not None and eagle_after is not None:
            diff_norm = (eagle_after - eagle_before).norm().item()
            before_norm = eagle_before.norm().item()
            if before_norm > 1e-8:
                ratio = diff_norm / before_norm
            else:
                ratio = 0.0
            lines.append(f"  change_ratio      : {ratio:.6f}  (expected: 0.001-0.05)")
            lines.append(f"  diff_norm         : {diff_norm:.4f}")
            lines.append(f"  before_norm       : {before_norm:.4f}")
            lines.append(f"  after_norm        : {eagle_after.norm().item():.4f}")

            # Trend detection
            if self._prev_change_ratio is not None and self._prev_change_ratio > 1e-8:
                trend = ratio / self._prev_change_ratio
                if trend < 0.1:
                    lines.append(f"  !! WARNING: change_ratio dropped {1/trend:.0f}x — depth branch may be dying!")
                elif trend > 10:
                    lines.append(f"  !! WARNING: change_ratio jumped {trend:.0f}x — depth branch may be exploding!")
            self._prev_change_ratio = ratio

        # 5. BatchNorm running stats health
        backbone = self.model
        chnet = backbone.chnet
        if chnet is not None:
            bn_eval_count = 0
            bn_stuck_count = 0
            bn_total = 0
            for name, mod in chnet.named_modules():
                if isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    bn_total += 1
                    if not mod.training:
                        bn_eval_count += 1
                    elif mod.running_mean is not None and step > 100:
                        if mod.running_mean.abs().max().item() < 1e-7:
                            bn_stuck_count += 1
            if bn_eval_count > 0:
                lines.append(
                    f"  !! WARNING: {bn_eval_count}/{bn_total} BN layers in eval mode "
                    f"— running stats won't update!"
                )
            if bn_stuck_count > 0:
                lines.append(
                    f"  !! NOTE: {bn_stuck_count}/{bn_total} BN layers have running_mean~0 "
                    f"(may be normal for this architecture)"
                )
            if bn_eval_count == 0 and bn_stuck_count == 0:
                lines.append(f"  BN layers: {bn_total} total, all in training mode, stats updating")

        return lines

    # ------------------------------------------------------------------
    # Gradient logging
    # ------------------------------------------------------------------
    def log_gradients(self, step, model):
        """Log CHNet gradient norms per sub-module after backward pass.

        Args:
            step: Current training step.
            model: The GR00TN15 model.
        """
        # Log every step for the first 10 steps (startup debugging),
        # then only every train_interval steps to reduce log spam.
        if step > 10 and step % self.train_interval != 0:
            return

        backbone = model.backbone
        chnet = backbone.chnet
        if chnet is None:
            return

        # --- Step 1: verify zero-init actually took effect ---
        if step == 1:
            out_proj = chnet.fusion.cross_attn.out_proj
            w_norm = out_proj.weight.data.detach().float().norm().item()
            b_norm = out_proj.bias.data.detach().float().norm().item()
            print(
                f"[CHNet Zero-Init Check @ step 1 BEFORE optimizer step]\n"
                f"  out_proj.weight norm: {w_norm:.8f}  (should be 0.0)\n"
                f"  out_proj.bias   norm: {b_norm:.8f}  (should be 0.0)\n"
                f"  {'OK — zero-init confirmed' if w_norm < 1e-7 and b_norm < 1e-7 else '!! FAILED — weights are NOT zero, HF _no_init_weights likely overwrote them'}",
                flush=True,
            )

        lines = [f"[CHNet Gradients @ step {step}]"]

        # Per named sub-module (top-level summary)
        module_grads = defaultdict(list)
        # Also collect per-param details for detailed logging
        param_grad_details = {}
        for name, param in chnet.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.detach().float().norm().item()
                # Group by first component (encoder, fusion)
                top_level = name.split(".")[0]
                module_grads[top_level].append(grad_norm)
                param_grad_details[name] = {
                    "grad_norm": grad_norm,
                    "weight_norm": param.data.detach().float().norm().item(),
                    "grad_to_weight": grad_norm / max(param.data.detach().float().norm().item(), 1e-10),
                }

        for mod_name, norms in sorted(module_grads.items()):
            if norms:
                mean_norm = sum(norms) / len(norms)
                max_norm = max(norms)
                lines.append(
                    f"  {mod_name:14s}: mean={mean_norm:.6f}, max={max_norm:.6f}, "
                    f"n_params={len(norms)}  (expected mean: 1e-5 to 1e-1)"
                )

        # --- Detailed per-param breakdown for fusion (every train_interval or first 10 steps) ---
        lines.append("  --- Fusion Per-Param Detail ---")
        for name in sorted(param_grad_details.keys()):
            if name.startswith("fusion."):
                d = param_grad_details[name]
                lines.append(
                    f"    {name:45s}: grad={d['grad_norm']:.6f}, "
                    f"weight={d['weight_norm']:.6f}, "
                    f"grad/weight={d['grad_to_weight']:.6f}"
                )

        # --- Encoder sub-module breakdown (stages, guides, projs) ---
        encoder_groups = defaultdict(list)
        for name, info in param_grad_details.items():
            if name.startswith("encoder."):
                # Group by second component (stem, stage1, guide1, proj1, etc.)
                parts = name.split(".")
                group = parts[1] if len(parts) > 1 else "other"
                encoder_groups[group].append(info["grad_norm"])

        if encoder_groups:
            lines.append("  --- Encoder Sub-Module Gradients ---")
            for group_name in sorted(encoder_groups.keys()):
                norms = encoder_groups[group_name]
                lines.append(
                    f"    {group_name:14s}: mean={sum(norms)/len(norms):.6f}, "
                    f"max={max(norms):.6f}, n={len(norms)}"
                )

        # Check for zero/missing gradients (common bug)
        total_params = sum(1 for _, p in chnet.named_parameters() if p.requires_grad)
        has_grad = sum(1 for _, p in chnet.named_parameters() if p.grad is not None)
        zero_grad = sum(
            1 for _, p in chnet.named_parameters()
            if p.grad is not None and p.grad.abs().max().item() < 1e-12
        )
        lines.append(f"  params_with_grad  : {has_grad}/{total_params}  (expected: all)")

        if has_grad == 0:
            lines.append("  !! CRITICAL: No gradients flowing through CHNet — backward graph is broken!")
        elif zero_grad > 0:
            # List the zero-grad params by name
            zero_names = [
                n for n, p in chnet.named_parameters()
                if p.grad is not None and p.grad.abs().max().item() < 1e-12
            ]
            lines.append(
                f"  !! WARNING: {zero_grad} params have exactly-zero gradients — "
                f"possible dead branches in CHNet"
            )
            for zn in zero_names[:10]:
                lines.append(f"    zero-grad: {zn}")

        # NaN in gradients
        nan_grad = sum(
            1 for _, p in chnet.named_parameters()
            if p.grad is not None and torch.isnan(p.grad).any()
        )
        if nan_grad > 0:
            lines.append(f"  !! CRITICAL: {nan_grad} params have NaN gradients — training will diverge!")

        text = "\n".join(lines)
        print(text, flush=True)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def _save_fig(self, fig, path):
        """Save figure and close."""
        fig.savefig(str(path), dpi=120, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 1. Cross-attention map
    # ------------------------------------------------------------------
    def _viz_cross_attn_map(self, step_dir):
        """Heatmap of cross-attention weights (Q=eagle, K/V=depth)."""
        attn = self._collected.get("attn_weights")
        if attn is None:
            return None

        # attn shape: (B, heads, seq_q, seq_kv) — take first batch
        attn_b0 = attn[0]  # (heads, seq_q, seq_kv)
        n_heads = attn_b0.shape[0]
        num_views = self._collected.get("num_views", 1)
        n_image_tokens = self._collected.get("n_image_tokens")
        tokens_per_view = 49

        # --- Figure 1: flat heatmap per head ---
        fig, axes = plt.subplots(2, (n_heads + 1) // 2, figsize=(3 * ((n_heads + 1) // 2), 6))
        axes = np.array(axes).flatten()
        fig.suptitle("Cross-Attention Weights (Q=eagle, K/V=depth)", fontsize=10)

        for h in range(n_heads):
            ax = axes[h]
            im = ax.imshow(attn_b0[h].numpy(), aspect="auto", cmap="hot")
            # Per-head entropy
            head_attn = attn_b0[h].numpy()
            head_entropy = -(head_attn * np.log(head_attn + 1e-10)).sum(axis=-1).mean()
            max_ent = np.log(head_attn.shape[-1])
            ax.set_title(f"Head {h} (ent={head_entropy / max_ent:.2f})", fontsize=7)
            ax.set_xlabel(f"depth token ({num_views}x7x7)", fontsize=6)
            if n_image_tokens:
                ax.set_ylabel(f"image token ({n_image_tokens})", fontsize=6)
            else:
                ax.set_ylabel("eagle token", fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046)

        for i in range(n_heads, len(axes)):
            axes[i].set_visible(False)

        self._save_fig(fig, step_dir / "cross_attn_map.png")

        # --- Figure 2: 2D spatial attention per head (7x7) ---
        # Average over Q dimension to get attention per depth token, reshape to 7x7
        fig2, axes2 = plt.subplots(num_views, n_heads, figsize=(2.5 * n_heads, 3 * num_views))
        if num_views == 1:
            axes2 = axes2[np.newaxis, :]
        if n_heads == 1:
            axes2 = axes2[:, np.newaxis]
        fig2.suptitle("Spatial Attention per Head (avg over Q, reshaped to 7x7)", fontsize=10)

        for v in range(num_views):
            start = v * tokens_per_view
            end = start + tokens_per_view
            if end > attn_b0.shape[2]:
                break
            for h in range(n_heads):
                ax = axes2[v, h]
                # Mean over Q, take this view's depth tokens
                spatial = attn_b0[h].mean(dim=0).numpy()[start:end]
                if len(spatial) == tokens_per_view:
                    im = ax.imshow(spatial.reshape(7, 7), cmap="hot")
                    fig2.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(f"H{h} V{v}", fontsize=7)

        self._save_fig(fig2, step_dir / "cross_attn_spatial.png")

        return {
            "file": "cross_attn_map.png",
            "what": "Heatmap of cross-attention weights (Q=eagle, K/V=depth). "
                    "Also see cross_attn_spatial.png for 2D 7x7 spatial view per head.",
            "how_to_read": "Bright spots show which depth tokens each eagle token attends to. "
                           "Each subplot is one attention head. Per-head entropy ratio shown in title.",
            "expect": "Structured patterns; heads specializing in different depth regions.",
            "concern_if": "Uniform/flat attention across all heads (entropy ratio near 1.0).",
        }

    # ------------------------------------------------------------------
    # 2. FastGuide stages
    # ------------------------------------------------------------------
    def _viz_fastguide_stages(self, step_dir):
        """Spatial attention from FastGuide at 4 CNN stages."""
        fg_attns = self._collected.get("fastguide_attns")
        if fg_attns is None:
            return None

        fig, axes = plt.subplots(1, len(fg_attns), figsize=(4 * len(fg_attns), 4))
        if len(fg_attns) == 1:
            axes = [axes]
        fig.suptitle("FastGuide Spatial Attention (4 CNN stages)", fontsize=10)

        for i, attn_map in enumerate(fg_attns):
            ax = axes[i]
            # attn_map: (B, 1, H, W) — take first batch
            am = attn_map[0, 0].numpy()
            im = ax.imshow(am, cmap="inferno")
            ax.set_title(f"Stage {i + 1} ({am.shape[0]}x{am.shape[1]})", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)

        path = step_dir / "fastguide_stages.png"
        self._save_fig(fig, path)

        return {
            "file": "fastguide_stages.png",
            "what": "Spatial attention maps from FastGuide at 4 CNN encoder stages.",
            "how_to_read": "Bright regions indicate where RGB features modulate depth features most. "
                           "Resolution decreases from stage 1 (56x56) to stage 4 (7x7).",
            "expect": "Attention concentrated on task-relevant areas (objects, surfaces).",
            "concern_if": "All-uniform or all-zero maps (RGB guidance not functioning).",
        }

    # ------------------------------------------------------------------
    # 3. Embedding similarity
    # ------------------------------------------------------------------
    def _viz_embedding_similarity(self, step_dir):
        """Cosine similarity heatmap: depth tokens vs eagle tokens (before/after fusion)."""
        depth_tok = self._collected.get("depth_tokens")
        eagle_before = self._collected.get("eagle_before")
        eagle_after = self._collected.get("eagle_after")
        if depth_tok is None or eagle_before is None or eagle_after is None:
            return None

        # First batch, subsample tokens for readability
        n_image_tokens = self._collected.get("n_image_tokens")
        if n_image_tokens:
            max_tokens = min(64, n_image_tokens)
        else:
            max_tokens = 64
        d = depth_tok[0][:max_tokens]   # (N_d, D)
        eb = eagle_before[0][:max_tokens]  # (N_e, D)
        ea = eagle_after[0][:max_tokens]

        token_label = f"image token (first {max_tokens})" if n_image_tokens else "eagle token"

        # Normalize
        d_norm = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        eb_norm = eb / (eb.norm(dim=-1, keepdim=True) + 1e-8)
        ea_norm = ea / (ea.norm(dim=-1, keepdim=True) + 1e-8)

        sim_before = (eb_norm @ d_norm.T).numpy()  # (N_e, N_d)
        sim_after = (ea_norm @ d_norm.T).numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Cosine Similarity: Eagle Tokens vs Depth Tokens", fontsize=10)

        im1 = ax1.imshow(sim_before, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax1.set_title("BEFORE fusion", fontsize=9)
        ax1.set_xlabel("depth token")
        ax1.set_ylabel(token_label)
        fig.colorbar(im1, ax=ax1, fraction=0.046)

        im2 = ax2.imshow(sim_after, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax2.set_title("AFTER fusion", fontsize=9)
        ax2.set_xlabel("depth token")
        ax2.set_ylabel(token_label)
        fig.colorbar(im2, ax=ax2, fraction=0.046)

        path = step_dir / "embedding_similarity.png"
        self._save_fig(fig, path)

        return {
            "file": "embedding_similarity.png",
            "what": "Cosine similarity between depth tokens and eagle tokens, before and after CHNet fusion.",
            "how_to_read": "Red = high similarity, blue = anti-correlated. "
                           "Compare left (before) vs right (after) to see how fusion changes alignment.",
            "expect": "After fusion: stronger diagonal/structured similarity patterns vs before.",
            "concern_if": "No difference between before/after (CHNet has no effect on token alignment).",
        }

    # ------------------------------------------------------------------
    # 4. Input overlay
    # ------------------------------------------------------------------
    def _viz_input_overlay(self, step_dir):
        """RGB | depth (viridis) | attention overlaid on depth."""
        depth = self._collected.get("depth_input")
        rgb = self._collected.get("rgb_pixels")
        attn = self._collected.get("attn_weights")

        if depth is None:
            return None

        # First batch
        depth_2d = depth[0, 0].numpy()  # (H, W)

        n_cols = 1
        has_rgb = rgb is not None
        has_attn = attn is not None

        if has_rgb:
            n_cols += 1
        if has_attn:
            n_cols += 1

        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]
        fig.suptitle("Input Overlay", fontsize=10)

        col = 0

        # RGB
        if has_rgb:
            rgb_np = rgb[0].permute(1, 2, 0).numpy()  # (H, W, 3)
            rgb_np = np.clip(rgb_np, 0, 1) if rgb_np.max() <= 1.0 else np.clip(rgb_np / 255.0, 0, 1)
            axes[col].imshow(rgb_np)
            axes[col].set_title("RGB", fontsize=9)
            col += 1

        # Depth viridis
        axes[col].imshow(depth_2d, cmap="viridis")
        axes[col].set_title(
            f"Depth (viridis)\nrange: [{depth_2d.min():.2f}, {depth_2d.max():.2f}]",
            fontsize=9,
        )
        col += 1

        # Attention overlaid on depth
        if has_attn:
            num_views = self._collected.get("num_views", 1)
            attn_avg = attn[0].mean(dim=0).mean(dim=0).numpy()  # (num_views * 49,)
            tokens_per_view = 49
            depth_h, depth_w = 7, 7
            view_attn = attn_avg[:tokens_per_view]  # first view
            attn_map = view_attn.reshape(depth_h, depth_w)

            # Compute entropy to detect uniform attention
            n_tokens = len(attn_avg)
            attn_probs = attn_avg / (attn_avg.sum() + 1e-8)
            entropy = -(attn_probs * np.log(attn_probs + 1e-10)).sum()
            max_entropy = np.log(n_tokens)
            entropy_ratio = entropy / max_entropy if max_entropy > 0 else 1.0
            is_uniform = entropy_ratio > 0.95

            if _HAS_PIL:
                attn_img = PILImage.fromarray(
                    (attn_map / (attn_map.max() + 1e-8) * 255).astype(np.uint8)
                )
                attn_img = attn_img.resize((depth_2d.shape[1], depth_2d.shape[0]), PILImage.BILINEAR)
                attn_resized = np.array(attn_img).astype(np.float32) / 255.0
            else:
                attn_resized = attn_map

            axes[col].imshow(depth_2d, cmap="viridis", alpha=0.6)
            axes[col].imshow(attn_resized, cmap="hot", alpha=0.4)

            # Annotate with stats
            title = "Attention on Depth (view 0)"
            if is_uniform:
                title += "\n!! NEAR-UNIFORM !!"
            title += f"\nrange: [{attn_map.min():.4f}, {attn_map.max():.4f}]"
            title += f"  entropy: {entropy_ratio:.3f}"
            axes[col].set_title(title, fontsize=8)

        path = step_dir / "input_overlay.png"
        self._save_fig(fig, path)

        return {
            "file": "input_overlay.png",
            "what": "Side-by-side: RGB | depth (viridis colormap) | attention overlaid on depth.",
            "how_to_read": "Leftmost: raw RGB image. Middle: depth as viridis heatmap (purple=near, yellow=far). "
                           "Right: cross-attention summed over heads, overlaid on depth.",
            "expect": "Attention focused on depth regions with objects or task-relevant geometry.",
            "concern_if": "Depth map looks constant (no geometry) or attention is random.",
        }

    # ------------------------------------------------------------------
    # 5. Depth contribution map
    # ------------------------------------------------------------------
    def _viz_depth_contribution_map(self, step_dir):
        """Per-token magnitude of (eagle_after - eagle_before)."""
        eagle_before = self._collected.get("eagle_before")
        eagle_after = self._collected.get("eagle_after")
        if eagle_before is None or eagle_after is None:
            return None

        # First batch
        diff = (eagle_after[0] - eagle_before[0])  # (seq, D)
        magnitudes = diff.norm(dim=-1).numpy()  # (seq,)

        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.bar(range(len(magnitudes)), magnitudes, width=1.0, color="steelblue")
        ax.set_xlabel("Eagle token index")
        ax.set_ylabel("||delta|| (L2 norm)")
        ax.set_title(
            f"Depth Contribution per Eagle Token\n"
            f"mean={magnitudes.mean():.4f}, max={magnitudes.max():.4f}  "
            f"(expected mean: 0.01-1.0)",
            fontsize=9,
        )

        # Spatial view using proper grid dimensions
        grid_h = self._collected.get("grid_h")
        grid_w = self._collected.get("grid_w")
        num_views = self._collected.get("num_views", 1)
        n_image_tokens = self._collected.get("n_image_tokens")

        if grid_h and grid_w and n_image_tokens:
            img_magnitudes = magnitudes[:n_image_tokens]
            tokens_per_view = grid_h * grid_w
            for v in range(num_views):
                start = v * tokens_per_view
                end = start + tokens_per_view
                if end <= len(img_magnitudes):
                    fig_v, ax_v = plt.subplots(1, 1, figsize=(5, 5))
                    im = ax_v.imshow(img_magnitudes[start:end].reshape(grid_h, grid_w), cmap="magma")
                    ax_v.set_title(f"Depth Contribution (view {v}, {grid_h}x{grid_w})", fontsize=9)
                    fig_v.colorbar(im, ax=ax_v)
                    self._save_fig(fig_v, step_dir / f"depth_contribution_spatial_view{v}.png")

        path = step_dir / "depth_contribution_map.png"
        self._save_fig(fig, path)

        return {
            "file": "depth_contribution_map.png",
            "what": "Per-token L2 magnitude of (eagle_after - eagle_before) from CHNet fusion.",
            "how_to_read": "Tall bars = tokens most modified by depth info. "
                           "If tokens form a spatial grid, also see depth_contribution_spatial.png.",
            "expect": "Non-zero contributions across many tokens; some spatial structure.",
            "concern_if": "All zeros (CHNet not modifying features) or extreme outliers only.",
        }

    # ------------------------------------------------------------------
    # 6. Proximity direction
    # ------------------------------------------------------------------
    def _viz_proximity_direction(self, step_dir):
        """Project eagle tokens onto near-far depth direction."""
        eagle_after = self._collected.get("eagle_after")
        depth_tok = self._collected.get("depth_tokens")
        depth_input = self._collected.get("depth_input")

        if eagle_after is None or depth_tok is None or depth_input is None:
            return None

        # Compute "near" and "far" direction from depth tokens
        # Use mean of depth_input per-patch to rank depth tokens by depth value
        d_flat = depth_input[0, 0]  # (H, W)
        d_mean_val = d_flat.mean().item()

        # Sort depth tokens by spatial depth: first half = near, second half = far
        n_depth = depth_tok.shape[1]
        half = n_depth // 2
        # Approximate: first tokens = top-left (often near), last = bottom-right
        near_centroid = depth_tok[0, :half].mean(dim=0)  # (D,)
        far_centroid = depth_tok[0, half:].mean(dim=0)

        direction = far_centroid - near_centroid
        dir_norm = direction.norm()
        if dir_norm < 1e-8:
            return None
        direction = direction / dir_norm

        # Project eagle tokens onto this direction
        projections = (eagle_after[0] @ direction).numpy()  # (seq,)

        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        colors = plt.cm.coolwarm(Normalize()(projections))
        ax.bar(range(len(projections)), projections, width=1.0, color=colors)
        ax.set_xlabel("Eagle token index")
        ax.set_ylabel("Projection onto near-far axis")
        ax.set_title(
            f"Proximity Direction Projection\n"
            f"(near=negative, far=positive; depth mean={d_mean_val:.2f})",
            fontsize=9,
        )
        ax.axhline(0, color="black", linewidth=0.5)

        path = step_dir / "proximity_direction.png"
        self._save_fig(fig, path)

        return {
            "file": "proximity_direction.png",
            "what": "Projection of eagle tokens onto the near-far depth direction.",
            "how_to_read": "Blue/negative = near-leaning tokens, red/positive = far-leaning. "
                           "Derived from centroids of near vs far depth tokens.",
            "expect": "Gradient from near to far across spatial tokens; some clustering.",
            "concern_if": "All projections near zero (depth direction not encoded in eagle features).",
        }

    # ------------------------------------------------------------------
    # 7. Token space PCA
    # ------------------------------------------------------------------
    def _viz_token_space_pca(self, step_dir):
        """PCA of depth + eagle tokens, colored by depth value."""
        if not _HAS_SKLEARN:
            return {
                "file": "token_space_pca.png",
                "what": "PCA of depth+eagle tokens (SKIPPED: sklearn not installed).",
                "how_to_read": "N/A",
                "expect": "N/A",
                "concern_if": "N/A",
            }

        depth_tok = self._collected.get("depth_tokens")
        eagle_after = self._collected.get("eagle_after")
        depth_input = self._collected.get("depth_input")

        if depth_tok is None or eagle_after is None:
            return None

        # First batch
        d_tokens = depth_tok[0].numpy()   # (N_d, D)
        e_tokens = eagle_after[0].numpy()  # (N_e, D)

        # Subsample for speed
        max_tok = 200
        if d_tokens.shape[0] > max_tok:
            idx = np.linspace(0, d_tokens.shape[0] - 1, max_tok).astype(int)
            d_tokens = d_tokens[idx]
        if e_tokens.shape[0] > max_tok:
            idx = np.linspace(0, e_tokens.shape[0] - 1, max_tok).astype(int)
            e_tokens = e_tokens[idx]

        all_tokens = np.concatenate([d_tokens, e_tokens], axis=0)
        labels = np.array([0] * len(d_tokens) + [1] * len(e_tokens))

        # Depth values for coloring depth tokens
        if depth_input is not None:
            d_flat = depth_input[0, 0].numpy().flatten()
            n_d = len(d_tokens)
            # Subsample depth values to match subsampled tokens
            if len(d_flat) >= n_d:
                step_size = max(1, len(d_flat) // n_d)
                depth_colors = d_flat[::step_size][:n_d]
            else:
                depth_colors = np.zeros(n_d)
        else:
            depth_colors = np.zeros(len(d_tokens))

        pca = SklearnPCA(n_components=2)
        projected = pca.fit_transform(all_tokens)

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        # Depth tokens colored by depth value
        n_d = len(d_tokens)
        scatter_d = ax.scatter(
            projected[:n_d, 0], projected[:n_d, 1],
            c=depth_colors, cmap="viridis", s=15, alpha=0.7, label="depth tokens",
            edgecolors="none",
        )
        fig.colorbar(scatter_d, ax=ax, label="Depth value", fraction=0.046)

        # Eagle tokens in red
        ax.scatter(
            projected[n_d:, 0], projected[n_d:, 1],
            c="red", s=10, alpha=0.4, marker="x", label="eagle tokens",
        )

        ax.set_title(
            f"PCA: Depth + Eagle Tokens\n"
            f"(explained var: {pca.explained_variance_ratio_.sum():.2f})",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        path = step_dir / "token_space_pca.png"
        self._save_fig(fig, path)

        return {
            "file": "token_space_pca.png",
            "what": "PCA projection of depth tokens (colored by depth value) and eagle tokens (red x).",
            "how_to_read": "Depth tokens colored purple-to-yellow by depth value. "
                           "Eagle tokens as red crosses. Overlap means shared representation space.",
            "expect": "Depth tokens form a gradient from near to far; eagle tokens mix in.",
            "concern_if": "Depth and eagle tokens completely separated (fusion not bridging modalities).",
        }

    # ------------------------------------------------------------------
    # 8. Nearest neighbors
    # ------------------------------------------------------------------
    def _viz_nearest_neighbors(self, step_dir):
        """Arrows from depth patches to nearest eagle patches in 2D PCA space."""
        if not _HAS_SKLEARN:
            return {
                "file": "nearest_neighbors.png",
                "what": "Nearest-neighbor arrows depth->eagle (SKIPPED: sklearn not installed).",
                "how_to_read": "N/A",
                "expect": "N/A",
                "concern_if": "N/A",
            }

        depth_tok = self._collected.get("depth_tokens")
        eagle_after = self._collected.get("eagle_after")

        if depth_tok is None or eagle_after is None:
            return None

        # First batch, subsample
        max_tok = 100
        d_tokens = depth_tok[0].numpy()
        e_tokens = eagle_after[0].numpy()
        if d_tokens.shape[0] > max_tok:
            idx = np.linspace(0, d_tokens.shape[0] - 1, max_tok).astype(int)
            d_tokens = d_tokens[idx]
        if e_tokens.shape[0] > max_tok:
            idx = np.linspace(0, e_tokens.shape[0] - 1, max_tok).astype(int)
            e_tokens = e_tokens[idx]

        all_tokens = np.concatenate([d_tokens, e_tokens], axis=0)
        pca = SklearnPCA(n_components=2)
        projected = pca.fit_transform(all_tokens)

        n_d = len(d_tokens)
        d_proj = projected[:n_d]
        e_proj = projected[n_d:]

        # Find nearest eagle token for each depth token (cosine distance in original space)
        d_norm = d_tokens / (np.linalg.norm(d_tokens, axis=1, keepdims=True) + 1e-8)
        e_norm = e_tokens / (np.linalg.norm(e_tokens, axis=1, keepdims=True) + 1e-8)
        sim_matrix = d_norm @ e_norm.T  # (n_d, n_e)
        nearest_idx = sim_matrix.argmax(axis=1)

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        ax.scatter(d_proj[:, 0], d_proj[:, 1], c="blue", s=20, alpha=0.6, label="depth")
        ax.scatter(e_proj[:, 0], e_proj[:, 1], c="red", s=20, alpha=0.6, marker="x", label="eagle")

        # Draw arrows (subsample to avoid clutter)
        arrow_step = max(1, n_d // 30)
        for i in range(0, n_d, arrow_step):
            j = nearest_idx[i]
            ax.annotate(
                "",
                xy=(e_proj[j, 0], e_proj[j, 1]),
                xytext=(d_proj[i, 0], d_proj[i, 1]),
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4, lw=0.7),
            )

        ax.set_title("Nearest Neighbors: Depth -> Eagle (PCA space)", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        path = step_dir / "nearest_neighbors.png"
        self._save_fig(fig, path)

        return {
            "file": "nearest_neighbors.png",
            "what": "Arrows from each depth patch to its nearest eagle patch (cosine similarity), "
                    "visualized in PCA space.",
            "how_to_read": "Blue dots = depth tokens, red crosses = eagle tokens. "
                           "Gray arrows show which eagle token is most similar to each depth token.",
            "expect": "Arrows pointing to spatially/semantically similar eagle tokens; some clustering.",
            "concern_if": "All arrows converge to a single eagle token (degenerate fusion).",
        }

    # ------------------------------------------------------------------
    # 9. Eagle token heatmap over RGB
    # ------------------------------------------------------------------
    def _viz_eagle_token_heatmap(self, step_dir):
        """Per-view heatmap of eagle image token depth-contribution overlaid on RGB."""
        eagle_before = self._collected.get("eagle_before")
        eagle_after = self._collected.get("eagle_after")
        rgb_pixels = self._collected.get("rgb_pixels")
        grid_h = self._collected.get("grid_h")
        grid_w = self._collected.get("grid_w")
        num_views = self._collected.get("num_views", 1)
        n_image_tokens = self._collected.get("n_image_tokens")

        if eagle_before is None or eagle_after is None:
            return None
        if grid_h is None or grid_w is None or n_image_tokens is None:
            return None

        diff = eagle_after[0] - eagle_before[0]  # (seq, D)
        magnitudes = diff.norm(dim=-1).numpy()    # (seq,)

        tokens_per_view = grid_h * grid_w
        saved_files = []

        for v in range(num_views):
            start = v * tokens_per_view
            end = start + tokens_per_view
            if end > n_image_tokens:
                break

            heatmap = magnitudes[start:end].reshape(grid_h, grid_w)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"Eagle Token Depth Contribution Heatmap (view {v})", fontsize=10)

            # Left: RGB
            if rgb_pixels is not None and v < rgb_pixels.shape[0]:
                rgb_np = rgb_pixels[v].permute(1, 2, 0).numpy()
                rgb_np = np.clip(rgb_np, 0, 1) if rgb_np.max() <= 1.0 else np.clip(rgb_np / 255.0, 0, 1)
                axes[0].imshow(rgb_np)
                axes[0].set_title("RGB", fontsize=9)
            else:
                axes[0].text(0.5, 0.5, "RGB not available", ha="center", va="center",
                             transform=axes[0].transAxes)
                axes[0].set_title("RGB (N/A)", fontsize=9)

            # Right: heatmap overlaid on RGB
            if rgb_pixels is not None and v < rgb_pixels.shape[0]:
                rgb_np = rgb_pixels[v].permute(1, 2, 0).numpy()
                rgb_np = np.clip(rgb_np, 0, 1) if rgb_np.max() <= 1.0 else np.clip(rgb_np / 255.0, 0, 1)
                axes[1].imshow(rgb_np, alpha=0.6)

                if _HAS_PIL:
                    h_img = PILImage.fromarray(
                        (heatmap / (heatmap.max() + 1e-8) * 255).astype(np.uint8)
                    )
                    rgb_h, rgb_w = rgb_np.shape[:2]
                    h_img = h_img.resize((rgb_w, rgb_h), PILImage.BILINEAR)
                    heatmap_resized = np.array(h_img).astype(np.float32) / 255.0
                    im = axes[1].imshow(heatmap_resized, cmap="hot", alpha=0.4)
                else:
                    im = axes[1].imshow(heatmap, cmap="hot", alpha=0.4)
            else:
                im = axes[1].imshow(heatmap, cmap="hot")

            axes[1].set_title(f"Depth Contribution ({grid_h}x{grid_w})", fontsize=9)
            fig.colorbar(im, ax=axes[1], fraction=0.046)

            fname = f"eagle_token_heatmap_view{v}.png"
            self._save_fig(fig, step_dir / fname)
            saved_files.append(fname)

        if not saved_files:
            return None

        return {
            "file": saved_files[0],
            "what": "Per-view heatmap of eagle image token depth-contribution (L2 of delta before/after fusion) overlaid on RGB.",
            "how_to_read": "Bright regions = image tokens most modified by CHNet depth fusion. "
                           "Overlaid on RGB for spatial context. One file per view.",
            "expect": "Contributions concentrated on task-relevant objects or depth-salient regions.",
            "concern_if": "Uniform heatmap (CHNet not differentially modifying tokens) or all zeros.",
        }

    # ------------------------------------------------------------------
    # 10. Depth token heatmap over depth
    # ------------------------------------------------------------------
    def _viz_depth_token_heatmap(self, step_dir):
        """Per-view heatmap of attention received by each depth token overlaid on depth image."""
        attn = self._collected.get("attn_weights")
        depth_input = self._collected.get("depth_input")
        num_views = self._collected.get("num_views", 1)

        if attn is None or depth_input is None:
            return None

        # attn: (B, heads, n_image_q, num_views*49)
        # Sum over Q (image tokens) and average over heads to get attention received per depth token
        attn_received = attn[0].mean(dim=0).sum(dim=0).numpy()  # (num_views*49,)

        saved_files = []

        for v in range(num_views):
            start = v * 49
            end = start + 49
            if end > len(attn_received):
                break

            attn_map = attn_received[start:end].reshape(7, 7)

            # Get depth image for this view
            depth_idx = v if v < depth_input.shape[0] else 0
            depth_2d = depth_input[depth_idx, 0].numpy()

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"Depth Token Attention Heatmap (view {v})", fontsize=10)

            # Left: depth image
            axes[0].imshow(depth_2d, cmap="viridis")
            axes[0].set_title(
                f"Depth (viridis)\nrange: [{depth_2d.min():.2f}, {depth_2d.max():.2f}]",
                fontsize=9,
            )

            # Right: attention overlaid on depth
            axes[1].imshow(depth_2d, cmap="viridis", alpha=0.6)
            if _HAS_PIL:
                a_img = PILImage.fromarray(
                    (attn_map / (attn_map.max() + 1e-8) * 255).astype(np.uint8)
                )
                a_img = a_img.resize((depth_2d.shape[1], depth_2d.shape[0]), PILImage.BILINEAR)
                attn_resized = np.array(a_img).astype(np.float32) / 255.0
                im = axes[1].imshow(attn_resized, cmap="hot", alpha=0.4)
            else:
                im = axes[1].imshow(attn_map, cmap="hot", alpha=0.4)
            axes[1].set_title("Attention Received (7x7)", fontsize=9)
            fig.colorbar(im, ax=axes[1], fraction=0.046)

            fname = f"depth_token_heatmap_view{v}.png"
            self._save_fig(fig, step_dir / fname)
            saved_files.append(fname)

        if not saved_files:
            return None

        return {
            "file": saved_files[0],
            "what": "Per-view heatmap of attention received by each depth token (summed over image Q tokens), overlaid on depth image.",
            "how_to_read": "Bright 7x7 regions overlaid on depth = depth tokens that are most attended to by image tokens. "
                           "One file per view.",
            "expect": "Attention concentrated on depth patches with salient geometry (objects, edges).",
            "concern_if": "Uniform attention (depth tokens equally attended = depth not discriminated) or single hot spot.",
        }

    # ------------------------------------------------------------------
    # 11. FastGuide pyramid
    # ------------------------------------------------------------------
    def _viz_fastguide_pyramid(self, step_dir):
        """3-row x 4-col grid: ViT projected, depth CNN, and FastGuide attention per stage."""
        vit_projected = self._collected.get("vit_projected")
        depth_stages = self._collected.get("depth_stages")
        fg_attns = self._collected.get("fastguide_attns")

        if vit_projected is None and depth_stages is None and fg_attns is None:
            return None

        n_stages = 4
        stage_sizes = [56, 28, 14, 7]

        fig, axes = plt.subplots(3, n_stages, figsize=(4 * n_stages, 12))
        fig.suptitle("FastGuide Pyramid (Row 0: ViT projected, Row 1: Depth CNN, Row 2: FastGuide attn)", fontsize=10)

        row_labels = ["ViT projected", "Depth CNN", "FastGuide attn"]
        for row in range(3):
            axes[row, 0].set_ylabel(row_labels[row], fontsize=9)

        for stage_idx in range(n_stages):
            sz = stage_sizes[stage_idx]

            # Row 0: ViT projected features (channel-mean magnitude, first view)
            ax0 = axes[0, stage_idx]
            if vit_projected is not None and stage_idx < len(vit_projected):
                feat = vit_projected[stage_idx]  # (B*num_views, C, H, W)
                feat_view0 = feat[0]  # (C, H, W)
                mag = feat_view0.abs().mean(dim=0).numpy()  # (H, W)
                im0 = ax0.imshow(mag, cmap="viridis")
                ax0.set_title(f"Stage {stage_idx + 1} ({sz}x{sz})", fontsize=8)
                fig.colorbar(im0, ax=ax0, fraction=0.046)
            else:
                ax0.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax0.transAxes)
                ax0.set_title(f"Stage {stage_idx + 1} ({sz}x{sz})", fontsize=8)

            # Row 1: Depth CNN features (channel-mean magnitude, first view)
            ax1 = axes[1, stage_idx]
            if depth_stages is not None and stage_idx < len(depth_stages):
                dfeat = depth_stages[stage_idx]  # (B*num_views, C, H, W)
                dfeat_view0 = dfeat[0]  # (C, H, W)
                dmag = dfeat_view0.abs().mean(dim=0).numpy()  # (H, W)
                im1 = ax1.imshow(dmag, cmap="inferno")
                fig.colorbar(im1, ax=ax1, fraction=0.046)
            else:
                ax1.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax1.transAxes)

            # Row 2: FastGuide attention (first view)
            ax2 = axes[2, stage_idx]
            if fg_attns is not None and stage_idx < len(fg_attns):
                fg = fg_attns[stage_idx]  # (B*num_views, 1, H, W)
                fg_view0 = fg[0, 0].numpy()  # (H, W)
                im2 = ax2.imshow(fg_view0, cmap="hot")
                fig.colorbar(im2, ax=ax2, fraction=0.046)
            else:
                ax2.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax2.transAxes)

        path = step_dir / "fastguide_pyramid.png"
        self._save_fig(fig, path)

        return {
            "file": "fastguide_pyramid.png",
            "what": "3-row x 4-col pyramid: ViT projected features, depth CNN features, and FastGuide spatial attention at each of 4 stages (56, 28, 14, 7).",
            "how_to_read": "Row 0: ViT feature magnitudes (channel mean). Row 1: Depth CNN feature magnitudes (channel mean). "
                           "Row 2: FastGuide attention maps. First view only. Columns = stages 1-4.",
            "expect": "ViT and depth features have matching spatial patterns; FastGuide focuses on relevant regions.",
            "concern_if": "ViT features look random or depth CNN features are all zero (no gradient signal).",
        }

    # ------------------------------------------------------------------
    # 12. Reshape verification
    # ------------------------------------------------------------------
    def _viz_reshape_verification(self, step_dir):
        """Per-view panel to verify ViT feature reshape is spatially correct."""
        vit_projected = self._collected.get("vit_projected")
        rgb_pixels = self._collected.get("rgb_pixels")
        depth_input = self._collected.get("depth_input")
        num_views = self._collected.get("num_views", 1)

        if vit_projected is None:
            return None
        if len(vit_projected) < 4:
            return None

        saved_files = []

        for v in range(num_views):
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f"Reshape Verification (view {v})", fontsize=10)

            # Col 0: RGB image
            ax = axes[0]
            if rgb_pixels is not None and v < rgb_pixels.shape[0]:
                rgb_np = rgb_pixels[v].permute(1, 2, 0).numpy()
                rgb_np = np.clip(rgb_np, 0, 1) if rgb_np.max() <= 1.0 else np.clip(rgb_np / 255.0, 0, 1)
                ax.imshow(rgb_np)
                ax.set_title("RGB (ground truth)", fontsize=9)
            else:
                ax.text(0.5, 0.5, "RGB N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title("RGB (N/A)", fontsize=9)

            # Col 1: ViT features at stage 0 (56x56, upsampled from 16x16)
            ax = axes[1]
            feat0 = vit_projected[0]  # (B*num_views, C, H, W)
            view_idx = v if v < feat0.shape[0] else 0
            mag0 = feat0[view_idx].abs().mean(dim=0).numpy()
            im1 = ax.imshow(mag0, cmap="viridis")
            ax.set_title(f"ViT stage 1 ({mag0.shape[0]}x{mag0.shape[1]})", fontsize=9)
            fig.colorbar(im1, ax=ax, fraction=0.046)

            # Col 2: ViT features at stage 3 (7x7, native)
            ax = axes[2]
            feat3 = vit_projected[3]  # (B*num_views, C, H, W)
            view_idx3 = v if v < feat3.shape[0] else 0
            mag3 = feat3[view_idx3].abs().mean(dim=0).numpy()
            im2 = ax.imshow(mag3, cmap="viridis")
            ax.set_title(f"ViT stage 4 ({mag3.shape[0]}x{mag3.shape[1]}, native)", fontsize=9)
            fig.colorbar(im2, ax=ax, fraction=0.046)

            # Col 3: Depth image (reference)
            ax = axes[3]
            if depth_input is not None:
                depth_idx = v if v < depth_input.shape[0] else 0
                depth_2d = depth_input[depth_idx, 0].numpy()
                im3 = ax.imshow(depth_2d, cmap="viridis")
                ax.set_title(
                    f"Depth (ref)\nrange: [{depth_2d.min():.2f}, {depth_2d.max():.2f}]",
                    fontsize=9,
                )
                fig.colorbar(im3, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, "Depth N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title("Depth (N/A)", fontsize=9)

            fname = f"reshape_verification_view{v}.png"
            self._save_fig(fig, step_dir / fname)
            saved_files.append(fname)

        if not saved_files:
            return None

        return {
            "file": saved_files[0],
            "what": "Per-view panel: RGB | ViT stage-1 features (56x56) | ViT stage-4 features (7x7) | depth image.",
            "how_to_read": "If reshape is correct, spatial patterns in ViT feature maps should match RGB structure. "
                           "Stage-1 (col 1) is upsampled from 16x16 ViT patches; stage-4 (col 2) is at native 7x7 resolution.",
            "expect": "ViT feature spatial patterns (edges, objects) align with RGB image structure.",
            "concern_if": "ViT features look random or have no correlation with RGB spatial structure (reshape is wrong).",
        }

    # ------------------------------------------------------------------
    # Index file
    # ------------------------------------------------------------------
    def _write_index(self, step_dir, entries, step):
        """Write index.txt documenting all files in this step directory."""
        lines = []
        lines.append(f"CHNet Diagnostics Index — Step {step}")
        lines.append(f"Generated: {datetime.datetime.now().isoformat()}")
        lines.append("=" * 60)

        for entry in entries:
            if entry is None:
                continue
            lines.append("")
            lines.append(f"FILE: {entry.get('file', 'unknown')}")
            if "error" in entry:
                lines.append(f"  ERROR: {entry['error']}")
                continue
            lines.append(f"  WHAT:       {entry.get('what', '')}")
            lines.append(f"  HOW TO READ: {entry.get('how_to_read', '')}")
            lines.append(f"  EXPECT:     {entry.get('expect', '')}")
            lines.append(f"  CONCERN IF: {entry.get('concern_if', '')}")

        lines.append("")
        lines.append("FILE: health_check.txt")
        lines.append("  WHAT:       Numeric health checks (depth stats, weight norms, drift, change ratio).")
        lines.append("  HOW TO READ: Each value has expected range in parentheses.")
        lines.append("  EXPECT:     Values within stated ranges.")
        lines.append("  CONCERN IF: depth_input is all zeros, change_ratio < 1e-5, or zero gradients.")

        index_path = step_dir / "index.txt"
        index_path.write_text("\n".join(lines) + "\n")
