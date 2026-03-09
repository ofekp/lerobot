# CHNET Depth Diagnostics and Interpretability Layer

**Date:** 2026-03-09
**Branch:** claude/chnet
**Status:** Approved

## Goal

Add a diagnostics and interpretability layer to the CHNET depth pipeline that:
- Validates the pipeline is working correctly without clogging logs
- Provides interpretable visualizations of what depth information the model uses
- Includes expected-value annotations so anyone reading logs knows if something is off
- Documents each visualization for human or AI readers

## Architecture: Single Diagnostics Module (Approach A)

New file `chnet_diagnostics.py` containing a `DepthDiagnostics` class. Hooks into the existing forward pass passively. Minimal changes to existing model code.

### Touch Points

1. **New file:** `src/lerobot/policies/groot/chnet_diagnostics.py` — all diagnostics logic
2. **Modified:** `chnet_modules.py` — add `return_diagnostics` flag to forward methods to optionally return intermediate tensors (attention weights, FastGuide spatial attention)
3. **Modified:** `groot_n1.py` — replace current inline print statements with calls to diagnostics module; pass intermediate data
4. **Modified:** Training/eval loop — call `diagnostics.report(step)` at throttle interval

## Component 1: Startup Banner (once per run)

Printed at the start of training and evaluation. Includes expected values inline.

```
==================================================
 GROOT DEPTH DIAGNOSTICS - Run Summary
==================================================
 Git hash:            8683490       (verify: matches your intended commit)
 Branch:              claude/chnet
 use_depth:           True          (expect: True for depth experiments)
 depth_scale/mean/std: 1.0/0.0/1.0 (expect: pass-through for metric depth in meters)
 chnet_tap_layers:    (5, 11, 17, 23)  (expect: 4 evenly-spaced layers in 24-layer ViT)
 chnet_channels:      (64, 128, 256, 256)
 Cameras (train):     ['top', 'wrist']  (2 cameras - verify this matches intent)
 CHNet params:        1.2M trainable    (expect: ~1-2M for default channels)
 Total model params:  3.1B / 1.2M trainable
==================================================
```

Information collected via:
- `subprocess.check_output(['git', 'rev-parse', 'HEAD'])` for git hash
- `subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])` for branch
- Config object attributes for depth settings
- `sum(p.numel() for p in model.parameters())` for param counts

## Component 2: Periodic Health Checks (every 500 training steps)

Compact multi-line summary with expected ranges:

```
[DIAG step=500 TRAIN]
  depth_input:    range=[0.12, 2.84] mean=1.03 zeros=0.0%    (expect: range~[0.01,5.0] for indoor sim, zeros<1%)
  chnet_grad:     encoder=0.042 fusion=0.031                  (expect: 0.001-1.0; <0.001=vanishing, >10=exploding)
  weight_norms:   encoder=12.3 fusion=8.7                     (expect: stable across steps; drift>50% from init=concern)
  change_ratio:   0.0031                                      (expect: 0.001-0.1; <0.0001=depth ignored, >1.0=depth dominates)
  depth_grad:     max=0.15 mean=0.003                         (expect: non-zero; 0.0=no gradient flowing to depth)
```

### Metrics collected

| Metric | Source | What it tells you |
|--------|--------|-------------------|
| depth_input range/mean/zeros | Input tensor to CHNet | Data quality — confirms depth values are physical (meters) |
| chnet_grad encoder/fusion | `param.grad.norm()` aggregated | Gradient health — vanishing or exploding |
| weight_norms encoder/fusion | `param.data.norm()` aggregated | Weight stability over time |
| change_ratio | `(after - before).norm() / before.norm()` | How much depth modifies the final features |
| depth_grad max/mean | Gradient of loss w.r.t. depth input tensor | Whether gradients flow back to depth at all |

### Throttling

- Training: every 500 steps (configurable)
- Eval: first episode + every N episodes
- Startup banner: once

## Component 3: Interpretability Visualizations (every 500 steps)

Saved to `{output_dir}/diagnostics/step_{N}/` with an `index.txt`.

### 3a. Cross-Attention Map (`cross_attn_map.png`)

**What:** Heatmap of attention weights from `DepthCrossAttentionFusion` (Q=eagle_tokens, K/V=depth_tokens).

**Implementation:** Change `attn_out, _ = self.cross_attn(...)` to `attn_out, attn_weights = self.cross_attn(..., need_weights=True)` when diagnostics are active. Average attention across heads, reshape depth token dimension to spatial grid (7x7 per view).

**Expect:** Bright regions on object surfaces and edges where depth is informative. Non-uniform distribution.
**Concern if:** Uniform attention (model ignoring depth structure) or all-zero weights.

### 3b. FastGuide Spatial Attention (`fastguide_stage{1-4}.png`)

**What:** The `avg_attn` tensor from each FastGuide module — shows which spatial regions the RGB-to-depth guidance focuses on at each CNN stage.

**Implementation:** Store `avg_attn` during forward when diagnostics flag is set. 4 images, one per stage, at resolutions 56x56, 28x28, 14x14, 7x7 (upsampled for display).

**Expect:** Early stages (1-2) capture low-level features (edges, surfaces). Later stages (3-4) capture semantic regions (objects, obstacles).
**Concern if:** All-uniform or all-zero at any stage.

### 3c. Depth-RGB Embedding Similarity (`embedding_similarity.png`)

**What:** Cosine similarity heatmap between depth tokens and eagle tokens after projection (before fusion) and after fusion.

**Implementation:** Compute pairwise cosine similarity between depth_tokens (N_depth x D) and eagle_features (N_eagle x D). Show as two side-by-side heatmaps (before/after).

**Expect:** After fusion, some depth tokens should show increased similarity to relevant eagle tokens. Pattern should be structured, not random.
**Concern if:** No change before vs after, or completely uniform similarity.

### 3d. Input Overlay (`input_overlay.png`)

**What:** Side-by-side visualization: RGB image | depth image (colormap) | cross-attention heatmap overlaid on depth.

**Implementation:** Take the first camera's RGB and depth from the batch. Apply a colormap (viridis) to depth. Overlay cross-attention map (from 3a, averaged across eagle tokens) with alpha blending.

**Expect:** Attention overlay should highlight physically meaningful regions (objects being manipulated, surfaces near the gripper).
**Concern if:** Attention on background or uniform.

### index.txt Format

Each step directory contains an `index.txt` with this structure per visualization:

```
=== GROOT Depth Diagnostics - Step {N} ({TRAIN/EVAL}) ===

1. cross_attn_map.png
   WHAT: Heatmap of cross-attention weights (Q=eagle_tokens, K/V=depth_tokens).
   HOW TO READ: Bright regions = depth locations the model attends to most.
         Reshaped to 7x7 spatial grid per camera view.
   EXPECT: Highlights on object surfaces, edges, and task-relevant regions.
   CONCERN IF: Uniform attention (depth structure ignored) or all-zero.

2. fastguide_stage1.png ... fastguide_stage4.png
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
```

## Component 4: DepthDiagnostics Class API

```python
class DepthDiagnostics:
    def __init__(self, config, model, output_dir, train_interval=500, eval_interval=1):
        """
        Args:
            config: GrootConfig with depth settings
            model: the EagleBackbone or full model (for param counting)
            output_dir: base directory for saving diagnostics
            train_interval: steps between diagnostics during training
            eval_interval: episodes between diagnostics during eval
        """

    def print_startup_banner(self, dataset_info=None):
        """Print once at run start. Pass dataset_info for camera names."""

    def should_report(self, step, is_training=True):
        """Check if this step should produce diagnostics."""

    def collect(self, step, depth_input, eagle_features_before, eagle_features_after,
                attn_weights=None, fastguide_attns=None, depth_tokens=None,
                rgb_image=None, is_training=True):
        """Collect data for a single step. Called from forward_eagle."""

    def report(self, step, is_training=True):
        """Generate text summary and visualizations for collected data."""

    def log_gradients(self, step, model):
        """Called after backward pass to log gradient norms. Called from training loop."""
```

## Non-Goals (Explicitly Out of Scope)

- No wandb/tensorboard integration (local directory only)
- No changes to model architecture or training dynamics
- No persistent state across runs (each run starts fresh)
- No real-time streaming or dashboard

## File Output Structure

```
{output_dir}/
  diagnostics/
    startup_banner.txt          # Saved copy of the startup banner
    step_500/
      index.txt                 # Documented guide to all files
      cross_attn_map.png
      fastguide_stage1.png
      fastguide_stage2.png
      fastguide_stage3.png
      fastguide_stage4.png
      embedding_similarity.png
      input_overlay.png
    step_1000/
      ...
    eval_episode_0/
      ...
```
