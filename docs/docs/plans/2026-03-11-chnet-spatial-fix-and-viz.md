# CHNet Spatial Fix & Diagnostic Visualizations

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the cross-attention fusion to only modify image tokens (not text), fix all diagnostic visualizations that use broken `sqrt()` spatial heuristics, and add 4 new visualizations (eagle token heatmap, depth token heatmap, FastGuide pyramid, reshape verification).

**Architecture:** Pass spatial metadata (`grid_h`, `grid_w`, `num_views`) through the diagnostics pipeline. Slice eagle features in the fusion module so only image tokens receive depth signal. Replace all `sqrt()`-based spatial guessing with actual dimensions. Add new per-view visualizations that overlay token activations on RGB/depth images.

**Tech Stack:** PyTorch, matplotlib, numpy, PIL (optional)

---

## File Structure

| File | Role | Changes |
|------|------|---------|
| `src/lerobot/policies/groot/chnet_modules.py` | CHNet modules (encoder, fusion, orchestrator) | Fusion: accept `n_image_tokens` to slice Q. Encoder: return projected ViT features in diagnostics. |
| `src/lerobot/policies/groot/groot_n1.py` | GR00T backbone forward pass | Compute & pass `num_views`, `n_image_tokens`, `grid_h`, `grid_w` to both CHNet forward and diagnostics collect. |
| `src/lerobot/policies/groot/chnet_diagnostics.py` | Diagnostic collection & visualization | Add spatial metadata to `collect()`. Fix 4 existing viz methods. Add 4 new viz methods. |

---

## Task 1: Fusion — Only Cross-Attend Image Tokens

The cross-attention currently uses ALL eagle tokens (image + text) as queries. Text tokens have no spatial correspondence to depth and produce uniform attention, diluting the signal. Fix: only cross-attend image tokens, pass text tokens through unchanged.

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_modules.py:224-275` (DepthCrossAttentionFusion)
- Modify: `src/lerobot/policies/groot/chnet_modules.py:278-321` (CHNetDepthProcessor.forward)
- Modify: `src/lerobot/policies/groot/groot_n1.py:501-529` (forward_eagle CHNet call site)

- [ ] **Step 1: Modify `DepthCrossAttentionFusion.forward` to accept `n_image_tokens`**

In `chnet_modules.py`, change the `forward` signature and add image-token slicing:

```python
def forward(self, eagle_features, depth_features, n_image_tokens=None, return_diagnostics=False):
    """
    Args:
        eagle_features: (B, seq_len, hidden_dim) from Eagle model
        depth_features: (N, C, H, W) from DepthCNNEncoder, where N = B * num_views
        n_image_tokens: Number of image tokens at the start of eagle sequence.
            If provided, only image tokens are cross-attended; text tokens pass through unchanged.
        return_diagnostics: if True, also return attention weights and depth tokens
    Returns:
        (B, seq_len, hidden_dim) enriched features
        If return_diagnostics: tuple of (enriched_features, attn_weights, depth_tokens)
    """
    n, c, h, w = depth_features.shape
    b_eagle = eagle_features.shape[0]
    depth_tokens = depth_features.flatten(2).transpose(1, 2)  # (N, H*W, C)
    depth_tokens = self.depth_proj(depth_tokens)  # (N, H*W, hidden_dim)

    # Handle multi-view: merge view tokens into batch dim
    if n != b_eagle:
        num_views = n // b_eagle
        tokens_per_view = depth_tokens.shape[1]
        depth_tokens = depth_tokens.view(b_eagle, num_views * tokens_per_view, -1)

    # Slice to image tokens only — text tokens don't need depth signal
    if n_image_tokens is not None and n_image_tokens < eagle_features.shape[1]:
        image_tokens = eagle_features[:, :n_image_tokens]
        text_tokens = eagle_features[:, n_image_tokens:]
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
            average_attn_weights=False,
        )
        fused_image = self.norm(image_tokens + attn_out)
        if text_tokens is not None:
            result = torch.cat([fused_image, text_tokens], dim=1)
        else:
            result = fused_image
        return result, attn_weights, depth_tokens
    else:
        attn_out, _ = self.cross_attn(
            query=image_tokens,
            key=depth_tokens,
            value=depth_tokens,
        )
        fused_image = self.norm(image_tokens + attn_out)
        if text_tokens is not None:
            result = torch.cat([fused_image, text_tokens], dim=1)
        else:
            result = fused_image
        return result
```

- [ ] **Step 2: Update `CHNetDepthProcessor.forward` to pass `n_image_tokens` through**

In `chnet_modules.py`, add `n_image_tokens=None` parameter to `CHNetDepthProcessor.forward` and pass it to `self.fusion(...)`:

```python
def forward(self, depth, vit_features, grid_h, grid_w, eagle_features,
            n_image_tokens=None, return_diagnostics=False):
    # ... encoder unchanged ...
    if return_diagnostics:
        depth_feat, fastguide_attns = self.encoder(
            depth, vit_features, grid_h, grid_w, return_diagnostics=True
        )
        result, attn_weights, depth_tokens = self.fusion(
            eagle_features, depth_feat, n_image_tokens=n_image_tokens,
            return_diagnostics=True,
        )
        return result, {
            "attn_weights": attn_weights,
            "fastguide_attns": fastguide_attns,
            "depth_tokens": depth_tokens,
        }
    else:
        depth_feat = self.encoder(depth, vit_features, grid_h, grid_w)
        return self.fusion(eagle_features, depth_feat, n_image_tokens=n_image_tokens)
```

- [ ] **Step 3: Compute `num_views` and `n_image_tokens` in `groot_n1.py` and pass to CHNet**

In `forward_eagle`, after computing `grid_h`, `grid_w` (line ~480), add:

```python
num_views = pixel_values.shape[0] // eagle_features.shape[0] if pixel_values is not None else 1
n_image_tokens = num_views * grid_h * grid_w
```

Then update both CHNet call sites (diag path line ~502 and normal path line ~523) to pass `n_image_tokens=n_image_tokens`.

- [ ] **Step 4: Verify no shape errors**

Run a quick import/syntax check:
```bash
cd /home/ofekpear/workspace/depth_vla_chnet/lerobot && python -c "from lerobot.policies.groot.chnet_modules import CHNetDepthProcessor; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/policies/groot/chnet_modules.py src/lerobot/policies/groot/groot_n1.py
git commit -m "fix: cross-attention fusion only modifies image tokens, not text tokens"
```

---

## Task 2: Return Projected ViT Features from Encoder for Diagnostics

The `DepthCNNEncoder` currently returns only `fastguide_attns` in diagnostics mode. To visualize the FastGuide pyramid and verify the 2D reshape, we also need the projected ViT features at each stage.

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_modules.py:173-221` (DepthCNNEncoder.forward)
- Modify: `src/lerobot/policies/groot/chnet_modules.py:306-317` (CHNetDepthProcessor diagnostics dict)

- [ ] **Step 1: Collect projected ViT features in DepthCNNEncoder.forward**

In the diagnostics path, capture each `rgb{N}` tensor alongside the fastguide attentions. Add a list:

```python
vit_projected = [] if return_diagnostics else None
```

After each `self.proj{N}(...)` call, append the result when in diagnostics mode:

```python
# Example for stage 1:
rgb1 = self.proj1(vit_features[0], grid_h, grid_w)
if return_diagnostics:
    vit_projected.append(rgb1)
    x, attn = self.guide1(x, rgb1, return_diagnostics=True)
    fastguide_attns.append(attn)
```

Same for stages 2, 3, 4.

Return both: `return x, fastguide_attns, vit_projected`

- [ ] **Step 2: Also capture depth CNN features at each stage (before FastGuide)**

To show the pyramid side-by-side (ViT projected vs depth CNN at each stage), capture the depth features too:

```python
depth_stages = [] if return_diagnostics else None
```

After each stage but BEFORE FastGuide:
```python
x = self.stage1(x)
rgb1 = self.proj1(vit_features[0], grid_h, grid_w)
if return_diagnostics:
    vit_projected.append(rgb1)
    depth_stages.append(x.clone())  # depth features BEFORE guidance
    x, attn = self.guide1(x, rgb1, return_diagnostics=True)
    fastguide_attns.append(attn)
```

Return: `return x, fastguide_attns, vit_projected, depth_stages`

- [ ] **Step 3: Update CHNetDepthProcessor to pass new data through diagnostics dict**

```python
return result, {
    "attn_weights": attn_weights,
    "fastguide_attns": fastguide_attns,
    "depth_tokens": depth_tokens,
    "vit_projected": vit_projected,
    "depth_stages": depth_stages,
}
```

Update the encoder call to unpack all 4 returns:
```python
depth_feat, fastguide_attns, vit_projected, depth_stages = self.encoder(
    depth, vit_features, grid_h, grid_w, return_diagnostics=True
)
```

- [ ] **Step 4: Commit**

```bash
git add src/lerobot/policies/groot/chnet_modules.py
git commit -m "feat: return projected ViT features and depth stages in diagnostics"
```

---

## Task 3: Add Spatial Metadata to Diagnostics Collection

Pass `grid_h`, `grid_w`, `num_views` and the new pyramid data into `collect()` so all visualizations can use actual spatial dimensions instead of `sqrt()` heuristics.

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_diagnostics.py:319-345` (collect method)
- Modify: `src/lerobot/policies/groot/groot_n1.py:510-520` (collect call site)

- [ ] **Step 1: Extend `collect()` signature with spatial metadata**

```python
def collect(self, step, depth_input, eagle_features_before, eagle_features_after,
            attn_weights=None, fastguide_attns=None, depth_tokens=None,
            rgb_pixels=None, is_training=True,
            grid_h=None, grid_w=None, num_views=1,
            n_image_tokens=None,
            vit_projected=None, depth_stages=None):
```

Store the new fields:
```python
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
    # Spatial metadata
    "grid_h": grid_h,
    "grid_w": grid_w,
    "num_views": num_views,
    "n_image_tokens": n_image_tokens,
    # Pyramid data
    "vit_projected": [_safe_detach(v) for v in vit_projected] if vit_projected else None,
    "depth_stages": [_safe_detach(d) for d in depth_stages] if depth_stages else None,
}
```

- [ ] **Step 2: Update the collect call in `groot_n1.py`**

```python
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
    grid_h=grid_h,
    grid_w=grid_w,
    num_views=num_views,
    n_image_tokens=n_image_tokens,
    vit_projected=diag_data["vit_projected"],
    depth_stages=diag_data["depth_stages"],
)
```

- [ ] **Step 3: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py src/lerobot/policies/groot/groot_n1.py
git commit -m "feat: pass spatial metadata and pyramid data to diagnostics"
```

---

## Task 4: Fix Existing Visualizations

Replace `sqrt()` heuristics in 4 existing viz methods with actual spatial metadata. The depth CNN always outputs `7×7` per view. Eagle image tokens are `grid_h × grid_w` per view.

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_diagnostics.py`

### 4a: Fix `_viz_input_overlay` (lines 738-816)

- [ ] **Step 1: Fix attention overlay — use `(7, 7)` per view instead of `sqrt()`**

Replace lines 784-800 (the `sqrt()` heuristic block) with:

```python
# Attention overlaid on depth
if has_attn:
    num_views = self._collected.get("num_views", 1)
    # Average attention over heads and image query tokens → per-depth-token map
    # attn shape: (B, heads, n_image_tokens, num_views*49)
    attn_avg = attn[0].mean(dim=0).mean(dim=0).numpy()  # (num_views * 49,)
    tokens_per_view = 49  # depth CNN outputs 7×7
    depth_h, depth_w = 7, 7

    # Show first view only (or could loop for multi-view)
    view_attn = attn_avg[:tokens_per_view]
    attn_map = view_attn.reshape(depth_h, depth_w)

    # Resize to depth image size
    if _HAS_PIL:
        attn_img = PILImage.fromarray(
            (attn_map / (attn_map.max() + 1e-8) * 255).astype(np.uint8)
        )
        attn_img = attn_img.resize((depth_2d.shape[1], depth_2d.shape[0]),
                                    PILImage.BILINEAR)
        attn_resized = np.array(attn_img).astype(np.float32) / 255.0
    else:
        attn_resized = attn_map

    axes[col].imshow(depth_2d, cmap="viridis", alpha=0.6)
    axes[col].imshow(attn_resized, cmap="hot", alpha=0.4)
    axes[col].set_title("Attention on Depth (view 0)", fontsize=9)
```

### 4b: Fix `_viz_depth_contribution_map` (lines 821-862)

- [ ] **Step 2: Fix spatial view — use `grid_h × grid_w` for image tokens**

Replace the `sqrt()` heuristic (lines 843-850) with:

```python
# Spatial view for image tokens using actual grid dimensions
grid_h = self._collected.get("grid_h")
grid_w = self._collected.get("grid_w")
num_views = self._collected.get("num_views", 1)
n_image_tokens = self._collected.get("n_image_tokens")

if grid_h and grid_w and n_image_tokens:
    # Only show image token contributions (first n_image_tokens)
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
```

### 4c: Fix `_viz_cross_attn_map` (lines 611-647)

- [ ] **Step 3: Annotate axes with spatial info**

Add spatial annotations so it's clear what the token indices map to:

```python
n_image_tokens = self._collected.get("n_image_tokens")
num_views = self._collected.get("num_views", 1)
tokens_per_view_depth = 49  # 7x7

for h in range(n_heads):
    ax = axes[h]
    im = ax.imshow(attn_b0[h].numpy(), aspect="auto", cmap="hot")
    ax.set_title(f"Head {h}", fontsize=8)
    ax.set_xlabel(f"depth token (={num_views}×7×7)", fontsize=7)
    if n_image_tokens:
        ax.set_ylabel(f"image token (={n_image_tokens})", fontsize=7)
    else:
        ax.set_ylabel("eagle token", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
```

Note: after the fusion fix (Task 1), `attn_weights` will have shape `(B, heads, n_image_tokens, num_views*49)` — only image tokens as Q. So the heatmap will only show image tokens, not text.

### 4d: Fix `_viz_embedding_similarity` (lines 686-733)

- [ ] **Step 4: Split tokens by image/text for clearer comparison**

No broken sqrt() here, but clarity improves by noting which tokens are image vs text. Add a comment annotation:

```python
n_image_tokens = self._collected.get("n_image_tokens")
# Subsample image tokens only for cleaner comparison
if n_image_tokens:
    max_tokens = min(64, n_image_tokens)
    eb = eagle_before[0][:max_tokens]
    ea = eagle_after[0][:max_tokens]
else:
    max_tokens = 64
    eb = eagle_before[0][:max_tokens]
    ea = eagle_after[0][:max_tokens]
```

Update axis labels to say "image token" when `n_image_tokens` is known.

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py
git commit -m "fix: replace sqrt() spatial heuristics with actual grid dimensions in diagnostics"
```

---

## Task 5: New Visualization — Eagle Token Heatmap Over RGB

Show eagle token activations (from attention received or contribution magnitude) as a heatmap overlaid on the RGB image, per view.

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_diagnostics.py` (add method + register in viz list)

- [ ] **Step 1: Add `_viz_eagle_token_heatmap` method**

```python
def _viz_eagle_token_heatmap(self, step_dir):
    """Per-view heatmap of eagle image token contribution overlaid on RGB."""
    eagle_before = self._collected.get("eagle_before")
    eagle_after = self._collected.get("eagle_after")
    rgb = self._collected.get("rgb_pixels")
    grid_h = self._collected.get("grid_h")
    grid_w = self._collected.get("grid_w")
    num_views = self._collected.get("num_views", 1)
    n_image_tokens = self._collected.get("n_image_tokens")

    if eagle_before is None or eagle_after is None or grid_h is None:
        return None

    # Per-token contribution magnitude (L2 of delta)
    diff = (eagle_after[0] - eagle_before[0])  # (seq, D)
    magnitudes = diff.norm(dim=-1).numpy()  # (seq,)

    tokens_per_view = grid_h * grid_w
    fig, axes = plt.subplots(1, num_views, figsize=(6 * num_views, 5),
                              squeeze=False)
    fig.suptitle("Eagle Token Depth-Contribution Heatmap over RGB", fontsize=10)

    for v in range(num_views):
        ax = axes[0, v]
        start = v * tokens_per_view
        end = start + tokens_per_view

        # Reshape image token magnitudes to spatial grid
        if end <= len(magnitudes):
            token_map = magnitudes[start:end].reshape(grid_h, grid_w)
        else:
            ax.set_title(f"View {v}: insufficient tokens", fontsize=9)
            continue

        # Show RGB if available (rgb_pixels is (B*num_views, C, H, W))
        if rgb is not None and v < rgb.shape[0]:
            rgb_np = rgb[v].permute(1, 2, 0).numpy()
            rgb_np = np.clip(rgb_np, 0, 1) if rgb_np.max() <= 1.0 else np.clip(rgb_np / 255.0, 0, 1)
            ax.imshow(rgb_np)

        # Overlay token heatmap, upsampled to image size
        if _HAS_PIL and rgb is not None:
            h_img, w_img = rgb.shape[2], rgb.shape[3]
            hm_img = PILImage.fromarray(
                (token_map / (token_map.max() + 1e-8) * 255).astype(np.uint8)
            )
            hm_img = hm_img.resize((w_img, h_img), PILImage.BILINEAR)
            hm_resized = np.array(hm_img).astype(np.float32) / 255.0
            ax.imshow(hm_resized, cmap="hot", alpha=0.5)
        else:
            ax.imshow(token_map, cmap="hot")

        ax.set_title(
            f"View {v} ({grid_h}x{grid_w})\n"
            f"max={token_map.max():.4f}, mean={token_map.mean():.4f}",
            fontsize=9,
        )

    path = step_dir / "eagle_token_heatmap.png"
    self._save_fig(fig, path)

    return {
        "file": "eagle_token_heatmap.png",
        "what": "Per-view heatmap of eagle image token depth-contribution (L2 of delta) overlaid on RGB.",
        "how_to_read": "Hot regions show which spatial locations were most modified by depth fusion. "
                       "Each panel is one camera view.",
        "expect": "Activation concentrated on task-relevant objects/surfaces with depth variation.",
        "concern_if": "Uniform activation (depth not discriminating spatially) or all zeros.",
    }
```

- [ ] **Step 2: Register in viz methods list**

In the `report()` method (around line 372-381), add to the `viz_methods` list:

```python
("eagle_token_heatmap", self._viz_eagle_token_heatmap),
```

- [ ] **Step 3: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py
git commit -m "feat: add eagle token heatmap over RGB visualization"
```

---

## Task 6: New Visualization — Depth Token Heatmap Over Depth

Show depth token activations (attention received from eagle, or token magnitude) as a `7×7` heatmap overlaid on the depth image, per view.

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_diagnostics.py`

- [ ] **Step 1: Add `_viz_depth_token_heatmap` method**

```python
def _viz_depth_token_heatmap(self, step_dir):
    """Per-view heatmap of depth token attention-received overlaid on depth image."""
    attn = self._collected.get("attn_weights")
    depth_input = self._collected.get("depth_input")
    num_views = self._collected.get("num_views", 1)

    if attn is None or depth_input is None:
        return None

    tokens_per_view = 49  # depth CNN outputs 7×7
    depth_h, depth_w = 7, 7

    # Attention received by each depth token: sum over heads and query tokens
    # attn: (B, heads, n_image_q, num_views*49)
    attn_received = attn[0].mean(dim=0).sum(dim=0).numpy()  # (num_views * 49,)

    fig, axes = plt.subplots(1, num_views, figsize=(6 * num_views, 5),
                              squeeze=False)
    fig.suptitle("Depth Token Attention-Received Heatmap over Depth", fontsize=10)

    for v in range(num_views):
        ax = axes[0, v]
        start = v * tokens_per_view
        end = start + tokens_per_view

        if end <= len(attn_received):
            token_map = attn_received[start:end].reshape(depth_h, depth_w)
        else:
            ax.set_title(f"View {v}: insufficient tokens", fontsize=9)
            continue

        # Show depth image (depth_input is (B*num_views, 1, H, W))
        if v < depth_input.shape[0]:
            depth_2d = depth_input[v, 0].numpy()
            ax.imshow(depth_2d, cmap="viridis", alpha=0.6)

        # Overlay attention heatmap
        if _HAS_PIL and depth_input is not None:
            h_img, w_img = depth_input.shape[2], depth_input.shape[3]
            hm_img = PILImage.fromarray(
                (token_map / (token_map.max() + 1e-8) * 255).astype(np.uint8)
            )
            hm_img = hm_img.resize((w_img, h_img), PILImage.BILINEAR)
            hm_resized = np.array(hm_img).astype(np.float32) / 255.0
            ax.imshow(hm_resized, cmap="hot", alpha=0.5)
        else:
            ax.imshow(token_map, cmap="hot")

        ax.set_title(
            f"View {v} (7x7→{depth_input.shape[3]}x{depth_input.shape[2]})\n"
            f"max={token_map.max():.4f}, mean={token_map.mean():.4f}",
            fontsize=9,
        )

    path = step_dir / "depth_token_heatmap.png"
    self._save_fig(fig, path)

    return {
        "file": "depth_token_heatmap.png",
        "what": "Per-view heatmap of attention received by each depth token (7x7) overlaid on depth image.",
        "how_to_read": "Hot regions show which depth spatial locations eagle tokens attend to most. "
                       "Each panel is one camera view.",
        "expect": "Attention on depth regions with objects, edges, or task-relevant geometry.",
        "concern_if": "Uniform (near-uniform attention warning) or concentrated on a single pixel.",
    }
```

- [ ] **Step 2: Register in viz methods list**

```python
("depth_token_heatmap", self._viz_depth_token_heatmap),
```

- [ ] **Step 3: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py
git commit -m "feat: add depth token heatmap over depth image visualization"
```

---

## Task 7: New Visualization — FastGuide Pyramid

Show the ViT projected features and depth CNN features side-by-side at each of the 4 pyramid stages. This makes visible the resolution mismatch: ViT features are all `16×16` upsampled/downsampled to stage resolution, while depth CNN features are naturally hierarchical.

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_diagnostics.py`

- [ ] **Step 1: Add `_viz_fastguide_pyramid` method**

```python
def _viz_fastguide_pyramid(self, step_dir):
    """Side-by-side ViT projected features vs depth CNN features at each pyramid stage."""
    vit_projected = self._collected.get("vit_projected")
    depth_stages = self._collected.get("depth_stages")
    fg_attns = self._collected.get("fastguide_attns")

    if vit_projected is None or depth_stages is None:
        return None

    n_stages = min(len(vit_projected), len(depth_stages))
    stage_sizes = [56, 28, 14, 7]

    fig, axes = plt.subplots(3, n_stages, figsize=(4 * n_stages, 12))
    fig.suptitle("FastGuide Pyramid: ViT Projected | Depth CNN | FastGuide Attention", fontsize=10)

    for i in range(n_stages):
        # Row 0: ViT projected features (channel-mean magnitude)
        vit_feat = vit_projected[i][0]  # first view, (C, H, W)
        vit_mag = vit_feat.abs().mean(dim=0).numpy()  # (H, W)
        axes[0, i].imshow(vit_mag, cmap="plasma")
        axes[0, i].set_title(
            f"ViT→{stage_sizes[i]}x{stage_sizes[i]}\n(from 16x16)",
            fontsize=8,
        )
        if i == 0:
            axes[0, i].set_ylabel("ViT projected", fontsize=9)

        # Row 1: Depth CNN features (channel-mean magnitude)
        depth_feat = depth_stages[i][0]  # first view, (C, H, W)
        depth_mag = depth_feat.abs().mean(dim=0).numpy()  # (H, W)
        axes[1, i].imshow(depth_mag, cmap="plasma")
        axes[1, i].set_title(f"Depth CNN {stage_sizes[i]}x{stage_sizes[i]}", fontsize=8)
        if i == 0:
            axes[1, i].set_ylabel("Depth CNN", fontsize=9)

        # Row 2: FastGuide attention
        if fg_attns is not None and i < len(fg_attns):
            fg = fg_attns[i][0, 0].numpy()  # (H, W)
            axes[2, i].imshow(fg, cmap="inferno")
            axes[2, i].set_title(f"FastGuide attn {fg.shape[0]}x{fg.shape[1]}", fontsize=8)
        else:
            axes[2, i].set_visible(False)
        if i == 0:
            axes[2, i].set_ylabel("FastGuide attn", fontsize=9)

    path = step_dir / "fastguide_pyramid.png"
    self._save_fig(fig, path)

    return {
        "file": "fastguide_pyramid.png",
        "what": "FastGuide pyramid: ViT projected features (row 1), depth CNN features (row 2), "
                "and FastGuide attention (row 3) at stages 1-4.",
        "how_to_read": "Columns are pyramid stages (56x56 → 7x7). Row 1 shows ViT features "
                       "resized from 16x16 to each stage resolution — stages 1-2 are upsampled "
                       "(smooth), stages 3-4 are close to native. Row 2 shows depth CNN features "
                       "which are naturally at the right resolution. Row 3 shows FastGuide spatial "
                       "attention (how RGB modulates depth).",
        "expect": "ViT features: stages 1-2 smooth, 3-4 sharper. Depth CNN: progressively abstract. "
                  "FastGuide: focused on task-relevant regions.",
        "concern_if": "ViT features look identical across stages (projection not differentiating). "
                      "Depth features all zero. FastGuide attention uniform.",
    }
```

- [ ] **Step 2: Register in viz methods list**

```python
("fastguide_pyramid", self._viz_fastguide_pyramid),
```

- [ ] **Step 3: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py
git commit -m "feat: add FastGuide pyramid visualization"
```

---

## Task 8: New Visualization — Reshape Verification

Show ViT tokens reshaped to `(grid_h, grid_w)` as a spatial feature map next to the original RGB image. If the reshape is correct, spatial patterns should align (bright features where objects are). If wrong (transposed, rotated, flipped), the patterns will be misaligned.

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_diagnostics.py`

- [ ] **Step 1: Add `_viz_reshape_verification` method**

```python
def _viz_reshape_verification(self, step_dir):
    """Verify that ViT token → 2D reshape produces spatially correct feature maps.

    Shows: RGB image | ViT tokens reshaped to (grid_h, grid_w) | depth image.
    If reshape is correct, the ViT feature map spatial patterns should match
    visible structure in the RGB image (bright where objects are, etc.).
    """
    vit_projected = self._collected.get("vit_projected")
    rgb = self._collected.get("rgb_pixels")
    depth_input = self._collected.get("depth_input")
    grid_h = self._collected.get("grid_h")
    grid_w = self._collected.get("grid_w")
    num_views = self._collected.get("num_views", 1)

    if vit_projected is None or grid_h is None:
        return None

    # Use last stage (7x7, closest to native 16x16) for clearest verification
    # AND first stage (56x56, upsampled from 16x16) to compare
    stages_to_show = [0, 3]  # stage 1 (56x56) and stage 4 (7x7)
    stage_names = ["Stage 1 (56x56)", "Stage 4 (7x7)"]

    for v in range(min(num_views, 2)):  # show up to 2 views
        n_cols = 1 + len(stages_to_show) + (1 if depth_input is not None else 0)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        fig.suptitle(f"Reshape Verification — View {v}", fontsize=10)
        col = 0

        # RGB
        if rgb is not None and v < rgb.shape[0]:
            rgb_np = rgb[v].permute(1, 2, 0).numpy()
            rgb_np = np.clip(rgb_np, 0, 1) if rgb_np.max() <= 1.0 else np.clip(rgb_np / 255.0, 0, 1)
            axes[col].imshow(rgb_np)
            axes[col].set_title("RGB (ground truth)", fontsize=9)
            col += 1

        # ViT feature maps at selected stages
        for s_idx, s_name in zip(stages_to_show, stage_names):
            if s_idx < len(vit_projected) and v < vit_projected[s_idx].shape[0]:
                feat = vit_projected[s_idx][v]  # (C, H, W)
                # Channel-mean magnitude
                feat_mag = feat.abs().mean(dim=0).numpy()
                axes[col].imshow(feat_mag, cmap="plasma")
                axes[col].set_title(f"ViT {s_name}\n(should match RGB layout)", fontsize=8)
            col += 1

        # Depth
        if depth_input is not None and v < depth_input.shape[0]:
            axes[col].imshow(depth_input[v, 0].numpy(), cmap="viridis")
            axes[col].set_title("Depth (reference)", fontsize=9)

        path = step_dir / f"reshape_verification_view{v}.png"
        self._save_fig(fig, path)

    return {
        "file": "reshape_verification_view0.png",
        "what": "Reshape verification: RGB | ViT features reshaped to 2D | depth, side by side.",
        "how_to_read": "ViT feature maps should show spatial patterns that match the RGB image "
                       "(bright where objects are, edges where RGB has edges). If the image looks "
                       "scrambled, rotated, or flipped relative to RGB, the reshape is wrong.",
        "expect": "ViT feature maps roughly align with RGB spatial structure.",
        "concern_if": "Feature map looks spatially scrambled, rotated 90°, or flipped vs RGB. "
                      "Stage 1 (56x56) should look smoother than stage 4 (7x7) since it's upsampled.",
    }
```

- [ ] **Step 2: Register in viz methods list**

```python
("reshape_verification", self._viz_reshape_verification),
```

- [ ] **Step 3: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py
git commit -m "feat: add reshape verification visualization"
```

---

## Task 9: Update Health Check Entropy Calculation

After Task 1, `attn_weights` will have shape `(B, heads, n_image_tokens, num_views*49)` — only image tokens as Q. The entropy calculation in `_health_checks` should note this in the output for clarity.

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_diagnostics.py:447-461`

- [ ] **Step 1: Update entropy logging to show token counts**

```python
# After computing entropy_ratio:
n_image_tokens = self._collected.get("n_image_tokens")
num_views = self._collected.get("num_views", 1)
lines.append(
    f"  attn_entropy      : {entropy:.3f} / {max_entropy:.3f} "
    f"(ratio={entropy_ratio:.3f}, Q={attn_b0.shape[1]} image tokens, "
    f"K={attn_b0.shape[2]} depth tokens = {num_views}×7×7)"
)
```

- [ ] **Step 2: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py
git commit -m "fix: update health check entropy to show image/depth token counts"
```

---

## Task 10: Final Verification

- [ ] **Step 1: Import check**

```bash
cd /home/ofekpear/workspace/depth_vla_chnet/lerobot
python -c "
from lerobot.policies.groot.chnet_modules import CHNetDepthProcessor
from lerobot.policies.groot.chnet_diagnostics import DepthDiagnostics
print('Imports OK')
"
```

- [ ] **Step 2: Verify all viz methods are registered**

```bash
python -c "
from lerobot.policies.groot.chnet_diagnostics import DepthDiagnostics
import inspect
src = inspect.getsource(DepthDiagnostics.report)
expected = ['cross_attn_map', 'fastguide_stages', 'embedding_similarity',
            'input_overlay', 'depth_contribution_map', 'proximity_direction',
            'token_space_pca', 'nearest_neighbors',
            'eagle_token_heatmap', 'depth_token_heatmap',
            'fastguide_pyramid', 'reshape_verification']
for name in expected:
    assert name in src, f'Missing viz method: {name}'
print(f'All {len(expected)} viz methods registered')
"
```

- [ ] **Step 3: Commit plan doc**

```bash
git add docs/docs/plans/2026-03-11-chnet-spatial-fix-and-viz.md
git commit -m "docs: add CHNet spatial fix and visualization plan"
```
