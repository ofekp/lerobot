# CHNet Zero-Init, Diagnostics & Visualization Fixes

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the root cause of uniform attention (too-large residual from random init), reduce gradient log spam, and improve diagnostic visualizations.

**Architecture:** Three independent changes: (1) zero-init the cross-attention output projection so the depth branch starts as identity, (2) gate gradient logging behind the existing `train_interval`, (3) improve attention overlay visualization with entropy annotations and fix BN health check.

**Tech Stack:** PyTorch, matplotlib, numpy

---

## Chunk 1: Zero-Init + Gradient Logging

### Task 1: Zero-init cross-attention output projection

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_modules.py:237-243`

- [ ] **Step 1: Add zero-init after cross_attn creation**

In `DepthCrossAttentionFusion.__init__`, add two lines after the `nn.MultiheadAttention` creation:

```python
class DepthCrossAttentionFusion(nn.Module):
    """Cross-attention to fuse depth features into ViT token sequence."""

    def __init__(self, depth_channels, hidden_dim, num_heads=8):
        super().__init__()
        self.depth_proj = nn.Linear(depth_channels, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        # Zero-init output projection so depth branch starts as identity.
        # Without this, random cross-attention output adds ~50% noise to
        # pretrained eagle features, destabilizing training and causing
        # encoder gradient collapse.
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        self.norm = nn.LayerNorm(hidden_dim)
```

- [ ] **Step 2: Commit**

```bash
git add src/lerobot/policies/groot/chnet_modules.py
git commit -m "fix(chnet): zero-init cross-attn output projection for stable training"
```

### Task 2: Gate gradient logging behind train_interval

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_diagnostics.py:565` (log_gradients method)

- [ ] **Step 1: Add frequency check to log_gradients**

Add a `step % self.train_interval == 0` guard at the top of `log_gradients`:

```python
def log_gradients(self, step, model):
    """Log CHNet gradient norms per sub-module after backward pass."""
    if step % self.train_interval != 0 and step > 10:
        return
    # ... rest of method unchanged
```

This logs every step for the first 10 steps (for debugging startup), then every 500 steps (matching `train_interval`).

- [ ] **Step 2: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py
git commit -m "fix(chnet): reduce gradient log frequency to every 500 steps"
```

## Chunk 2: Visualization & Diagnostic Fixes

### Task 3: Fix input_overlay attention visualization

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_diagnostics.py:775-851` (_viz_input_overlay method)

- [ ] **Step 1: Improve the attention overlay panel**

Replace the attention overlay section (lines 820-839) with entropy-aware visualization:

```python
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
            title = f"Attention on Depth (view 0)"
            if is_uniform:
                title += "\n!! NEAR-UNIFORM !!"
            title += f"\nrange: [{attn_map.min():.4f}, {attn_map.max():.4f}]"
            title += f"  entropy: {entropy_ratio:.3f}"
            axes[col].set_title(title, fontsize=8)
```

- [ ] **Step 2: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py
git commit -m "fix(chnet): annotate input_overlay with entropy and uniform-attention warning"
```

### Task 4: Add 2D spatial view to cross_attn_map

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_diagnostics.py:636-678` (_viz_cross_attn_map method)

- [ ] **Step 1: Add spatial attention view per head per view**

Replace the `_viz_cross_attn_map` method to show both the flat heatmap and a 2D spatial summary:

```python
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

        # --- Figure 1: flat heatmap per head (existing) ---
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
            ax.set_title(f"Head {h} (ent={head_entropy/max_ent:.2f})", fontsize=7)
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
```

- [ ] **Step 2: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py
git commit -m "feat(chnet): add 2D spatial cross-attention map and per-head entropy"
```

### Task 5: Fix BN health check to distinguish eval mode vs running_mean

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_diagnostics.py:539-558`

- [ ] **Step 1: Make BN check more informative**

Replace the BN check block with one that separates the two conditions:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py
git commit -m "fix(chnet): separate BN eval-mode vs running-mean-stuck diagnostics"
```

### Task 6: Fix eval drift false 0%

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_diagnostics.py:117-118` (_snapshot_initial_weights)

- [ ] **Step 1: Prevent re-snapshotting after checkpoint load**

The issue: during eval, `DepthDiagnostics.__init__` snapshots weights AFTER the checkpoint is loaded, so drift is always 0%. Fix by adding a flag to skip re-snapshot if weights are already captured:

No code change needed for the snapshot itself — the real fix is that during eval, the diagnostics instance is created fresh with the loaded checkpoint weights. The drift comparison during eval is meaningless since there's no training. Instead, skip drift reporting during eval:

In `_health_checks`, around lines 490-530 (the weight drift section), add a guard:

```python
        # 4. Weight drift from init
        if self._collected.get("is_training", True):
            lines.append("  --- Weight Drift (vs init) ---")
            # ... existing drift code ...
        else:
            lines.append("  --- Weight Drift ---")
            lines.append("  (skipped during eval — init snapshot is post-checkpoint)")
```

- [ ] **Step 2: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py
git commit -m "fix(chnet): skip misleading weight drift during eval"
```

### Task 7: Add debug logging for new visualizations returning None

**Files:**
- Modify: `src/lerobot/policies/groot/chnet_diagnostics.py:402-414` (report method)

- [ ] **Step 1: Log when a visualization returns None**

In the viz loop, add a debug print for None returns:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/lerobot/policies/groot/chnet_diagnostics.py
git commit -m "fix(chnet): log when diagnostics visualizations return None"
```
