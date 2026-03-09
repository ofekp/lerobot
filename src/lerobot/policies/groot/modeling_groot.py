#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Groot Policy Wrapper for LeRobot Integration

Minimal integration that delegates to Isaac-GR00T components where possible
without porting their code. The intent is to:

- Download and load the pretrained GR00T model via GR00TN15.from_pretrained
- Optionally align action horizon similar to gr00t_finetune.py
- Expose predict_action via GR00T model.get_action
- Provide a training forward that can call the GR00T model forward if batch
  structure matches.

Notes:
- Dataset loading and full training orchestration is handled by Isaac-GR00T
  TrainRunner in their codebase. If you want to invoke that flow end-to-end
  from LeRobot, see `GrootPolicy.finetune_with_groot_runner` below.
"""

import os
from collections import deque

import torch
from torch import Tensor

import logging

from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.groot_n1 import GR00TN15
from lerobot.policies.pretrained import PreTrainedPolicy


class GrootPolicy(PreTrainedPolicy):
    """Wrapper around external Groot model for LeRobot integration."""

    name = "groot"
    config_class = GrootConfig

    def __init__(self, config: GrootConfig, **kwargs):
        """Initialize Groot policy wrapper."""
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize GR00T model using ported components
        self._groot_model = self._create_groot_model()

        self.reset()

    def _create_groot_model(self):
        """Create and initialize the GR00T model using Isaac-GR00T API.

        This is only called when creating a NEW policy (not when loading from checkpoint).

        Steps (delegating to Isaac-GR00T):
        1) Download and load pretrained model via GR00TN15.from_pretrained
        2) Align action horizon with data_config if provided
        """
        # Handle Flash Attention compatibility issues
        self._handle_flash_attention_compatibility()

        pretrained_model_name_or_path = self.config.base_model_path
        if self.config.pretrained_path is not None and os.path.exists(self.config.pretrained_path):
            pretrained_model_name_or_path = self.config.pretrained_path

        # Debug: Print config values to verify CLI args are being respected
        print(f"[GROOT DEBUG] _create_groot_model config values:")
        print(f"  tune_llm={self.config.tune_llm}")
        print(f"  tune_visual={self.config.tune_visual}")
        print(f"  tune_projector={self.config.tune_projector}")
        print(f"  tune_diffusion_model={self.config.tune_diffusion_model}")
        print(f"  use_depth={self.config.use_depth}")
        print(f"  dgcnn_num_points={self.config.dgcnn_num_points}")
        print(f"  dgcnn_k={self.config.dgcnn_k}")
        print(f"  dgcnn_num_tokens={self.config.dgcnn_num_tokens}")
        print(f"  dgcnn_camera_weights={self.config.dgcnn_camera_weights}")

        model = GR00TN15.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tune_llm=self.config.tune_llm,
            tune_visual=self.config.tune_visual,
            tune_projector=self.config.tune_projector,
            tune_diffusion_model=self.config.tune_diffusion_model,
            use_depth=self.config.use_depth,
            debug_dir=self.config.debug_dir,
            dgcnn_workspace_bounds=self.config.dgcnn_workspace_bounds,
            dgcnn_num_points=self.config.dgcnn_num_points,
            dgcnn_k=self.config.dgcnn_k,
            dgcnn_num_tokens=self.config.dgcnn_num_tokens,
            dgcnn_camera_weights=self.config.dgcnn_camera_weights,
        )

        model.compute_dtype = "bfloat16" if self.config.use_bf16 else model.compute_dtype
        model.config.compute_dtype = model.compute_dtype

        return model

    def reset(self):
        """Reset policy state when environment resets."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass.

        Delegates to Isaac-GR00T model.forward when inputs are compatible.
        """
        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id", "depth_raw"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_") or k.startswith("dgcnn_")) and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        # Run GR00T forward under bf16 autocast when enabled to reduce activation memory
        # Rationale: Matches original GR00T finetuning (bf16 compute, fp32 params) and avoids fp32 upcasts.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(groot_inputs)

        # Isaac-GR00T returns a BatchFeature; loss key is typically 'loss'
        loss = outputs.get("loss")

        loss_dict = {"loss": loss.item()}

        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions for inference by delegating to Isaac-GR00T.

        Returns a tensor of shape (B, n_action_steps, action_dim).
        """
        self.eval()

        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        # Preprocessing is handled by the processor pipeline, so we just filter the batch
        # NOTE: During inference, we should NOT pass action/action_mask (that's what we're predicting)
        allowed_base = {"state", "state_mask", "embodiment_id", "depth_raw"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_") or k.startswith("dgcnn_")) and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        # Use bf16 autocast for inference to keep memory low and match backbone dtype
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.get_action(groot_inputs)

        actions = outputs.get("action_pred")

        original_action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action from action queue."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def verify_architecture(self) -> None:
        """Verify GR00T model architecture matches config expectations for DGCNN.

        When ``use_depth=True``, checks:
          1. ``EagleBackbone.use_depth`` flag is set.
          2. ``EagleBackbone.point_cloud_encoder`` exists and is a ``DGCNNEncoder``.
          3. DGCNN hyperparams match config (num_points, k, num_tokens).
          4. DGCNN ``hidden_dim`` matches the Eagle projection dim (``eagle_linear``
             output) so that ``torch.cat([eagle_embeds, pc_tokens], dim=1)``
             works at forward time.
          5. DGCNN sub-modules (conv1-4, projection, norm) are present.
          6. Patch embedding stays 3-channel (RGB only; depth goes through DGCNN).

        When ``use_depth=False``, checks:
          1. ``point_cloud_encoder`` is ``None``.
          2. Patch embedding is 3-channel RGB.
        """
        from lerobot.policies.groot.dgcnn_encoder import DGCNNEncoder

        backbone = self._groot_model.backbone  # EagleBackbone
        cfg = self.config

        if cfg.use_depth:
            # --- 1. Flag ---
            assert getattr(backbone, "use_depth", False), (
                "[GROOT] verify_architecture FAILED: config.use_depth=True but "
                "EagleBackbone.use_depth is False."
            )

            # --- 2. Encoder exists and is the right type ---
            pce = getattr(backbone, "point_cloud_encoder", None)
            assert pce is not None, (
                "[GROOT] verify_architecture FAILED: config.use_depth=True but "
                "EagleBackbone.point_cloud_encoder is None."
            )
            assert isinstance(pce, DGCNNEncoder), (
                f"[GROOT] verify_architecture FAILED: point_cloud_encoder is "
                f"{type(pce).__name__}, expected DGCNNEncoder."
            )

            # --- 3. Hyperparams ---
            assert pce.num_points == cfg.dgcnn_num_points, (
                f"[GROOT] verify_architecture FAILED: DGCNN num_points={pce.num_points} "
                f"!= config.dgcnn_num_points={cfg.dgcnn_num_points}."
            )
            assert pce.k == cfg.dgcnn_k, (
                f"[GROOT] verify_architecture FAILED: DGCNN k={pce.k} "
                f"!= config.dgcnn_k={cfg.dgcnn_k}."
            )
            assert pce.num_tokens == cfg.dgcnn_num_tokens, (
                f"[GROOT] verify_architecture FAILED: DGCNN num_tokens={pce.num_tokens} "
                f"!= config.dgcnn_num_tokens={cfg.dgcnn_num_tokens}."
            )

            # --- 4. Hidden dim matches Eagle projection output ---
            eagle_linear = backbone.eagle_linear
            if hasattr(eagle_linear, "out_features"):
                eagle_out_dim = eagle_linear.out_features
            else:
                # nn.Identity — output size is the raw Eagle hidden size (2048)
                eagle_out_dim = 2048
            assert pce.hidden_dim == eagle_out_dim, (
                f"[GROOT] verify_architecture FAILED: DGCNN hidden_dim={pce.hidden_dim} "
                f"!= Eagle projection output dim={eagle_out_dim}. "
                f"Tokens cannot be concatenated."
            )

            # --- 5. Sub-modules ---
            for name in ("conv1", "conv2", "conv3", "conv4", "projection", "norm"):
                assert hasattr(pce, name), (
                    f"[GROOT] verify_architecture FAILED: DGCNNEncoder missing "
                    f"sub-module '{name}'."
                )

            # --- 6. Patch embedding stays 3-channel (RGB only) ---
            patch_key = (
                "_groot_model.backbone.eagle_model.vision_model."
                "vision_model.embeddings.patch_embedding.weight"
            )
            patch_weight = self.state_dict().get(patch_key)
            if patch_weight is not None:
                assert patch_weight.shape[1] == 3, (
                    f"[GROOT] verify_architecture FAILED: use_depth=True (DGCNN mode) "
                    f"but patch_embedding has {patch_weight.shape[1]} channels instead "
                    f"of 3. Depth should go through DGCNN, not the patch embedding."
                )

            n_params = sum(p.numel() for p in pce.parameters())
            logging.info(
                f"[GROOT] verify_architecture OK (use_depth=True): "
                f"DGCNNEncoder({pce.num_points} pts, k={pce.k}, "
                f"{pce.num_tokens} tokens, hidden={pce.hidden_dim}, "
                f"{n_params:,} params) → concat with Eagle ({eagle_out_dim}-d)"
            )
        else:
            # --- depth disabled: encoder must be absent ---
            pce = getattr(backbone, "point_cloud_encoder", None)
            assert pce is None, (
                "[GROOT] verify_architecture FAILED: config.use_depth=False but "
                f"EagleBackbone.point_cloud_encoder is {type(pce).__name__} (should be None)."
            )

            patch_key = (
                "_groot_model.backbone.eagle_model.vision_model."
                "vision_model.embeddings.patch_embedding.weight"
            )
            patch_weight = self.state_dict().get(patch_key)
            if patch_weight is not None:
                assert patch_weight.shape[1] == 3, (
                    f"[GROOT] verify_architecture FAILED: patch_embedding has "
                    f"{patch_weight.shape[1]} channels, expected 3 (RGB)."
                )

            logging.info(
                "[GROOT] verify_architecture OK (use_depth=False): "
                "no DGCNN encoder, patch_embedding=3ch RGB"
            )

    # -------------------------
    # Internal helpers
    # -------------------------
    def _handle_flash_attention_compatibility(self) -> None:
        """Handle Flash Attention compatibility issues by setting environment variables.

        This addresses the common 'undefined symbol' error that occurs when Flash Attention
        is compiled against a different PyTorch version than what's currently installed.
        """

        # Set environment variables to handle Flash Attention compatibility
        # These help with symbol resolution issues
        os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "0")

        # Try to import flash_attn and handle failures gracefully
        try:
            import flash_attn

            print(f"[GROOT] Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            print(f"[GROOT] Flash Attention not available: {e}")
            print("[GROOT] Will use fallback attention mechanism")
        except Exception as e:
            if "undefined symbol" in str(e):
                print(f"[GROOT] Flash Attention compatibility issue detected: {e}")
                print("[GROOT] This is likely due to PyTorch/Flash Attention version mismatch")
                print("[GROOT] Consider reinstalling Flash Attention with compatible version:")
                print("  pip uninstall flash-attn")
                print("  pip install --no-build-isolation flash-attn==2.6.3")
                print("[GROOT] Continuing with fallback attention mechanism")
            else:
                print(f"[GROOT] Flash Attention error: {e}")
                print("[GROOT] Continuing with fallback attention mechanism")
