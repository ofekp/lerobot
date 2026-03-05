# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
    from transformers.feature_extraction_utils import BatchFeature
else:
    AutoConfig = None
    AutoModel = None
    PretrainedConfig = object
    PreTrainedModel = object
    BatchFeature = None

try:
    import tree
except ImportError:
    tree = None

from lerobot.policies.groot.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from lerobot.policies.groot.depth_encoder import DepthBranchEncoder
from lerobot.policies.groot.utils import ensure_eagle_cache_ready
from lerobot.utils.constants import HF_LEROBOT_HOME

DEFAULT_VENDOR_EAGLE_PATH = str((Path(__file__).resolve().parent / "eagle2_hg_model").resolve())
DEFAULT_TOKENIZER_ASSETS_REPO = "lerobot/eagle2hg-processor-groot-n1p5"


class EagleBackbone(nn.Module):
    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str = DEFAULT_VENDOR_EAGLE_PATH,
        tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO,
        project_to_dim: int = 1536,
        use_depth: bool = False,
        depth_fourier_dim: int = 64,
        depth_hidden_dim: int = 256,
        depth_min_freq: float = 1.0,
        depth_max_freq: float = 50.0,
        depth_learnable_freqs: bool = False,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
            use_depth: whether to use a separate depth branch (Fourier + CNN + gate + zero-init depth_proj)
            depth_fourier_dim: Fourier positional encoding dimension for depth (default 64).
            depth_hidden_dim: Hidden channels in the depth CNN branch (default 256).
            depth_min_freq: Minimum Fourier frequency (default 1.0).
            depth_max_freq: Maximum Fourier frequency (default 50.0).
            depth_learnable_freqs: Whether Fourier frequencies are learnable (default False).

        config - https://huggingface.co/lerobot/eagle2hg-processor-groot-n1p5/blob/main/config.json
        from_pretrained - https://github.com/huggingface/transformers/blob/v5.0.0rc2/src/transformers/modeling_utils.py#L3656
        """
        print(f"[GROOT] Initializing EagleBackbone with use_depth={use_depth}")
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        self.use_depth = use_depth

        # Prefer loading Eagle model config from the cache directory where vendor files were copied.
        vendor_dir = DEFAULT_VENDOR_EAGLE_PATH
        cache_dir = HF_LEROBOT_HOME / tokenizer_assets_repo
        try:
            ensure_eagle_cache_ready(vendor_dir, cache_dir, tokenizer_assets_repo)
        except Exception as exc:  # nosec: B110
            print(f"[GROOT] Warning: failed to prepare Eagle cache for backbone: {exc}")

        config = AutoConfig.from_pretrained(str(cache_dir), trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)
        
        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # Backbone output dim (after eagle_linear projection or Identity)
        self._backbone_output_dim = project_to_dim if project_to_dim is not None else 2048

        # --- Depth branch: separate encoder → project → concatenate with eagle tokens ---
        # Depth tokens are produced independently and concatenated along the sequence
        # dimension with Eagle VL tokens, giving the action head access to both
        # RGB/language features and depth features through cross-attention.
        self.depth_branch = None
        self.depth_proj = None
        if self.use_depth:
            vision_cfg = config.vision_config
            embed_dim = vision_cfg.hidden_size  # 1152
            assert embed_dim == 1152, f"Expected embed_dim of 1152 from the Eagle vision config, got {embed_dim}"
            patch_size = vision_cfg.patch_size  # 14
            assert patch_size == 14, f"Expected patch_size of 14 from the Eagle vision config, got {patch_size}"

            self.depth_branch = DepthBranchEncoder(
                fourier_dim=depth_fourier_dim,
                embed_dim=embed_dim,
                patch_size=patch_size,
                hidden_dim=depth_hidden_dim,
                min_freq=depth_min_freq,
                max_freq=depth_max_freq,
                learnable_freqs=depth_learnable_freqs,
            )
            # Project depth tokens from vision embed_dim to backbone output dim
            self.depth_proj = nn.Linear(embed_dim, self._backbone_output_dim)
            print(f"[GROOT] Depth branch created: fourier_dim={depth_fourier_dim}, "
                  f"hidden_dim={depth_hidden_dim}, embed_dim={embed_dim} → {self._backbone_output_dim}, "
                  f"patch_size={patch_size}")
            print(f"[GROOT] Depth tokens will be concatenated with Eagle tokens for cross-attention")

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)

        # For depth gradient monitoring
        self._depth_grad_step_counter = 0

    def register_depth_gradient_hook(self, log_every_n_steps: int = 5000):
        """Register a hook to monitor depth branch gradients during training."""
        if not self.use_depth or self.depth_branch is None:
            print("[GROOT] Cannot register depth gradient hook: depth branch not available.")
            return

        # Monitor the gate weights (informative for learning progress)
        gate = self.depth_branch.patch_embedding.gate

        def hook(grad):
            if self._depth_grad_step_counter % log_every_n_steps == 0:
                # Gate weight gradient and value
                gate_weight_norm = gate.weight.data.norm().item()
                gate_grad_norm = grad.norm().item()
                # depth_proj weight and gradient
                dp_weight_norm = self.depth_proj.weight.data.norm().item()
                dp_bias_norm = self.depth_proj.bias.data.norm().item()
                dp_weight_grad = self.depth_proj.weight.grad.norm().item() if self.depth_proj.weight.grad is not None else 0.0
                dp_bias_grad = self.depth_proj.bias.grad.norm().item() if self.depth_proj.bias.grad is not None else 0.0
                # CNN weights (first layer)
                cnn_first = self.depth_branch.patch_embedding.cnn[0]
                cnn_weight_norm = cnn_first.weight.data.norm().item()
                # Fourier freqs
                freq_vals = self.depth_branch.fourier_encoding.freqs
                print(
                    f"[Depth Branch Monitor @ step {self._depth_grad_step_counter}] "
                    f"Gate weight norm: {gate_weight_norm:.6f}, "
                    f"Gate grad norm: {gate_grad_norm:.6f}, "
                    f"depth_proj weight norm: {dp_weight_norm:.6f}, "
                    f"depth_proj bias norm: {dp_bias_norm:.6f}, "
                    f"depth_proj weight grad: {dp_weight_grad:.6f}, "
                    f"depth_proj bias grad: {dp_bias_grad:.6f}, "
                    f"CNN[0] weight norm: {cnn_weight_norm:.6f}, "
                    f"Fourier freq range: [{freq_vals.min().item():.2f}, {freq_vals.max().item():.2f}]",
                    flush=True,
                )
            self._depth_grad_step_counter += 1
            return grad

        gate.weight.register_hook(hook)
        print(f"[GROOT] Depth branch gradient hook registered (logging every {log_every_n_steps} steps)")

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        if self.use_depth and self.depth_branch is not None:
            # The depth branch + projection are always trainable (randomly initialized)
            self.depth_branch.requires_grad_(True)
            if self.depth_proj is not None:
                self.depth_proj.requires_grad_(True)
            n_depth_params = sum(p.numel() for p in self.depth_branch.parameters())
            n_trainable_depth = sum(p.numel() for p in self.depth_branch.parameters() if p.requires_grad)
            if self.depth_proj is not None:
                n_depth_params += sum(p.numel() for p in self.depth_proj.parameters())
                n_trainable_depth += sum(p.numel() for p in self.depth_proj.parameters() if p.requires_grad)
            print(f"[GROOT] Depth branch + projection: {n_depth_params} params ({n_trainable_depth} trainable)")
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v for k, v in vl_input.items() if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]

        # If depth is enabled, split RGBD pixel_values → RGB for Eagle + depth for branch
        depth_tokens = None
        if self.use_depth and self.depth_branch is not None and "pixel_values" in eagle_input:
            pixel_values = eagle_input["pixel_values"]  # (N, 4, H, W) where N = B*T*V
            eagle_input["pixel_values"] = pixel_values[:, :3, :, :]  # RGB only for Eagle
            depth_input = pixel_values[:, 3:4, :, :]  # (N, 1, H, W)

            # Depth branch: Fourier encode → CNN → gate → (N, n_patches, 1152)
            depth_tokens = self.depth_branch(depth_input)
            # Project to backbone output dim: (N, n_patches, backbone_dim)
            depth_tokens = self.depth_proj(depth_tokens)

        # Run Eagle on RGB-only input
        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]
        eagle_features = self.eagle_linear(eagle_features)

        attn_mask = eagle_input["attention_mask"]

        # Concatenate depth tokens with Eagle VL tokens along sequence dimension
        if depth_tokens is not None:
            B = eagle_features.shape[0]
            N = depth_tokens.shape[0]
            depth_seq = depth_tokens.shape[1]  # n_patches (e.g. 256 for 224×224)
            views_per_sample = N // B  # T * V (typically 1)

            depth_tokens = depth_tokens.to(dtype=eagle_features.dtype)
            depth_tokens = depth_tokens.reshape(B, views_per_sample * depth_seq, -1)

            # backbone_features = concat(eagle_tokens, depth_tokens)
            eagle_features = torch.cat([eagle_features, depth_tokens], dim=1)

            # Extend attention mask for depth tokens (all ones = attend to all)
            depth_mask = torch.ones(
                B, views_per_sample * depth_seq,
                dtype=attn_mask.dtype, device=attn_mask.device,
            )
            attn_mask = torch.cat([attn_mask, depth_mask], dim=1)

        return eagle_features, attn_mask

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and (self.tune_visual or self.use_depth):
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            # Also include depth branch + projection parameters for DDP compatibility
            if self.depth_branch is not None:
                for param in self.depth_branch.parameters():
                    if param.requires_grad:
                        dummy_term = dummy_term + 0.0 * param.sum()
            if self.depth_proj is not None:
                for param in self.depth_proj.parameters():
                    if param.requires_grad:
                        dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        return BatchFeature(
            data={"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}
        )  # [B, T2, hidden_size]


BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00TN15Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00TN15(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00TN15Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00TN15Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype
        
        # Track if depth is enabled (for validation)
        self.use_depth = config.backbone_cfg.get("use_depth", False)
        self.expected_color_channels = 4 if self.use_depth else N_COLOR_CHANNELS

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            # In inference, action may be omitted or None; validate only when it's a tensor.
            if action is None:
                pass  # allow None during inference
            elif isinstance(action, torch.Tensor):
                shape_ok = (
                    len(action.shape) == 3
                    and action.shape[1] == self.action_horizon
                    and action.shape[2] == self.action_dim
                )
                if not shape_ok:
                    error_msg += f"\n{action.shape=}"
                    detected_error = True
            else:
                # Unexpected non-tensor type provided for action
                error_msg += f"\nInvalid type for action: {type(action)}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            # Allow 3 channels (RGB) or 4 channels (RGB-D) depending on config
            shape_ok = len(video.shape) == 6 and video.shape[3] == self.expected_color_channels
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=} (expected channels={self.expected_color_channels})"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature) or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Cast floating tensors to a memory-efficient compute dtype when requested.
            # Rationale: Upcasting backbone activations to fp32 significantly increases VRAM.
            # When compute_dtype is bfloat16, prefer bf16 for activations to match AMP behavior.
            if not isinstance(x, torch.Tensor):
                return x
            if torch.is_floating_point(x):
                if getattr(self, "compute_dtype", None) == "bfloat16":
                    return x.to(self.device, dtype=torch.bfloat16)
                # Fallback: preserve previous behavior if not using bf16 compute
                return x.to(self.device, dtype=self.action_head.dtype)
            # Non-floating tensors: move device only
            return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        
        # Depth branch settings
        use_depth = kwargs.pop("use_depth", False)
        depth_fourier_dim = kwargs.pop("depth_fourier_dim", 64)
        depth_hidden_dim = kwargs.pop("depth_hidden_dim", 256)
        depth_min_freq = kwargs.pop("depth_min_freq", 1.0)
        depth_max_freq = kwargs.pop("depth_max_freq", 50.0)
        depth_learnable_freqs = kwargs.pop("depth_learnable_freqs", False)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")
        print(f"Use depth branch: {use_depth}")
        if use_depth:
            print(f"Depth branch config: fourier_dim={depth_fourier_dim}, hidden_dim={depth_hidden_dim}, "
                  f"freq_range=[{depth_min_freq}, {depth_max_freq}], learnable_freqs={depth_learnable_freqs}")

        # Download model or use local path
        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(f"Model not found on HuggingFace hub. Loading from local path: {pretrained_model_name_or_path}")
            local_model_path = pretrained_model_name_or_path

        # Detect if this is a LeRobot checkpoint (for eval) or HuggingFace model (for training)
        safetensor_path = os.path.join(local_model_path, "model.safetensors")
        is_lerobot_checkpoint = False

        if os.path.exists(safetensor_path):
            try:
                from safetensors import safe_open
                with safe_open(safetensor_path, framework="pt", device="cpu") as f:
                    all_keys = list(f.keys())
                    is_lerobot_checkpoint = any(k.startswith("_groot_model.") for k in all_keys[:10])
                    patch_embed_keys = [
                        "_groot_model.backbone.eagle_model.vision_model.vision_model.embeddings.patch_embedding.weight",
                        "_groot_model.backbone.depth_branch.patch_embedding.cnn.0.weight",
                        "_groot_model.backbone.depth_branch.patch_embedding.cnn.0.bias",
                        "_groot_model.backbone.depth_branch.patch_embedding.cnn.3.weight",
                        "_groot_model.backbone.depth_branch.patch_embedding.cnn.3.bias",
                        "_groot_model.backbone.depth_branch.patch_embedding.gate.weight",
                        "_groot_model.backbone.depth_branch.patch_embedding.gate.bias",
                    ]
                    for patch_embed_key in patch_embed_keys:
                        if patch_embed_key in all_keys:
                            weights = f.get_tensor(patch_embed_key)
                            shape = weights.shape
                            print(f"Patch embed key: {patch_embed_key}: shape={shape}")
                            print(f"  Sample weights start: {weights.flatten()[0:5]}")
                            print(f"  Sample weights end: {weights.flatten()[-5:]}")
            except Exception as e:
                print(f"[GROOT] Warning: Could not inspect checkpoint: {e}")

        # Depth branch config to inject into backbone_cfg
        depth_cfg_kwargs = {
            "use_depth": use_depth,
            "depth_fourier_dim": depth_fourier_dim,
            "depth_hidden_dim": depth_hidden_dim,
            "depth_min_freq": depth_min_freq,
            "depth_max_freq": depth_max_freq,
            "depth_learnable_freqs": depth_learnable_freqs,
        }

        if is_lerobot_checkpoint:
            # EVAL FLOW: Loading a LeRobot-saved checkpoint
            print("[GROOT] Detected LeRobot checkpoint format (eval flow)")

            import json
            base_model_name = "nvidia/GR00T-N1.5-3B"
            try:
                base_model_path = snapshot_download(base_model_name, repo_type="model")
            except (HFValidationError, RepositoryNotFoundError):
                raise RuntimeError(f"Cannot download base GROOT model config from {base_model_name}")

            config_path = os.path.join(base_model_path, "config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            if "backbone_cfg" in config_dict:
                config_dict["backbone_cfg"].update(depth_cfg_kwargs)

            config = cls.config_class(**config_dict)
            pretrained_model = cls(config, local_model_path=local_model_path)
            # Weights will be loaded by PreTrainedPolicy.from_pretrained via load_model_as_safetensor

        else:
            # TRAINING FLOW: Loading from HuggingFace (base model)
            print("[GROOT] Detected HuggingFace checkpoint format (training flow)")

            if use_depth:
                import json
                config_path = Path(local_model_path) / "config.json"
                with open(config_path, "r") as f:
                    config_dict = json.load(f)

                if "backbone_cfg" in config_dict:
                    config_dict["backbone_cfg"].update(depth_cfg_kwargs)

                config = cls.config_class(**config_dict)
                kwargs["config"] = config

            # Use HuggingFace's from_pretrained to load base model weights
            # The RGB patch embedding stays 3-channel (no modification needed)
            pretrained_model = super().from_pretrained(
                local_model_path, local_model_path=local_model_path, **kwargs
            )

        if use_depth:
            # Explicitly initialize ALL depth branch weights AFTER model construction.
            # HuggingFace's from_pretrained wraps construction in no_init_weights()
            # which replaces all torch.nn.init.* with no-ops, leaving new (non-pretrained)
            # layers as uninitialized torch.empty() garbage — non-deterministic across runs.
            # reset_parameters() uses direct tensor ops, so it works correctly and
            # produces reproducible weights when a seed is set beforehand.
            # For eval flow (LeRobot checkpoint), these weights will be overwritten
            # when the checkpoint is loaded later.
            pretrained_model.backbone.depth_branch.reset_parameters()
            # Zero-init depth_proj (both weight AND bias).
            # The gate has Kaiming init → gate_out ≠ 0 during forward pass.
            # depth_proj(gate_out) = 0·gate_out + 0 = 0 → safe init (no perturbation).
            # Gradient for depth_proj.weight: dL/dW = dL/dy · gate_out^T ≠ 0
            #   (because gate_out ≠ 0) → depth_proj escapes zero on step 1.
            # After step 1, depth_proj.W ≠ 0, so gradients flow back through
            #   it to the gate and CNN: dL/d(gate_out) = W^T · dL/dy ≠ 0.
            pretrained_model.backbone.depth_proj.weight.data.zero_()
            pretrained_model.backbone.depth_proj.bias.data.zero_()
            print(f"[GROOT] depth_proj initialized (weight norm: "
                  f"{pretrained_model.backbone.depth_proj.weight.data.norm().item():.4f}, "
                  f"bias norm: {pretrained_model.backbone.depth_proj.bias.data.norm().item()})")

            pretrained_model.backbone.register_depth_gradient_hook(log_every_n_steps=1000)

        pretrained_model.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model
