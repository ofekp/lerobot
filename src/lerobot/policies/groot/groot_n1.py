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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from lerobot.utils.import_utils import _transformers_available
from lerobot.policies.groot.chnet_modules import CHNetDepthProcessor

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
        depth_weight_init: str = "rgb_average",
        _skip_depth_init: bool = False,
        chnet_tap_layers: tuple = (5, 11, 17, 23),
        chnet_channels: tuple = (64, 128, 256, 256),
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
            use_depth: whether to use RGB-D (4-channel) input instead of RGB (3-channel)
            depth_weight_init: method to initialize depth channel weights
                - "rgb_average": Initialize as average of RGB channels (recommended)
                - "zero": Initialize to zero
                - "random": Random initialization (default PyTorch init)

        config - https://huggingface.co/lerobot/eagle2hg-processor-groot-n1p5/blob/main/config.json
        from_pretrained - https://github.com/huggingface/transformers/blob/v5.0.0rc2/src/transformers/modeling_utils.py#L3656
        """
        print(f"[GROOT] Initializing EagleBackbone with use_depth={use_depth}, _skip_depth_init={_skip_depth_init}")
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        self.use_depth = use_depth
        self.depth_weight_init = depth_weight_init
        self._depth_extended = False

        # Prefer loading Eagle model config from the cache directory where vendor files were copied.
        vendor_dir = DEFAULT_VENDOR_EAGLE_PATH
        cache_dir = HF_LEROBOT_HOME / tokenizer_assets_repo
        try:
            ensure_eagle_cache_ready(vendor_dir, cache_dir, tokenizer_assets_repo)
        except Exception as exc:  # nosec: B110
            print(f"[GROOT] Warning: failed to prepare Eagle cache for backbone: {exc}")

        config = AutoConfig.from_pretrained(str(cache_dir), trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        # Depth extension strategy:
        # - Eval (loading 4-channel checkpoint): extend now so shapes match when loading weights
        # - Training (loading 3-channel HF checkpoint): defer extension until after weights are loaded,
        #   so we can initialize depth channel as rgb_average of pretrained RGB weights
        if self.use_depth and not _skip_depth_init:
            self._extend_patch_embedding_for_depth()
            self._depth_extended = True

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        hidden_dim = 2048  # eagle_linear input dim

        # CHNet depth processing (only when depth is enabled)
        self._chnet_tap_layers = tuple(chnet_tap_layers)
        self._vit_hook_features = {}
        self._vit_hooks = []
        if self.use_depth:
            # Derive vit_dim from actual model config instead of hardcoding
            vit_dim = getattr(config.vision_config, 'hidden_size', 1152)
            self.chnet = CHNetDepthProcessor(
                vit_dim=vit_dim,
                hidden_dim=hidden_dim,
                channels=tuple(chnet_channels),
            )
            self._register_vit_hooks()
            print(f"[GROOT] CHNet depth processor initialized (vit_dim={vit_dim}, hidden_dim={hidden_dim}, tapping ViT layers {chnet_tap_layers})")
        else:
            self.chnet = None
            print("[GROOT] CHNet disabled (use_depth=False)")

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)
        
        # For depth gradient monitoring
        self._depth_grad_step_counter = 0
        self._depth_grad_hook_handle = None

    def register_depth_gradient_hook(self, log_every_n_steps: int = 5000):
        """Register a hook to monitor depth channel gradients during training.
        Args:
            log_every_n_steps: Print gradient stats every N backward passes.
        """
        if not self.use_depth or not self._depth_extended:
            print("[GROOT] Cannot register depth gradient hook: depth not enabled or not extended yet.")
            return
        patch_embed = self.eagle_model.vision_model.vision_model.embeddings.patch_embedding
        def hook(grad):
            if self._depth_grad_step_counter % log_every_n_steps == 0:
                depth_grad = grad[:, 3:4, :, :]
                rgb_grad = grad[:, :3, :, :]
                # Also print current weight and bias values to track updates
                depth_weight = patch_embed.weight[:, 3:4, :, :]
                rgb_weight = patch_embed.weight[:, :3, :, :]
                bias_str = ""
                if patch_embed.bias is not None:
                    bias_str = f", bias norm: {patch_embed.bias.norm().item():.6f}, bias mean: {patch_embed.bias.mean().item():.6f}"
                print(f"[Depth Grad Monitor @ step {self._depth_grad_step_counter}] "
                      f"Depth grad norm: {depth_grad.norm().item():.6f}, "
                      f"RGB grad norm: {rgb_grad.norm().item():.6f}, "
                      f"Depth weight norm: {depth_weight.norm().item():.6f}, "
                      f"RGB weight norm: {rgb_weight.norm().item():.6f}"
                      f"{bias_str}",
                      flush=True)
            self._depth_grad_step_counter += 1
            return grad
        self._depth_grad_hook_handle = patch_embed.weight.register_hook(hook)
        print(f"[GROOT] Depth gradient hook registered (logging every {log_every_n_steps} steps)")

    def _register_vit_hooks(self):
        """Register forward hooks on ViT layers to capture intermediate features for CHNet."""
        try:
            vit_layers = self.eagle_model.vision_model.vision_model.encoder.layers
        except AttributeError:
            try:
                vit_layers = self.eagle_model.vision_model.encoder.layers
            except AttributeError:
                raise RuntimeError(
                    "[GROOT] Cannot find ViT encoder layers for CHNet hooks."
                )

        self._vit_hooks = []
        for layer_idx in self._chnet_tap_layers:
            if layer_idx >= len(vit_layers):
                raise ValueError(
                    f"[GROOT] ViT layer index {layer_idx} out of range "
                    f"(model has {len(vit_layers)} layers)"
                )
            handle = vit_layers[layer_idx].register_forward_hook(
                self._make_vit_hook(layer_idx)
            )
            self._vit_hooks.append(handle)
        print(f"[GROOT] Registered {len(self._vit_hooks)} ViT hooks for CHNet")

    def _make_vit_hook(self, layer_idx):
        """Create a hook closure for a specific ViT layer."""
        def hook(module, input, output):
            # detach() because Eagle ViT is frozen — no need to track ViT computation graph.
            # Gradients still flow through CHNet's projectors and FastGuide modules.
            if isinstance(output, tuple):
                self._vit_hook_features[layer_idx] = output[0].detach()
            else:
                self._vit_hook_features[layer_idx] = output.detach()
        return hook

    def extend_patch_embedding_for_depth(self):
        """Public method to extend patch embedding for depth after pretrained weights are loaded.
        
        This should be called after from_pretrained() loads the 3-channel weights.
        If the loaded weights already have 4 channels (i.e., checkpoint was trained with depth),
        this method will skip the extension to preserve the trained depth weights.
        """
        assert False, "For CHNET we do not use the 4-channel extension strategy such as in the modality-augmented paper"
        if self._depth_extended:
            print("[GROOT] Patch embedding already extended for depth, skipping.")
            return
        if not self.use_depth:
            print("[GROOT] use_depth is False, skipping depth extension.")
            return
        
        # Check if patch embedding already has 4 channels (loaded from depth-trained checkpoint)
        vision_model = self.eagle_model.vision_model
        patch_embed = vision_model.vision_model.embeddings.patch_embedding
        
        if isinstance(patch_embed, nn.Conv2d):
            current_in_channels = patch_embed.weight.shape[1]
        elif isinstance(patch_embed, nn.Linear):
            # For linear, assume patch_size can be inferred
            in_features = patch_embed.weight.shape[1]
            # Try 4 channels first
            patch_area_4ch = in_features // 4
            patch_size_4ch = int(patch_area_4ch ** 0.5)
            if patch_size_4ch * patch_size_4ch * 4 == in_features:
                current_in_channels = 4
            else:
                current_in_channels = 3  # Assume 3 if not 4
        else:
            current_in_channels = 3  # Default assumption
        
        if current_in_channels == 4:
            print("[GROOT] Patch embedding already has 4 channels (loaded from depth-trained checkpoint), skipping extension.")
            self._depth_extended = True
            return
        
        self._extend_patch_embedding_for_depth()
        self._depth_extended = True

    def _extend_patch_embedding_for_depth(self):
        """Extend the vision model's patch embedding from 3 to 4 channels.
        
        Following the paper "Modality-Augmented Fine-Tuning of Foundation Robot Policies":
        - Expand the patch embedding from 3 to 4 channels for RGB-D fusion
        - Initialize depth-channel weights using RGB kernel averaging:
          W_patch(D) = 1/3 * (W_patch(R) + W_patch(G) + W_patch(B))
        """
        assert False, "For CHNET we do not use the 4-channel extension strategy such as in the modality-augmented paper"
        vision_model = self.eagle_model.vision_model
        
        # Find the patch embedding layer - it's typically in the embeddings
        # patch_embed = None
        # patch_embed_attr = None
        
        # Try common paths for patch embedding in various ViT architectures
        # possible_paths = [   
        #     ('embeddings', 'patch_embedding'),
        #     ('embeddings', 'patch_embeddings', 'projection'),
        #     ('patch_embed', 'proj'),
        #     ('patch_embedding',),
        # ]
        
        patch_embed_attr = ["vision_model", "embeddings", "patch_embedding"]
        obj = vision_model
        for attr in patch_embed_attr:
            obj = getattr(obj, attr)
        patch_embed = obj

        if not (isinstance(obj, nn.Conv2d) or isinstance(obj, nn.Linear)):
            raise ValueError(f"Unexpected patch embedding type: {type(obj)}")
        
        print(f"[GROOT] Original patch embedding shape: {patch_embed.weight.shape}")
        
        if isinstance(patch_embed, nn.Conv2d):
            # Conv2d patch embedding: weight shape is (out_channels, in_channels, H, W)
            old_weight = patch_embed.weight.data  # (out, 3, H, W)
            out_channels, in_channels, kh, kw = old_weight.shape
            
            if in_channels != 3:
                print(f"[GROOT] Warning: Expected 3 input channels, got {in_channels}. Skipping depth extension.")
                return
            
            # Create new Conv2d with 4 input channels
            new_conv = nn.Conv2d(
                in_channels=4,
                out_channels=out_channels,
                kernel_size=(kh, kw),
                stride=patch_embed.stride,
                padding=patch_embed.padding,
                bias=patch_embed.bias is not None,
            )
            
            # Initialize: copy RGB weights, init depth channel
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_weight
                if self.depth_weight_init == "rgb_average":
                    # Average of RGB channels as recommended in paper
                    depth_init = old_weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:4, :, :] = depth_init
                elif self.depth_weight_init == "zero":
                    new_conv.weight[:, 3:4, :, :] = 0.0
                # "random" uses default PyTorch initialization (already done)
                if patch_embed.bias is not None:
                    new_conv.bias.copy_(patch_embed.bias)
            
            # Replace the old conv with the new one
            self._replace_module(vision_model, patch_embed_attr, new_conv)
            print(f"[GROOT] Extended patch embedding to 4 channels (RGB-D)")
            print(f"[GROOT] New patch embedding shape: {new_conv.weight.shape}")
            
        elif isinstance(patch_embed, nn.Linear):
            # Linear patch embedding: weight shape is (out_features, in_features)
            # in_features = patch_size * patch_size * 3
            old_weight = patch_embed.weight.data
            out_features, in_features = old_weight.shape
            
            # Assume patch_size is square root of in_features / 3
            patch_area = in_features // 3
            patch_size = int(patch_area ** 0.5)
            
            if patch_size * patch_size * 3 != in_features:
                print(f"[GROOT] Warning: Cannot determine patch size from in_features={in_features}. Skipping.")
                return
            
            # Create new Linear with 4 channels
            new_in_features = patch_size * patch_size * 4
            new_linear = nn.Linear(new_in_features, out_features, bias=patch_embed.bias is not None)
            
            with torch.no_grad():
                # Reshape old weights to (out, patch_size, patch_size, 3) for manipulation
                old_reshaped = old_weight.view(out_features, patch_size, patch_size, 3)
                
                if self.depth_weight_init == "rgb_average":
                    depth_init = old_reshaped.mean(dim=-1, keepdim=True)
                elif self.depth_weight_init == "zero":
                    depth_init = torch.zeros(out_features, patch_size, patch_size, 1, device=old_weight.device)
                else:  # random
                    depth_init = torch.randn(out_features, patch_size, patch_size, 1, device=old_weight.device) * 0.02
                
                # Concatenate RGB + D
                new_reshaped = torch.cat([old_reshaped, depth_init], dim=-1)
                new_linear.weight.copy_(new_reshaped.view(out_features, -1))
                
                if patch_embed.bias is not None:
                    new_linear.bias.copy_(patch_embed.bias)
            
            self._replace_module(vision_model, patch_embed_attr, new_linear)
            print(f"[GROOT] Extended patch embedding to 4 channels (RGB-D)")
            print(f"[GROOT] New patch embedding shape: {new_linear.weight.shape}")

    def _replace_module(self, parent: nn.Module, path: tuple, new_module: nn.Module):
        """Replace a nested module given its path."""
        obj = parent
        for attr in path[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, path[-1], new_module)

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
        # CHNet modules live on self.chnet (not under eagle_model), so they are
        # unaffected by the LLM/ViT freezing above. Explicitly ensure trainability.
        if self.chnet is not None:
            self.chnet.requires_grad_(True)
            print("[GROOT] CHNet modules set to trainable")
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")
    
    def _register_depth_only_gradient_hook(self, patch_embed: nn.Module):
        """Register a gradient hook that zeros out RGB channel gradients.
        
        Following the paper "Modality-Augmented Fine-Tuning of Foundation Robot Policies":
        - The vision tower (including RGB patch embedding weights) is frozen
        - Only the newly added depth channel weights are trained
        
        For Conv2d: weight shape is (out_channels, in_channels, H, W)
            - in_channels 0:3 are RGB (frozen)
            - in_channels 3:4 is depth (trainable)
        """
        def zero_rgb_gradients(grad):
            # Clone to avoid in-place modification issues
            new_grad = grad.clone()
            # Zero out gradients for RGB channels (indices 0, 1, 2)
            new_grad[:, :3, :, :] = 0.0
            return new_grad
        
        # Store the hook handle so we can remove it if needed
        if hasattr(self, '_depth_grad_hook'):
            self._depth_grad_hook.remove()
        self._depth_grad_hook = patch_embed.weight.register_hook(zero_rgb_gradients)
        print(f"[GROOT] Registered gradient hook to freeze RGB channels (0:3), train only depth channel (3)")

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

        # Extract depth for CHNet before removing non-Eagle keys
        depth_normalized = eagle_input.pop("depth_normalized", None)

        del eagle_input["image_sizes"]

        # Verify depth is flowing through when expected
        if not hasattr(self, '_depth_fwd_count'):
            self._depth_fwd_count = 0
        if self.use_depth:
            if depth_normalized is None:
                raise RuntimeError(
                    f"[GROOT] use_depth=True but depth_normalized is None! "
                    f"Depth data is not reaching the model. Check processor/dataset. "
                    f"(forward call #{self._depth_fwd_count})"
                )
            self._depth_fwd_count += 1
            if self._depth_fwd_count <= 3 or self._depth_fwd_count % 5000 == 0:
                mode = "TRAIN" if self.training else "EVAL"
                print(f"[GROOT DEPTH OK] [{mode}] fwd #{self._depth_fwd_count}: "
                      f"depth shape={depth_normalized.shape}, "
                      f"range=[{depth_normalized.min().item():.4f}, {depth_normalized.max().item():.4f}]",
                      flush=True)

        # Clear hook features before forward
        self._vit_hook_features.clear()

        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        # CHNet depth processing
        if depth_normalized is not None:
            # Collect ViT intermediate features captured by hooks
            vit_feats = [self._vit_hook_features[idx] for idx in self._chnet_tap_layers]

            # Determine ViT spatial grid from pixel_values
            pixel_values = eagle_input.get("pixel_values")
            if pixel_values is not None:
                _, _, h_pv, w_pv = pixel_values.shape
                patch_size = 14  # SigLIP patch size
                grid_h = h_pv // patch_size
                grid_w = w_pv // patch_size
            else:
                grid_h = grid_w = 16  # fallback for 224x224

            # Resize depth to match expected input size (224x224)
            if depth_normalized.shape[-1] != 224 or depth_normalized.shape[-2] != 224:
                depth_normalized = F.interpolate(
                    depth_normalized, size=(224, 224), mode='nearest'
                )

            eagle_features_before = eagle_features
            eagle_features = self.chnet(
                depth=depth_normalized,
                vit_features=vit_feats,
                grid_h=grid_h,
                grid_w=grid_w,
                eagle_features=eagle_features,
            )

            # Verify CHNet actually modified the features meaningfully
            diff = (eagle_features - eagle_features_before).abs()
            diff_norm = diff.norm().item()
            all_zero = eagle_features.abs().max().item() == 0.0
            unchanged = diff_norm == 0.0

            if all_zero:
                raise RuntimeError(
                    "[GROOT] CHNet output is all zeros! Cross-attention fusion produced empty features."
                )
            if unchanged:
                raise RuntimeError(
                    "[GROOT] CHNet did not change eagle_features at all! "
                    "Cross-attention residual is zero — depth signal is not being fused."
                )

            if self._depth_fwd_count <= 3 or self._depth_fwd_count % 5000 == 0:
                mode = "TRAIN" if self.training else "EVAL"
                ratio = diff_norm / eagle_features_before.norm().item()
                print(f"[GROOT CHNet VERIFY] [{mode}] fwd #{self._depth_fwd_count}: "
                      f"diff_norm={diff_norm:.4f}, "
                      f"before_norm={eagle_features_before.norm().item():.4f}, "
                      f"after_norm={eagle_features.norm().item():.4f}, "
                      f"change_ratio={ratio:.6f}",
                      flush=True)

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
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
        # CHNet keeps 3-channel ViT; legacy depth uses 4-channel
        self.expected_color_channels = 3 if self.use_depth else N_COLOR_CHANNELS

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
        
        # Depth modality settings (following "Modality-Augmented Fine-Tuning" paper)
        use_depth = kwargs.pop("use_depth", False)
        depth_weight_init = kwargs.pop("depth_weight_init", "rgb_average")

        # CHNet settings (alternative depth processing with FastGuide)
        chnet_tap_layers = kwargs.pop("chnet_tap_layers", (5, 11, 17, 23))
        chnet_channels = kwargs.pop("chnet_channels", (64, 128, 256, 256))

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")
        print(f"Use depth (RGB-D): {use_depth}")
        if use_depth:
            print(f"Depth weight initialization: {depth_weight_init}")

        # Download model or use local path
        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(f"Model not found on HuggingFace hub. Loading from local path: {pretrained_model_name_or_path}")
            local_model_path = pretrained_model_name_or_path

        # Detect if this is a LeRobot checkpoint (for eval) or HuggingFace model (for training)
        # LeRobot checkpoints have keys prefixed with "_groot_model."
        # Also check if checkpoint has 4-channel patch embedding (depth-trained)
        safetensor_path = os.path.join(local_model_path, "model.safetensors")
        is_lerobot_checkpoint = False
        checkpoint_has_4_channels = False
        
        patch_embed_key = "_groot_model.backbone.eagle_model.vision_model.vision_model.embeddings.patch_embedding.weight"
        
        def _print_patch_embed_weights(weights, label, use_depth_flag):
            """Print first few weights of each channel for debugging."""
            # weights shape: (out_channels, in_channels, H, W) for Conv2d
            print(f"\n[GROOT PATCH EMBED DEBUG] {label}")
            print(f"  Shape: {weights.shape}")
            # Flatten spatial dims and take first 5 values per channel
            flat = weights[:, :, :, :].reshape(weights.shape[0], weights.shape[1], -1)
            # Take first output channel, first few spatial values
            for ch, name in enumerate(['R', 'G', 'B']):
                if ch < weights.shape[1]:
                    vals = flat[0, ch, :5].tolist()
                    print(f"  Channel {name}: {[f'{v:.6f}' for v in vals]}")
            if use_depth_flag and weights.shape[1] >= 4:
                vals = flat[0, 3, :5].tolist()
                print(f"  Channel D: {[f'{v:.6f}' for v in vals]}")
            print("")
        
        if os.path.exists(safetensor_path):
            try:
                from safetensors import safe_open
                with safe_open(safetensor_path, framework="pt", device="cpu") as f:
                    all_keys = list(f.keys())
                    is_lerobot_checkpoint = any(k.startswith("_groot_model.") for k in all_keys[:10])
                    
                    if patch_embed_key in all_keys:
                        weights = f.get_tensor(patch_embed_key)
                        shape = weights.shape
                        checkpoint_has_4_channels = (shape[1] == 4)
                        print(f"[GROOT] Checkpoint patch embedding: {shape} ({'4-ch depth' if checkpoint_has_4_channels else '3-ch RGB'})")
                        _print_patch_embed_weights(weights, "Stage 1: From safetensor file", use_depth)
            except Exception as e:
                print(f"[GROOT] Warning: Could not inspect checkpoint: {e}")
        
        if is_lerobot_checkpoint:
            # EVAL FLOW: Loading a LeRobot-saved checkpoint
            # We just need to create the model with correct architecture.
            # Weights will be loaded by PreTrainedPolicy.from_pretrained via load_model_as_safetensor.
            print("[GROOT] Detected LeRobot checkpoint format (eval flow)")
            
            # LeRobot's config.json doesn't have backbone_cfg, so load from base model
            import json
            base_model_name = "nvidia/GR00T-N1.5-3B"
            try:
                base_model_path = snapshot_download(base_model_name, repo_type="model")
            except (HFValidationError, RepositoryNotFoundError):
                raise RuntimeError(f"Cannot download base GROOT model config from {base_model_name}")
            
            config_path = os.path.join(base_model_path, "config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            
            # Inject depth settings into backbone_cfg so EagleBackbone
            # is constructed with the correct architecture for the checkpoint.
            if use_depth and "backbone_cfg" in config_dict:
                config_dict["backbone_cfg"]["use_depth"] = True
                config_dict["backbone_cfg"]["depth_weight_init"] = depth_weight_init
                config_dict["backbone_cfg"]["_skip_depth_init"] = not checkpoint_has_4_channels
                config_dict["backbone_cfg"]["chnet_tap_layers"] = chnet_tap_layers
                config_dict["backbone_cfg"]["chnet_channels"] = chnet_channels
            
            config = cls.config_class(**config_dict)
            
            # Create model with correct architecture (4-ch if depth)
            # Don't load weights here - load_model_as_safetensor will handle it
            pretrained_model = cls(config, local_model_path=local_model_path)
            
            # Print weights after model creation (before external loading by PreTrainedPolicy)
            patch_embed = pretrained_model.backbone.eagle_model.vision_model.vision_model.embeddings.patch_embedding
            _print_patch_embed_weights(patch_embed.weight.data, "Stage 2: Model created with random init (checkpoint will be loaded next by PreTrainedPolicy)", use_depth)
            print("[GROOT] Note: Stage 2 shows random weights - this is expected. Stage 3 will show correct weights after checkpoint loading.")
            
        else:
            # TRAINING FLOW: Loading from HuggingFace (base model)
            # Need to load 3-ch weights first, then extend to 4-ch with rgb_average
            print("[GROOT] Detected HuggingFace checkpoint format (training flow)")
            
            if use_depth:
                import json
                config_path = Path(local_model_path) / "config.json"
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                
                if "backbone_cfg" in config_dict:
                    config_dict["backbone_cfg"]["use_depth"] = use_depth
                    config_dict["backbone_cfg"]["depth_weight_init"] = depth_weight_init
                    config_dict["backbone_cfg"]["_skip_depth_init"] = True  # Don't extend in __init__
                    config_dict["backbone_cfg"]["chnet_tap_layers"] = chnet_tap_layers
                    config_dict["backbone_cfg"]["chnet_channels"] = chnet_channels
                
                config = cls.config_class(**config_dict)
                kwargs["config"] = config
            
            # Use HuggingFace's from_pretrained to load base model weights
            pretrained_model = super().from_pretrained(
                local_model_path, local_model_path=local_model_path, **kwargs
            )
            
            # Print weights after HuggingFace loading (Stage 2 & 3 combined - HF creates and loads in one step)
            patch_embed = pretrained_model.backbone.eagle_model.vision_model.vision_model.embeddings.patch_embedding
            _print_patch_embed_weights(patch_embed.weight.data, "Stage 2+3: After HuggingFace from_pretrained (weights loaded)", use_depth)
            
            # Now extend to 4 channels using rgb_average of the loaded pretrained weights

        pretrained_model.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model
