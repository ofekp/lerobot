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
        """
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

        # NOTE: Depth extension is NOT done here. Call extend_patch_embedding_for_depth()
        # explicitly after loading pretrained weights to extend from 3 to 4 channels.

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

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
            self._depth_grad_step_counter += 1
            if self._depth_grad_step_counter % log_every_n_steps == 0:
                depth_grad = grad[:, 3:4, :, :]
                rgb_grad = grad[:, :3, :, :]
                print(f"[Depth Grad Monitor @ step {self._depth_grad_step_counter}] "
                      f"Depth grad norm: {depth_grad.norm().item():.6f}, "
                      f"RGB grad norm: {rgb_grad.norm().item():.6f}, ",
                      flush=True)
            return grad
        self._depth_grad_hook_handle = patch_embed.weight.register_hook(hook)
        print(f"[GROOT] Depth gradient hook registered (logging every {log_every_n_steps} steps)")

    def extend_patch_embedding_for_depth(self):
        """Public method to extend patch embedding for depth after pretrained weights are loaded.
        
        This should be called after from_pretrained() loads the 3-channel weights.
        """
        if self._depth_extended:
            print("[GROOT] Patch embedding already extended for depth, skipping.")
            return
        if not self.use_depth:
            print("[GROOT] use_depth is False, skipping depth extension.")
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
        if self.use_depth:
            self.eagle_model.vision_model.vision_model.embeddings.patch_embedding.weight.requires_grad = True
            print(f"Tune backbone patch embedding (depth): True")
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

        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

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
        
        # Depth modality settings (following "Modality-Augmented Fine-Tuning" paper)
        use_depth = kwargs.pop("use_depth", False)
        depth_weight_init = kwargs.pop("depth_weight_init", "rgb_average")

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")
        print(f"Use depth (RGB-D): {use_depth}")
        if use_depth:
            print(f"Depth weight initialization: {depth_weight_init}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        # If using depth, we need to modify the config before model instantiation
        if use_depth:
            # Load the config, modify backbone_cfg, then pass it explicitly
            import json
            config_path = os.path.join(local_model_path, "config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            
            # Inject depth settings into backbone_cfg
            if "backbone_cfg" in config_dict:
                config_dict["backbone_cfg"]["use_depth"] = use_depth
                config_dict["backbone_cfg"]["depth_weight_init"] = depth_weight_init
            
            # Create config object from modified dict
            config = cls.config_class(**config_dict)
            kwargs["config"] = config

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        # Extend patch embedding for depth AFTER pretrained weights are loaded
        # This ensures we start from the pretrained 3-channel weights and extend to 4 channels
        if use_depth:
            pretrained_model.backbone.extend_patch_embedding_for_depth()
            pretrained_model.backbone.register_depth_gradient_hook(log_every_n_steps=1000)

        pretrained_model.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model
