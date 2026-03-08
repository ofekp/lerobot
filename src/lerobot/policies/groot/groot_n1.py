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
from lerobot.policies.groot.dgcnn_encoder import DGCNNEncoder
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
        dgcnn_workspace_bounds: tuple = ((-0.3, 0.3), (-0.3, 0.3), (0.6, 1.0)),
        dgcnn_num_points: int = 2048,
        dgcnn_k: int = 20,
        dgcnn_num_tokens: int = 64,
        dgcnn_camera_weights: str | None = None,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
            use_depth: whether to enable DGCNN point cloud encoder + depth rendering in eval

        config - https://huggingface.co/lerobot/eagle2hg-processor-groot-n1p5/blob/main/config.json
        from_pretrained - https://github.com/huggingface/transformers/blob/v5.0.0rc2/src/transformers/modeling_utils.py#L3656
        """
        print(f"[GROOT] Initializing EagleBackbone with use_depth={use_depth}")
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        # Prefer loading Eagle model config from the cache directory where vendor files were copied.
        vendor_dir = DEFAULT_VENDOR_EAGLE_PATH
        cache_dir = HF_LEROBOT_HOME / tokenizer_assets_repo
        try:
            ensure_eagle_cache_ready(vendor_dir, cache_dir, tokenizer_assets_repo)
        except Exception as exc:  # nosec: B110
            print(f"[GROOT] Warning: failed to prepare Eagle cache for backbone: {exc}")

        config = AutoConfig.from_pretrained(str(cache_dir), trust_remote_code=True)

        # Auto-detect attention backend: flash_attention_2 needs Ampere+ (SM ≥ 8.0)
        import torch as _torch
        _attn_impl = "flash_attention_2"
        if _torch.cuda.is_available():
            major, _ = _torch.cuda.get_device_capability()
            if major < 8:
                _attn_impl = "sdpa"
                print(f"[GROOT] GPU SM {major}.x < 8.0 — using sdpa attention (flash_attention_2 requires Ampere+)")
        config._attn_implementation = _attn_impl
        if hasattr(config, "text_config"):
            config.text_config._attn_implementation = _attn_impl
        if hasattr(config, "vision_config"):
            config.vision_config._attn_implementation = _attn_impl

        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # DGCNN point cloud encoder for 3D spatial tokens
        self.use_depth = use_depth
        self.point_cloud_encoder = None
        if self.use_depth:
            self.point_cloud_encoder = DGCNNEncoder(
                num_points=dgcnn_num_points,
                k=dgcnn_k,
                hidden_dim=project_to_dim if project_to_dim is not None else 2048,
                num_tokens=dgcnn_num_tokens,
                workspace_bounds=dgcnn_workspace_bounds,
                camera_weights=dgcnn_camera_weights,
            )
            print(f"[GROOT] DGCNNEncoder initialized: points={dgcnn_num_points}, k={dgcnn_k}, "
                  f"tokens={self.point_cloud_encoder.num_tokens}, "
                  f"params={sum(p.numel() for p in self.point_cloud_encoder.parameters()):,}")

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)

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

        # DDP compatibility hack for tune_visual
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        # DGCNN point cloud encoder: concat 3D spatial tokens with Eagle tokens
        if self.use_depth and self.point_cloud_encoder is not None:
            pc_depth = vl_input.get("dgcnn_depth")
            pc_intrinsics = vl_input.get("dgcnn_intrinsics")
            pc_extrinsics = vl_input.get("dgcnn_extrinsics")
            pc_camera_names = vl_input.get("dgcnn_camera_names")

            if all(v is not None for v in [pc_depth, pc_intrinsics, pc_extrinsics]):
                pc_tokens = self.point_cloud_encoder(
                    pc_depth, pc_intrinsics, pc_extrinsics,
                    camera_names=pc_camera_names,
                )
                pc_tokens = pc_tokens.to(dtype=eagle_embeds.dtype, device=eagle_embeds.device)

                # --- DGCNN verification assertions ---
                assert pc_tokens.abs().sum() > 0, (
                    "DGCNN tokens are all zeros — depth encoder is not producing useful features"
                )
                assert pc_tokens.shape[1:] == (self.point_cloud_encoder.num_tokens, eagle_embeds.shape[-1]), (
                    f"DGCNN token shape mismatch: got {pc_tokens.shape}, "
                    f"expected (B, {self.point_cloud_encoder.num_tokens}, {eagle_embeds.shape[-1]})"
                )

                n_eagle = eagle_embeds.shape[1]
                eagle_embeds = torch.cat([eagle_embeds, pc_tokens], dim=1)

                assert eagle_embeds.shape[1] == n_eagle + pc_tokens.shape[1], (
                    f"DGCNN tokens not aggregated: {eagle_embeds.shape[1]} != {n_eagle} + {pc_tokens.shape[1]}"
                )

                pc_mask = torch.ones(
                    pc_tokens.shape[0], pc_tokens.shape[1],
                    dtype=eagle_mask.dtype, device=eagle_mask.device,
                )
                eagle_mask = torch.cat([eagle_mask, pc_mask], dim=1)
            else:
                import warnings
                warnings.warn(
                    "use_depth=True but dgcnn_depth/intrinsics/extrinsics missing from input — "
                    "DGCNN is configured but NOT being used this forward pass",
                    RuntimeWarning, stacklevel=2,
                )

        return BatchFeature(
            data={"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}
        )


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
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=} (expected channels={N_COLOR_CHANNELS})"
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

        # DGCNN point cloud encoder settings
        use_depth = kwargs.pop("use_depth", False)
        dgcnn_workspace_bounds = kwargs.pop("dgcnn_workspace_bounds", ((-0.3, 0.3), (-0.3, 0.3), (0.6, 1.0)))
        dgcnn_num_points = kwargs.pop("dgcnn_num_points", 2048)
        dgcnn_k = kwargs.pop("dgcnn_k", 20)
        dgcnn_num_tokens = kwargs.pop("dgcnn_num_tokens", 64)
        dgcnn_camera_weights = kwargs.pop("dgcnn_camera_weights", None)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")
        print(f"Use depth (DGCNN): {use_depth}")

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
            except Exception as e:
                print(f"[GROOT] Warning: Could not inspect checkpoint: {e}")

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

            if use_depth and "backbone_cfg" in config_dict:
                config_dict["backbone_cfg"]["use_depth"] = use_depth
                config_dict["backbone_cfg"]["dgcnn_workspace_bounds"] = dgcnn_workspace_bounds
                config_dict["backbone_cfg"]["dgcnn_num_points"] = dgcnn_num_points
                config_dict["backbone_cfg"]["dgcnn_k"] = dgcnn_k
                config_dict["backbone_cfg"]["dgcnn_num_tokens"] = dgcnn_num_tokens
                config_dict["backbone_cfg"]["dgcnn_camera_weights"] = dgcnn_camera_weights

            config = cls.config_class(**config_dict)
            pretrained_model = cls(config, local_model_path=local_model_path)

        else:
            # TRAINING FLOW: Loading from HuggingFace (base model)
            print("[GROOT] Detected HuggingFace checkpoint format (training flow)")

            if use_depth:
                import json
                config_path = Path(local_model_path) / "config.json"
                with open(config_path, "r") as f:
                    config_dict = json.load(f)

                if "backbone_cfg" in config_dict:
                    config_dict["backbone_cfg"]["use_depth"] = use_depth
                    config_dict["backbone_cfg"]["dgcnn_workspace_bounds"] = dgcnn_workspace_bounds
                    config_dict["backbone_cfg"]["dgcnn_num_points"] = dgcnn_num_points
                    config_dict["backbone_cfg"]["dgcnn_k"] = dgcnn_k
                    config_dict["backbone_cfg"]["dgcnn_num_tokens"] = dgcnn_num_tokens
                    config_dict["backbone_cfg"]["dgcnn_camera_weights"] = dgcnn_camera_weights

                config = cls.config_class(**config_dict)
                kwargs["config"] = config

            pretrained_model = super().from_pretrained(
                local_model_path, local_model_path=local_model_path, **kwargs
            )

        pretrained_model.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model
