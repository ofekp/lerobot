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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, ProcessorMixin
else:
    AutoProcessor = None
    ProcessorMixin = object

from lerobot.configs.types import (
    FeatureType,
    NormalizationMode,
    PolicyFeature,
)
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    HF_LEROBOT_HOME,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

# Defaults for Eagle processor locations
DEFAULT_TOKENIZER_ASSETS_REPO = "lerobot/eagle2hg-processor-groot-n1p5"


def make_groot_pre_post_processors(
    config: GrootConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create preprocessor and postprocessor for Groot policy.

    This creates a processing pipeline that transforms LeRobot data format into
    the format expected by Isaac-GR00T models:

    Preprocessing steps:
    1. Optional key renaming (dataset-specific key mapping)
    2. Add batch dimension to unbatched data
    3. Pack video/state/action/language/embodiment and apply optional min-max normalization before padding
    4. Encode video+language with Eagle VLM into intermediate eagle_content
    5. Collate eagle_content into batched eagle_* tensors
    6. Move tensors to device (GPU)

    NOTE: We optionally apply min-max normalization to STATE and ACTION using
    dataset-provided statistics prior to padding, mapping values to [-1, 1].
    This mirrors SO100-style preprocessing and keeps scales consistent with GR00T.

    Args:
        config: Groot configuration containing data_config, embodiment_tag, etc.
        dataset_stats: Optional per-key min/max statistics for normalization before padding.

    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """

    # Get horizon/dimension parameters from config
    # These should match the config used for the pretrained model
    # Default values match most GR00T configs (state_horizon=1, action_horizon=16)
    state_horizon = 1
    # CRITICAL: Pretrained GR00T models use action_horizon=16 max!
    # The model architecture hardcodes this limit
    action_horizon = min(config.chunk_size, 16)
    max_state_dim = config.max_state_dim
    max_action_dim = config.max_action_dim

    # Pass raw dataset_stats; normalization will occur inside pack step before padding
    padded_stats = dataset_stats or {}

    # Define feature specs for optional normalization steps
    _features: dict[str, PolicyFeature] = {
        # Observation features (only add those we may normalize)
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_horizon, max_state_dim)),
        # Action feature
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_horizon, max_action_dim)),
    }

    # Normalize STATE and ACTION with min_max (SO100-like default)
    _norm_map = {
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
        FeatureType.STATE: NormalizationMode.MIN_MAX,
    }

    # Determine env action dimension from config (simple, object-like PolicyFeature)
    try:
        env_action_dim = int(config.output_features["action"].shape[0])
    except Exception:
        env_action_dim = 0

    input_steps: list[ProcessorStep] = [
        # 1. Rename keys if needed (e.g., dataset-specific camera names)
        # Leave empty for now - add mappings if your dataset uses different key names
        RenameObservationsProcessorStep(rename_map={}),
        # 2. Add batch dimension for single samples
        AddBatchDimensionProcessorStep(),
        # 3. Pack video/state/action/language/embodiment; apply optional min-max normalization before padding
        # Depth is kept separate from RGB for late fusion
        GrootPackInputsStep(
            state_horizon=state_horizon,
            action_horizon=action_horizon,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            language_key="task",
            formalize_language=False,
            embodiment_tag=config.embodiment_tag,
            normalize_min_max=True,
            stats=padded_stats,
            use_depth=getattr(config, 'use_depth', False),
        ),
        # 4. Eagle encode RGB images (creates eagle_content), depth passes through unchanged
        GrootEagleEncodeStep(
            tokenizer_assets_repo=config.tokenizer_assets_repo,
        ),
        # 5. Collate eagle_content -> eagle_* tensors, then concatenate depth with RGB pixel_values
        GrootEagleCollateStep(
            tokenizer_assets_repo=config.tokenizer_assets_repo,
            depth_scale=getattr(config, 'depth_scale', 0.001),
            depth_mean=getattr(config, 'depth_mean', 0.5),
            depth_std=getattr(config, 'depth_std', 0.5),
        ),
        # 6. Move to device
        DeviceProcessorStep(device=config.device),
    ]

    # Postprocessing: slice to env action dim and unnormalize to env scale, then move to CPU
    output_steps: list[ProcessorStep] = [
        GrootActionUnpackUnnormalizeStep(
            env_action_dim=env_action_dim,
            stats=padded_stats,
            normalize_min_max=True,
        ),
        # Finally, move to CPU for env interaction
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


# GR00T specific processor steps


def _to_uint8_np_bhwc(img_t: torch.Tensor) -> np.ndarray:
    # img_t: (B, C, H, W) float in [0,1] or uint8
    if img_t.dtype.is_floating_point:
        img_t = (img_t.clamp(0, 1) * 255.0).to(torch.uint8)
    return rearrange(img_t.cpu().numpy(), "b c h w -> b h w c")


def _prepare_depth_tensor(depth_t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Prepare depth tensor for late fusion with RGB.
    
    Keeps depth values unnormalized (preserving metric depth information).
    Only reshapes and resizes to match RGB spatial dimensions.
    
    Args:
        depth_t: Depth tensor of shape (B, 1, H, W) or (B, H, W), any dtype
        target_h: Target height to match RGB
        target_w: Target width to match RGB
        
    Returns:
        torch.Tensor of shape (B, 1, H, W) as float32, unnormalized
    """
    # Handle different input shapes
    if depth_t.dim() == 3:  # (B, H, W)
        depth_t = depth_t.unsqueeze(1)  # (B, 1, H, W)
    
    # Convert to float32 if needed (preserve actual values)
    depth_t = depth_t.to(torch.float32)
    
    # Resize if needed to match RGB dimensions
    b, c, h, w = depth_t.shape
    if h != target_h or w != target_w:
        depth_t = torch.nn.functional.interpolate(
            depth_t, size=(target_h, target_w), mode='nearest'
        )
    
    return depth_t.cpu()


def _build_eagle_processor(tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO) -> ProcessorMixin:
    # Validate that the cache directory is ready. If not, instruct the user.
    cache_dir = HF_LEROBOT_HOME / tokenizer_assets_repo
    required = [
        cache_dir / "processor_config.json",
        cache_dir / "preprocessor_config.json",
        cache_dir / "image_processing_eagle2_5_vl_fast.py",
    ]
    if not all(p.exists() for p in required):
        raise FileNotFoundError(
            f"[GROOT] Eagle processor cache at '{cache_dir}' is not populated. "
            "Vendor files are copied during model creation. Create the policy/model first, "
            "or call ensure_eagle_cache_ready() before building processors."
        )
    proc = AutoProcessor.from_pretrained(str(cache_dir), trust_remote_code=True, use_fast=True)
    proc.tokenizer.padding_side = "left"
    return proc


@dataclass
@ProcessorStepRegistry.register(name="groot_pack_inputs_v3")
class GrootPackInputsStep(ProcessorStep):
    state_horizon: int = 1
    action_horizon: int = 16
    max_state_dim: int = 64
    max_action_dim: int = 32
    language_key: str = "task"
    formalize_language: bool = False
    embodiment_tag: str = "new_embodiment"
    embodiment_mapping: dict[str, int] = field(
        default_factory=lambda: {
            "new_embodiment": 31,  # Match original GR00T EMBODIMENT_TAG_MAPPING
            "oxe_droid": 17,
            "agibot_genie1": 26,
            "gr1": 24,
            "so100": 2,
            "unitree_g1": 3,
        }
    )
    # Depth modality augmentation support (paper: "Modality-Augmented Fine-Tuning")
    use_depth: bool = False
    # Min-max normalization (SO100-like) applied BEFORE padding
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        def _align_vec(vec: Any, target_dim: int, *, default: float) -> torch.Tensor:
            t = torch.as_tensor(vec)
            t = t.flatten().to(
                dtype=torch.float32,
                device=next(
                    (v.device for v in obs.values() if isinstance(v, torch.Tensor)), torch.device("cpu")
                ),
            )
            d = int(t.shape[-1]) if t.numel() > 0 else 0
            if d == target_dim:
                return t
            if d < target_dim:
                pad = torch.full((target_dim - d,), default, dtype=t.dtype, device=t.device)
                return torch.cat([t, pad], dim=0)
            return t[:target_dim]

        def _min_max_norm(x: torch.Tensor, key: str) -> torch.Tensor:
            if not self.normalize_min_max:
                return x
            if self.stats is None or key not in self.stats:
                return x
            stats_k = self.stats[key]
            last_dim = x.shape[-1]
            min_v = _align_vec(stats_k.get("min", torch.zeros(last_dim)), last_dim, default=0.0)
            max_v = _align_vec(stats_k.get("max", torch.ones(last_dim)), last_dim, default=1.0)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            mapped = 2 * (x - min_v) / safe_denom - 1
            return torch.where(mask, mapped, torch.zeros_like(mapped))

        # 1) Video (B, T=1, V, C, H, W) uint8 - RGB only
        # Depth is kept separate and concatenated AFTER Eagle processing
        img_keys = sorted([k for k in obs if k.startswith("observation.images.") and "depth" not in k.lower()])
        if not img_keys and "observation.image" in obs:
            img_keys = ["observation.image"]
        
        if img_keys:
            cams = [_to_uint8_np_bhwc(obs[k]) for k in img_keys]  # List of (B, H, W, 3)
            b, h, w, _ = cams[0].shape  # Get spatial dims from first camera
            
            # Handle depth: keep separate for late fusion (after Eagle RGB processing)
            if self.use_depth:
                # Find depth images based on the img_keys
                depth_keys = []
                found_depth = False
                # TODO(ofekp): note that this means we do not support depth streams that have no corresponding RGB steam
                for img_key in img_keys:
                    if img_key == "observation.image":
                        assert "observation.image.depth" in obs, "[GROOT] Error: depth_key not found in observations."
                        depth_keys.append("observation.image.depth")
                        found_depth = True
                        break
                    depth_key = [f"{img_key}.depth" for img_key in img_key]
                    if depth_key in obs:
                        found_depth = True
                        depth_keys.append(depth_key)
                    else:
                        depth_keys.append(None)  # indicates that this camera stream has no depth

                if not depth_keys or len(depth_keys) == 0:
                    raise Exception("[GROOT] Warning: use_depth=True but no depth images found. Using RGB only.")
                
                # Prepare depth tensors (unnormalized, just resized to match RGB)
                # TODO(ofekp): take care of the device when torch.zeros is called e.g. device=next(iter(obs.values())).device
                # import pdb; pdb.set_trace()
                depth_tensors = [_prepare_depth_tensor(obs[k], h, w) if k is not None else torch.zeros((b, 1, h, w)).cpu() for k in depth_keys]  # List of (B, 1, H, W)
                
                # Stack depth tensors: (B, V, 1, H, W) where V = num cameras
                depth_stacked = torch.stack(depth_tensors, dim=1)  # (B, V, 1, H, W)
                depth_stacked = depth_stacked.unsqueeze(1)  # (B, 1, V, 1, H, W) to match video T dim
                obs["depth_raw"] = depth_stacked  # Keep for late fusion
                    
            # Create RGB-only video (depth will be concatenated after Eagle processing)
            video = np.stack(cams, axis=1)  # (B, V, H, W, 3)
            video = np.expand_dims(video, axis=1)  # (B, 1, V, H, W, 3)
            # Reorder to (B, T, V, C, H, W) - always C=3 for RGB
            video = np.transpose(video, (0, 1, 2, 5, 3, 4))  # (B, 1, V, 3, H, W)
            obs["video"] = video
            # Drop raw images to avoid confusion downstream
            for k in img_keys:
                obs.pop(k, None)
            # Drop original depth keys (we've stored processed depth in depth_raw)
            if self.use_depth:
                for k in list(obs.keys()):
                    if "depth" in k.lower() and k != "depth_raw":
                        obs.pop(k, None)

        # 2) Language (string)
        lang = comp.get(self.language_key)
        if isinstance(lang, list):
            lang = lang[0] if len(lang) > 0 else None
        if not lang:
            lang = "Perform the task."
        if self.formalize_language:
            lang = (lang or "").lower()
            lang = "".join(ch for ch in lang if ch.isalnum() or ch.isspace())
        comp["language"] = lang

        # 3) State/state_mask -> (B, 1, max_state_dim)
        if "observation.state" in obs:
            state = obs["observation.state"]  # (B, D)
            if state.dim() != 2:
                raise ValueError(f"state must be (B, D), got {tuple(state.shape)}")
            bsz, d = state.shape
            # Normalize BEFORE padding
            if self.normalize_min_max:
                state = _min_max_norm(state, "observation.state")
            state = state.unsqueeze(1)  # (B, 1, D)
            if d > self.max_state_dim:
                state = state[:, :, : self.max_state_dim]
                d = self.max_state_dim
            elif d < self.max_state_dim:
                pad = torch.zeros(bsz, 1, self.max_state_dim - d, dtype=state.dtype, device=state.device)
                state = torch.cat([state, pad], dim=2)
            state_mask = torch.zeros(bsz, 1, self.max_state_dim, dtype=torch.bool, device=state.device)
            state_mask[:, :, :d] = True
            obs["state"] = state
            obs["state_mask"] = state_mask

        # 4) Action/action_mask -> (B, action_horizon, max_action_dim)
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            # Normalize BEFORE temporal expansion/padding
            if self.normalize_min_max:
                if action.dim() == 2:
                    action = _min_max_norm(action, "action")
                elif action.dim() == 3:
                    b, t, d = action.shape
                    flat = action.reshape(b * t, d)
                    flat = _min_max_norm(flat, "action")
                    action = flat.view(b, t, d)
            if action.dim() == 2:
                action = action.unsqueeze(1).repeat(1, self.action_horizon, 1)
            elif action.dim() == 3:
                b, t, d = action.shape
                if t < self.action_horizon:
                    last = action[:, -1:, :]
                    pad = last.repeat(1, self.action_horizon - t, 1)
                    action = torch.cat([action, pad], dim=1)
                elif t > self.action_horizon:
                    action = action[:, : self.action_horizon, :]
            else:
                raise ValueError(f"action must be (B, D) or (B, T, D), got {tuple(action.shape)}")

            b, t, d = action.shape
            if d > self.max_action_dim:
                action = action[:, :, : self.max_action_dim]
                d = self.max_action_dim
            elif d < self.max_action_dim:
                pad = torch.zeros(b, t, self.max_action_dim - d, dtype=action.dtype, device=action.device)
                action = torch.cat([action, pad], dim=2)
            action_mask = torch.zeros(b, t, self.max_action_dim, dtype=torch.bool, device=action.device)
            action_mask[:, :, :d] = True
            transition[TransitionKey.ACTION] = action
            comp["action_mask"] = action_mask

        # 5) Embodiment id as LongTensor (B,)
        emb_id = self.embodiment_mapping.get(self.embodiment_tag, 0)
        # Infer batch size/device from any tensor in obs or action
        bsz = None
        device = torch.device("cpu")
        for v in list(obs.values()) + [transition.get(TransitionKey.ACTION)]:
            if isinstance(v, torch.Tensor):
                bsz = v.shape[0]
                device = v.device
                break
        if bsz is None and "video" in obs and isinstance(obs["video"], np.ndarray):
            bsz = obs["video"].shape[0]
        if bsz is None:
            bsz = 1
        comp["embodiment_id"] = torch.full((bsz,), emb_id, dtype=torch.long, device=device)

        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    # Pipeline API requirement: declare how features change (we keep it simple)
    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        """
        Returns a serializable dictionary of the processor's configuration.

        Excludes 'stats' since they are saved separately via state_dict().
        """
        return {
            "state_horizon": self.state_horizon,
            "action_horizon": self.action_horizon,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "language_key": self.language_key,
            "formalize_language": self.formalize_language,
            "embodiment_tag": self.embodiment_tag,
            "embodiment_mapping": self.embodiment_mapping,
            "normalize_min_max": self.normalize_min_max,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Returns normalization statistics as a flat state dictionary.

        This enables saving stats to safetensors files, similar to normalizer_processor.
        """
        if not self.stats:
            return {}

        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                tensor = torch.as_tensor(value).cpu()
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """
        Loads normalization statistics from a flat state dictionary.

        This enables loading stats from safetensors files during from_pretrained.
        """
        if not state:
            return

        reconstructed: dict[str, dict[str, Any]] = {}
        for flat_key, tensor in state.items():
            if "." in flat_key:
                key, stat_name = flat_key.rsplit(".", 1)
                if key not in reconstructed:
                    reconstructed[key] = {}
                reconstructed[key][stat_name] = tensor

        if reconstructed:
            self.stats = reconstructed


@dataclass
@ProcessorStepRegistry.register(name="groot_eagle_encode_v3")
class GrootEagleEncodeStep(ProcessorStep):
    tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO
    _proc: ProcessorMixin | None = field(default=None, init=False, repr=False)

    @property
    def proc(self) -> ProcessorMixin:
        if self._proc is None:
            self._proc = _build_eagle_processor(self.tokenizer_assets_repo)
        return self._proc

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        if "video" not in obs:
            return transition

        video = obs["video"]  # (B, T, V, C, H, W) uint8, C=3 (RGB only)
        assert video.shape[3] == 3, f"[GROOT] Expected RGB video with 3 channels, got {video.shape[3]} channels."
        lang = comp.get("language", "Perform the task.")
        if isinstance(lang, list):
            lang = lang[0] if len(lang) > 0 else "Perform the task."

        bsz = video.shape[0]
        eagle_contents: list[dict[str, Any]] = []
        for b in range(bsz):
            vt = video[b]  # (T, V, C, H, W) after reorder
            if vt.ndim != 5:
                # Fallback: assume (T, V, H, W, C)
                t, v, h, w, c = vt.shape
                flat = rearrange(vt, "t v h w c -> (t v) h w c")
            else:
                t, v, c, h, w = vt.shape
                flat = rearrange(vt, "t v c h w -> (t v) h w c")
            # Create PIL images from RGB (3-channel) data only
            images = [Image.fromarray(flat[i]) for i in range(t * v)]
            # Format language as string list representation to match Original GROOT
            lang_formatted = str([lang])
            text_content = [{"type": "text", "text": lang_formatted}]
            image_content = [{"type": "image", "image": img} for img in images]
            conv = [{"role": "user", "content": image_content + text_content}]
            text_list = [self.proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)]
            img_inputs, vid_inputs = self.proc.process_vision_info(conv)
            eagle_contents.append(
                {
                    "text_list": text_list,
                    "image_inputs": img_inputs,
                    "video_inputs": vid_inputs,
                }
            )

        comp["eagle_content"] = eagle_contents
        # Pass through depth_raw unchanged for late fusion in collate step
        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    # Pipeline API requirement: declare how features change (no schema change here)
    def transform_features(self, features):
        return features


# Original GR00T-style collate: converts eagle_content -> eagle_* tensors
def collate(features: list[dict[str, Any]], eagle_processor: ProcessorMixin) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            text_list: list[str] = []
            image_inputs: list[Any] = []
            # DEBUG(ofekp):
            # len(values) == 3 - this is the batch size
            # values[2]["image_inputs"][0].getpixel((100,100)) --> (185, 183, 177)
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs
            # eagle_processor is of type Eagle25VLProcessor
            # contains image_processor of type Eagle25VLImageProcessorFast with params:
            # do_convert_rgb = true
            # do_normalize = true
            # do_rescale = true
            # do_resize = false
            # data_format = "channels_first"
            # image_mean = [0.5, 0.5, 0.5]
            # tokens_per_tile = 256
            # use_thumbnail = true
            # tokenizer is Qwen2TokenizerFast
            # most likely the scaling happens in the func `rescale_and_normalize` (see https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L558)
            eagle_inputs = eagle_processor(
                text=text_list,
                images=image_inputs,
                images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},
                return_tensors="pt",
                padding=True,
            )
            for k, v in eagle_inputs.items():
                k = "eagle_" + k
                batch[k] = v
            # DEBUG(ofekp):
            # batch.keys() --> ['eagle_input_ids', 'eagle_attention_mask', 'eagle_pixel_values', 'eagle_image_sizes']
            # batch["eagle_pixel_values"].shape --> torch.Size([3, 3, 224, 224]) which is b,c,h,w
            # batch["eagle_pixel_values"][2][:,10,100] --> tensor([0.5373, 0.5059, 0.4588])
            # batch["eagle_pixel_values"][2][:,100,100] --> tensor([-0.0196, -0.0196, -0.0196])
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


def _normalize_depth_for_fusion(
    depth: torch.Tensor,
    depth_scale: float = 1.0,
    depth_mean: float = 0.5,
    depth_std: float = 0.5,
) -> torch.Tensor:
    """Normalize depth tensor to match the scale of Eagle-processed RGB.
    
    Eagle RGB processing: pixel_values are normalized with ImageNet stats,
    resulting in values roughly in [-2, 2] range.
    
    For depth, we apply a simple linear normalization:
    1. Scale raw depth by depth_scale (e.g., 1/1000 for mm->m, or 1/10 for m->normalized)
    2. Apply (depth - mean) / std to center around 0
    
    Args:
        depth: Raw depth tensor (B, 1, H, W) in original units (mm or m)
        depth_scale: Scale factor to apply to raw depth (default 1.0 = no scaling)
        depth_mean: Mean for normalization (default 0.5)
        depth_std: Std for normalization (default 0.5)
        
    Returns:
        Normalized depth tensor suitable for concatenation with RGB pixel_values
    """
    # Apply scale (e.g., convert mm to meters, or normalize to expected range)
    depth_scaled = depth * depth_scale
    # Normalize similar to how RGB is normalized
    depth_normalized = (depth_scaled - depth_mean) / depth_std
    return depth_normalized


@dataclass
@ProcessorStepRegistry.register(name="groot_eagle_collate_v3")
class GrootEagleCollateStep(ProcessorStep):
    tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO
    # Depth normalization parameters for late fusion
    # Set depth_scale based on your depth units: 1/1000 for mm, 1/10 for m (to get ~[0,1] range)
    depth_scale: float = 0.001  # Default assumes depth in mm, converts to meters
    depth_mean: float = 0.5  # Center depth around 0 after scaling
    depth_std: float = 0.5   # Scale to roughly match RGB normalized range
    _proc: ProcessorMixin | None = field(default=None, init=False, repr=False)
    _logged_first_depth: bool = field(default=False, init=False, repr=False)

    @property
    def proc(self) -> ProcessorMixin:
        if self._proc is None:
            self._proc = _build_eagle_processor(self.tokenizer_assets_repo)
        return self._proc

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        contents = comp.get("eagle_content")
        if not contents:
            return transition

        # Build features list as original API expects: one dict per batch item
        features = [{"eagle_content": content} for content in contents]
        batched = collate(features, self.proc)

        # Inject eagle_* tensors and remove the temporary content and raw video to free memory
        for k, v in batched.items():
            comp[k] = v
        comp.pop("eagle_content", None)
        
        # Late fusion: concatenate depth with RGB pixel_values if depth_raw exists
        depth_raw = obs.get("depth_raw")
        if depth_raw is not None and "eagle_pixel_values" in comp:
            # depth_raw: (B, 1, V, 1, H, W) - raw depth values
            # eagle_pixel_values: (N, 3, H', W') where N = B * T * V, normalized RGB
            pixel_values = comp["eagle_pixel_values"]  # (N, 3, H', W')
            n, c, h_pv, w_pv = pixel_values.shape  # [3, 3, 224, 224]
            
            # Reshape depth to match pixel_values layout
            # depth_raw is (B, T=1, V, 1, H, W), we need (N, 1, H', W')
            b, t, v, _, h_d, w_d = depth_raw.shape  # [3, 1, 1, 1, 256, 256]
            depth_flat = depth_raw.view(b * t * v, 1, h_d, w_d)  # (N, 1, H, W)

            assert depth_flat.shape[0] == pixel_values.shape[0]
            
            # Resize depth to match pixel_values spatial dims if needed
            # even though we already matched the depth to the rgb, the rgb is resized inside eagle processor
            # this resize happens inside the `collate(features, self.proc)` call above
            if h_d != h_pv or w_d != w_pv:
                depth_flat = torch.nn.functional.interpolate(
                    depth_flat, size=(h_pv, w_pv), mode='nearest'
                )
            
            # Normalize depth for fusion (converts raw values to normalized scale)
            depth_normalized = _normalize_depth_for_fusion(
                depth_flat,
                depth_scale=self.depth_scale,
                depth_mean=self.depth_mean,
                depth_std=self.depth_std,
            )

            # our scaling is not doing anything on purpose (mean 0, std 1, scale 1) given in the cli command
            # depth_flat == depth_normalized and are in meters

            # Concatenate depth as 4th channel: (N, 4, H', W')
            # TODO(ofekp): verify that cam-depth alignment is correct
            pixel_values_rgbd = torch.cat([pixel_values, depth_normalized], dim=1)
            comp["eagle_pixel_values"] = pixel_values_rgbd
            
            # Log depth stats for first frame only (deterministic, for comparison between runs)
            if not self._logged_first_depth:
                self._logged_first_depth = True
                # First frame of first batch element
                first_depth_raw = depth_flat[0, 0]  # RAW depth before normalization
                first_depth = depth_normalized[0, 0]  # (H', W')
                first_rgb = pixel_values[0]  # (3, H', W')
                
                # Check for suspicious patterns (all rows identical = likely bug)
                unique_rows = torch.unique(first_depth_raw, dim=0).shape[0]
                unique_cols = torch.unique(first_depth_raw, dim=1).shape[0]
                
                print(f"[GROOT DEPTH DEBUG RAW] Before normalization:")
                print(f"  Shape: {first_depth_raw.shape}")
                print(f"  Min: {first_depth_raw.min().item():.6f}, Max: {first_depth_raw.max().item():.6f}")
                print(f"  Mean: {first_depth_raw.mean().item():.6f}, Std: {first_depth_raw.std().item():.6f}")
                print(f"  Unique rows: {unique_rows}/{first_depth_raw.shape[0]}, Unique cols: {unique_cols}/{first_depth_raw.shape[1]}")
                print(f"  Corner samples: TL={first_depth_raw[0,0].item():.6f}, TR={first_depth_raw[0,-1].item():.6f}, BL={first_depth_raw[-1,0].item():.6f}, BR={first_depth_raw[-1,-1].item():.6f}")
                if unique_rows < 10:
                    print(f"  WARNING: Only {unique_rows} unique rows - depth image may be corrupted!")
                
                print(first_depth)
                print(f"[GROOT DEPTH DEBUG] First frame depth stats (normalized, before network):")
                print(f"  Shape: {first_depth.shape}")
                print(f"  Min: {first_depth.min().item():.6f}, Max: {first_depth.max().item():.6f}")
                print(f"  Mean: {first_depth.mean().item():.6f}, Std: {first_depth.std().item():.6f}")
                print(f"  Sample values [0,0]: {first_depth[0,0].item():.6f}, [H//2,W//2]: {first_depth[first_depth.shape[0]//2, first_depth.shape[1]//2].item():.6f}")
                print(f"  RGB channel means: R={first_rgb[0].mean().item():.4f}, G={first_rgb[1].mean().item():.4f}, B={first_rgb[2].mean().item():.4f}")
                
                # Save depth and RGB images for visual comparison between train and eval
                try:
                    import os
                    from PIL import Image as PILImage
                    
                    # Determine if this is training or eval based on model mode
                    # During training, self.training would be True on the model, but we don't have direct access
                    # Use a heuristic: check if we're in output_rgb or output_rgbd folder context
                    mode_suffix = "unknown"
                    # Try to detect from environment or just use timestamp to make unique
                    import time
                    timestamp = int(time.time())
                    
                    output_dir = "./output"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save RAW depth (before normalization) - in meters, normalize to 0-255 for visualization
                    depth_vis = first_depth_raw.cpu().numpy()
                    # Clip to 0-5m range and normalize to 0-255
                    depth_vis_clipped = np.clip(depth_vis, 0, 5.0)
                    depth_vis_uint8 = (depth_vis_clipped / 5.0 * 255).astype(np.uint8)
                    depth_img = PILImage.fromarray(depth_vis_uint8, mode='L')
                    depth_path = os.path.join(output_dir, f"debug_depth_raw_{timestamp}.png")
                    depth_img.save(depth_path)
                    print(f"  [DEBUG] Saved raw depth image to: {depth_path}")
                    
                    # Save normalized depth
                    depth_norm_vis = first_depth.cpu().numpy()
                    # Normalized depth is roughly in [-2, 2] range, map to 0-255
                    depth_norm_clipped = np.clip((depth_norm_vis + 2.0) / 4.0, 0, 1)
                    depth_norm_uint8 = (depth_norm_clipped * 255).astype(np.uint8)
                    depth_norm_img = PILImage.fromarray(depth_norm_uint8, mode='L')
                    depth_norm_path = os.path.join(output_dir, f"debug_depth_normalized_{timestamp}.png")
                    depth_norm_img.save(depth_norm_path)
                    print(f"  [DEBUG] Saved normalized depth image to: {depth_norm_path}")
                    
                    # Save RGB image
                    rgb_vis = first_rgb.cpu().numpy()  # (3, H, W)
                    # RGB is normalized with ImageNet stats, denormalize: x * std + mean
                    # ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    # But Eagle uses mean=0.5, std=0.5
                    rgb_denorm = rgb_vis * 0.5 + 0.5  # Assuming Eagle normalization
                    rgb_denorm = np.clip(rgb_denorm, 0, 1)
                    rgb_uint8 = (rgb_denorm * 255).astype(np.uint8)
                    rgb_uint8 = np.transpose(rgb_uint8, (1, 2, 0))  # (H, W, 3)
                    rgb_img = PILImage.fromarray(rgb_uint8, mode='RGB')
                    rgb_path = os.path.join(output_dir, f"debug_rgb_{timestamp}.png")
                    rgb_img.save(rgb_path)
                    print(f"  [DEBUG] Saved RGB image to: {rgb_path}")
                    
                except Exception as e:
                    print(f"  [DEBUG] Failed to save debug images: {e}")
                
                # Log Eagle processor config for train vs eval comparison
                print(f"\n[GROOT EAGLE PROCESSOR CONFIG]")
                print(f"  Processor type: {type(self.proc).__name__}")
                if hasattr(self.proc, 'image_processor'):
                    img_proc = self.proc.image_processor
                    print(f"  Image processor type: {type(img_proc).__name__}")
                    # Print all config attributes
                    for attr in ['do_convert_rgb', 'do_normalize', 'do_rescale', 'do_resize', 
                                 'image_mean', 'image_std', 'rescale_factor', 'size',
                                 'data_format', 'tokens_per_tile', 'use_thumbnail',
                                 'min_dynamic_tiles', 'max_dynamic_tiles']:
                        if hasattr(img_proc, attr):
                            print(f"    {attr}: {getattr(img_proc, attr)}")
                if hasattr(self.proc, 'tokenizer'):
                    tok = self.proc.tokenizer
                    print(f"  Tokenizer type: {type(tok).__name__}")
                    print(f"    padding_side: {getattr(tok, 'padding_side', 'N/A')}")
                print("")
            
            # Clean up depth_raw
            obs.pop("depth_raw", None)
        
        obs.pop(
            "video", None
        )  # The video has been fully encoded into eagle_* tensors, so we don't need the raw video anymore
        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register(name="groot_action_unpack_unnormalize_v1")
class GrootActionUnpackUnnormalizeStep(ProcessorStep):
    env_action_dim: int = 0
    # Apply inverse of min-max normalization if it was used in preprocessor
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Expect model outputs to be in TransitionKey.ACTION as (B, T, D_model)
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return transition

        # Select last timestep and slice to env dimension
        if action.dim() == 3:
            action = action[:, -1, :]
        # Now action is (B, D_model)
        if self.env_action_dim and action.shape[-1] >= self.env_action_dim:
            action = action[..., : self.env_action_dim]

        # Inverse min-max normalization mirroring _min_max_norm:
        # forward: y = 2 * (x - min) / denom - 1, with y=0 when denom==0
        # inverse: x = (y+1)/2 * denom + min, and when denom==0 -> x = min
        if self.normalize_min_max and self.stats is not None:
            stats_k = self.stats.get("action", {})
            d = action.shape[-1]
            min_v = torch.as_tensor(
                stats_k.get("min", torch.zeros(d)), dtype=action.dtype, device=action.device
            )
            max_v = torch.as_tensor(
                stats_k.get("max", torch.ones(d)), dtype=action.dtype, device=action.device
            )
            if min_v.numel() != d:
                min_v = torch.nn.functional.pad(min_v.flatten()[:d], (0, max(0, d - min_v.numel())))
                min_v = min_v.to(action.device, dtype=action.dtype)
            if max_v.numel() != d:
                max_v = torch.nn.functional.pad(max_v.flatten()[:d], (0, max(0, d - max_v.numel())))
                max_v = max_v.to(action.device, dtype=action.dtype)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            inv = (action + 1.0) * 0.5 * safe_denom + min_v
            action = torch.where(mask, inv, min_v)

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        """
        Returns a serializable dictionary of the processor's configuration.

        Excludes 'stats' since they are saved separately via state_dict().
        """
        return {
            "env_action_dim": self.env_action_dim,
            "normalize_min_max": self.normalize_min_max,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Returns normalization statistics as a flat state dictionary.

        This enables saving stats to safetensors files, similar to normalizer_processor.
        """
        if not self.stats:
            return {}

        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                tensor = torch.as_tensor(value).cpu()
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """
        Loads normalization statistics from a flat state dictionary.

        This enables loading stats from safetensors files during from_pretrained.
        """
        if not state:
            return

        reconstructed: dict[str, dict[str, Any]] = {}
        for flat_key, tensor in state.items():
            if "." in flat_key:
                key, stat_name = flat_key.rsplit(".", 1)
                if key not in reconstructed:
                    reconstructed[key] = {}
                reconstructed[key][stat_name] = tensor

        if reconstructed:
            self.stats = reconstructed
