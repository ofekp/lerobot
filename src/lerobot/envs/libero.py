#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite
import libero
import re


def fix_resource_paths(xml_string):
    if isinstance(xml_string, (bytes, np.bytes_)):
        xml_string = xml_string.decode("utf-8")
    
    # 1. Get the absolute base paths for your current environment
    rs_base = os.path.dirname(robosuite.__file__)
    # Use __path__[0] if os.path.dirname comes up empty
    libero_base = os.path.abspath(libero.__path__[0])

    # 2. Fix Robosuite Paths
    # Finds anything like ".../robosuite/models/assets/..." and points to local rs_base
    xml_string = re.sub(r'\"/[^"]+/robosuite/models/', f'"{rs_base}/models/', xml_string)
    
    # 3. Fix LIBERO Paths
    # LIBERO assets in the XML often look like ".../chiliocosm/assets/..."
    # These map to your local LIBERO installation's assets folder
    libero_assets_path = os.path.join(libero_base, "libero", "assets")
    xml_string = re.sub(r'\"/[^"]+/chiliocosm/assets/', f'"{libero_assets_path}/', xml_string)
    
    # 4. Fix relative meshdir path to absolute path
    # The XML has meshdir="meshes/" which is relative - MuJoCo needs absolute path
    # LIBERO uses Panda robot, so meshes are in robosuite/models/assets/robots/panda/meshes/
    robot_meshdir = os.path.join(rs_base, "models", "assets", "robots", "panda", "meshes")
    xml_string = re.sub(r'meshdir="meshes/"', f'meshdir="{robot_meshdir}/"', xml_string)
    
    # 5. Final safety check: Replace any remaining /Users/yifengz/ with local paths
    # (Just in case the regex missed a non-standard path)
    if "/Users/yifengz/" in xml_string:
        xml_string = xml_string.replace("/Users/yifengz/workspace/robosuite-master/robosuite/", rs_base + "/")
        xml_string = xml_string.replace("/Users/yifengz/workspace/libero-dev/chiliocosm/assets/", libero_assets_path + "/")

    return xml_string


def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize camera_name into a non-empty list of strings."""
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list | tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


def _get_suite(name: str) -> benchmark.Benchmark:
    """Instantiate a LIBERO suite by name with clear validation."""
    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite


def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    """Validate/normalize task ids. If None → all tasks."""
    if task_ids is None:
        return list(range(total_tasks))
    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    return ids


def get_task_init_states(task_suite: Any, i: int) -> np.ndarray:
    init_states_path = (
        Path(get_libero_path("init_states"))
        / task_suite.tasks[i].problem_folder
        / task_suite.tasks[i].init_states_file
    )
    # e.g.
    # init_states_path = /usr/local/lib/python3.10/dist-packages/libero/libero/init_files/libero_goal/open_the_top_drawer_and_put_the_bowl_inside.pruned_init
    init_states = torch.load(init_states_path, weights_only=False)  # nosec B614
    return init_states


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def mujoco_depth_to_meters(z_buf, near, far):
    """Convert MuJoCo's normalized z-buffer depth to actual depth in meters."""
    return (near * far) / (far - z_buf * (far - near))


ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


class LiberoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite: Any,
        task_id: int,
        task_suite_name: str,
        episode_length: int | None = None,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        obs_type: str = "pixels",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        init_states: bool = True,
        episode_index: int = 0,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
        control_mode: str = "relative",
        use_depth: bool = False,
        use_voxel: bool = False,
    ):
        super().__init__()
        self.task_id = task_id
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.use_depth = use_depth
        self.use_voxel = use_voxel
        self.camera_name = _parse_camera_names(
            camera_name
        )  # agentview_image (main) or robot0_eye_in_hand_image (wrist)

        # Map raw camera names to "image" and "image2".
        # The preprocessing step `preprocess_observation` will then prefix these with `.images.*`,
        # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
        # This ensures the policy consistently receives observations in the
        # expected format regardless of the original camera naming.
        if camera_name_mapping is None or not camera_name_mapping:
            # Default mapping for standard LIBERO camera setup
            camera_name_mapping = {
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "image2",
            }
        self.camera_name_mapping = camera_name_mapping
        self.num_steps_wait = num_steps_wait
        self.episode_index = episode_index
        self.episode_length = episode_length
        # Load once and keep
        self._init_states = get_task_init_states(task_suite, self.task_id) if self.init_states else None
        self._init_state_id = self.episode_index  # tie each sub-env to a fixed init state

        self._env = self._make_envs_task(task_suite, self.task_id)
        default_steps = 500
        self._max_episode_steps = (
            TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)
            if self.episode_length is None
            else self.episode_length
        )
        self.control_mode = control_mode
        images = {}
        depths = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
            # Add depth space if depth is enabled
            if self.use_depth:
                depth_key = cam.replace("_image", "_depth")
                depths[self.camera_name_mapping[depth_key]] = spaces.Box(
                    low=0,
                    high=65535,  # uint16 max value (millimeters)
                    shape=(self.observation_height, self.observation_width),
                    dtype=np.uint16,
                )

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )

        elif self.obs_type == "pixels":
            obs_space = {"pixels": spaces.Dict(images)}
            if self.use_depth:
                obs_space["depths"] = spaces.Dict(depths)
            self.observation_space = spaces.Dict(obs_space)
        elif self.obs_type == "pixels_agent_pos":
            obs_space = {
                "pixels": spaces.Dict(images),
                "robot_state": spaces.Dict(
                    {
                        "eef": spaces.Dict(
                            {
                                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                                "quat": spaces.Box(
                                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                                ),
                                "mat": spaces.Box(
                                    low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64
                                ),
                            }
                        ),
                        "gripper": spaces.Dict(
                            {
                                "qpos": spaces.Box(
                                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                ),
                                "qvel": spaces.Box(
                                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                ),
                            }
                        ),
                        "joints": spaces.Dict(
                            {
                                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                            }
                        ),
                    }
                ),
            }
            if self.use_depth and depths:
                obs_space["depths"] = spaces.Dict(depths)
            self.observation_space = spaces.Dict(obs_space)

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def render(self):
        raw_obs = self._env.env._get_observations()
        formatted_obs = self._format_raw_obs(raw_obs)
        # image = formatted_obs["pixels"]["image"]
        image = formatted_obs["pixels"][self.camera_name_mapping[self.camera_name[0]]]
        image = image[::-1, ::-1]  # flip both H and W for visualization
        assert image.shape[2] == 3
        
        # If depth is enabled, visualize it side-by-side with RGB
        if self.use_depth and "depths" in formatted_obs:
            # Get depth for the same camera as the main image
            # Assuming "image" key exists, look for corresponding depth
            depth_key = None
            for cam_name in self.camera_name:
                if self.camera_name_mapping.get(cam_name) == "image":
                    depth_key = self.camera_name_mapping.get(cam_name.replace("_image", "_depth"))
                    break
            if depth_key and depth_key in formatted_obs["depths"]:
                depth = formatted_obs["depths"][depth_key]
                depth = depth[::-1, ::-1]  # flip to match image orientation
                # Normalize depth to 0-255 for visualization
                # Clip to reasonable range (e.g., 0-5 meters = 0-5000mm)
                depth_vis = np.clip(depth, 0, 5000)
                depth_vis = (depth_vis / 5000.0 * 255).astype(np.uint8)
                # Convert to RGB (grayscale colormap)
                depth_rgb = np.stack([depth_vis, depth_vis, depth_vis], axis=-1)
                # Concatenate horizontally: [RGB | Depth]
                image = np.concatenate([image, depth_rgb], axis=1)

        # If voxel encoder is active, add voxel projection to the right
        if self.use_voxel:
            try:
                from lerobot.policies.groot.voxel_encoder import (
                    _voxel_grid_cache,
                    render_voxel_projections,
                )
                if _voxel_grid_cache is not None:
                    voxel_img = render_voxel_projections(_voxel_grid_cache, scale=1)
                    # Resize voxel image to match the height of the render
                    h_target = image.shape[0]
                    h_vox, w_vox = voxel_img.shape[:2]
                    if h_vox != h_target:
                        from PIL import Image as _PILImage
                        w_new = int(w_vox * h_target / h_vox)
                        voxel_img = np.array(
                            _PILImage.fromarray(voxel_img).resize((w_new, h_target), _PILImage.NEAREST)
                        )
                    image = np.concatenate([image, voxel_img], axis=1)
            except Exception:
                pass  # visualization is best-effort

        return image

    def _make_envs_task(self, task_suite: Any, task_id: int = 0):
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        # Extract base camera names (without _image suffix) for robosuite
        # e.g., "frontview_image" -> "frontview"
        camera_names_for_env = [cam.replace("_image", "") for cam in self.camera_name]
        
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.observation_height,
            "camera_widths": self.observation_width,
            "camera_depths": self.use_depth,  # Enable depth rendering
            "camera_names": camera_names_for_env,  # Specify which cameras to render
        }
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        return env

    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        images = {}
        depths = {}
        should_flip = False
        for camera_name in self.camera_name:
            image = raw_obs[camera_name]
            # LIBERO/robosuite images are rotated 180° - flip both H and W to match training data orientation
            if should_flip:
                image = image[::-1, ::-1]
            #image = image[:, ::-1, :]  # Flip W to match LeRobot dataset format
            assert image.shape[2] == 3
            images[self.camera_name_mapping[camera_name]] = image
            
            # Handle depth if enabled
            # Note: robosuite renders depth for ALL cameras when camera_depths=True
            # We extract only the cameras specified in self.camera_name
            if self.use_depth:
                # LIBERO/robosuite depth keys are camera_name with "_depth" suffix
                # e.g., "agentview_image" -> "agentview_depth"
                depth_key = camera_name.replace("_image", "_depth")
                if depth_key in raw_obs:
                    depth = raw_obs[depth_key]
                    # Robosuite depth is (H, W, 1) - squeeze to (H, W)
                    if depth.ndim == 3 and depth.shape[2] == 1:
                        depth = depth.squeeze(-1)
                    # Flip depth to match image orientation
                    if should_flip:
                        depth = depth[::-1, ::-1]
                    
                    # CRITICAL: Robosuite depth is in normalized z-buffer format (0-1), NOT meters!
                    # Must convert using camera near/far planes to get actual depth in meters.
                    # This matches what was done during training data collection in replay.py
                    # new_sim = MjSim.from_xml_string(modified_xml)
                    # # Manually initialize the offscreen rendering context
                    # # This is necessary because MjSim.from_xml_string doesn't auto-init the renderer
                    # render_context = MjRenderContextOffscreen(new_sim, device_id=-1) # -1 uses default/OSMesa
                    # new_sim._render_context_offscreen = render_context

                    sim = self._env.env.sim
                    near = sim.model.vis.map.znear * sim.model.stat.extent
                    far = sim.model.vis.map.zfar * sim.model.stat.extent
                    # print(f"Near: {near}, Far: {far}, Extent: {sim.model.stat.extent}")
                    depth_meters = mujoco_depth_to_meters(depth, near, far)

                    # import h5py
                    # from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
                    # hdf5_path = "/data_host/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5"
                    # modified_xml = None
                    # with h5py.File(hdf5_path, "r") as f:
                    #     # import pdb; pdb.set_trace()
                    #     demo_group = f['data']['demo_0']
                    #     model_xml = demo_group.attrs.get("model_file")
                    #     modified_xml = fix_resource_paths(model_xml)
                    # new_sim = MjSim.from_xml_string(modified_xml)
                    # # Manually initialize the offscreen rendering context
                    # # This is necessary because MjSim.from_xml_string doesn't auto-init the renderer
                    # render_context = MjRenderContextOffscreen(new_sim, device_id=-1) # -1 uses default/OSMesa
                    # new_sim._render_context_offscreen = render_context
                    # near = new_sim.model.vis.map.znear * new_sim.model.stat.extent
                    # far = new_sim.model.vis.map.zfar * new_sim.model.stat.extent
                    # print(f"Near: {near}, Far: {far} Extent: {new_sim.model.stat.extent}")
                    # depth_meters = mujoco_depth_to_meters(depth, near, far)

                    # import pdb; pdb.set_trace()
                    
                    # Convert to uint16 millimeters for consistency with LeRobot dataset format
                    depth_mm = np.clip(depth_meters * 1000.0, 0, 65535).astype(np.uint16)
                    # Use same mapping key as RGB (camera_name, not depth_key)
                    # so depths["image"] matches pixels["image"]
                    depths[self.camera_name_mapping[depth_key]] = depth_mm

        eef_pos = raw_obs.get("robot0_eef_pos")
        eef_quat = raw_obs.get("robot0_eef_quat")

        # rotation matrix from controller
        eef_mat = self._env.robots[0].controller.ee_ori_mat if eef_pos is not None else None
        gripper_qpos = raw_obs.get("robot0_gripper_qpos")
        gripper_qvel = raw_obs.get("robot0_gripper_qvel")
        joint_pos = raw_obs.get("robot0_joint_pos")
        joint_vel = raw_obs.get("robot0_joint_vel")
        obs = {
            "pixels": images,
            "robot_state": {
                "eef": {
                    "pos": eef_pos,  # (3,)
                    "quat": eef_quat,  # (4,)
                    "mat": eef_mat,  # (3, 3)
                },
                "gripper": {
                    "qpos": gripper_qpos,  # (2,)
                    "qvel": gripper_qvel,  # (2,)
                },
                "joints": {
                    "pos": joint_pos,  # (7,)
                    "vel": joint_vel,  # (7,)
                },
            },
        }
        
        # Add depth observations if available
        if self.use_depth:
            obs["depths"] = depths

        # Extract camera intrinsics/extrinsics for voxel encoder
        if self.use_voxel:
            sim = self._env.env.sim
            cam_name_base = self.camera_name[0].replace("_image", "")
            cam_id = sim.model.camera_name2id(cam_name_base)

            # Intrinsics from FOV
            fovy = sim.model.cam_fovy[cam_id]
            f = (self.observation_height / 2.0) / np.tan(np.radians(fovy) / 2.0)
            cx = self.observation_width / 2.0
            cy = self.observation_height / 2.0
            intrinsics = np.array([
                [f,  0, cx],
                [0,  f, cy],
                [0,  0,  1],
            ], dtype=np.float32)

            # Extrinsics: store [R_cam_to_world | cam_pos] — same convention
            # as replay.py's get_camera_params().  The voxel encoder's
            # backproject() expects this format and handles the OpenGL→OpenCV
            # axis conversion internally.
            cam_pos = sim.data.cam_xpos[cam_id].copy()
            R_cam_to_world = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
            extrinsics = np.eye(4, dtype=np.float32)
            extrinsics[:3, :3] = R_cam_to_world
            extrinsics[:3, 3] = cam_pos

            obs["camera_intrinsics"] = intrinsics
            obs["camera_extrinsics"] = extrinsics

        # images['image'].shape --> (256, 256, 3), uint8
        # images['image.depth'].shape --> (256, 256), millimeters as uint16
        # self.obs_type is "pixels_agent_pos"

        if self.obs_type == "pixels":
            result = {"pixels": images.copy()}
            if self.use_depth:
                result["depths"] = depths.copy()
            if self.use_voxel:
                result["camera_intrinsics"] = obs["camera_intrinsics"]
                result["camera_extrinsics"] = obs["camera_extrinsics"]
            return result

        if self.obs_type == "pixels_agent_pos":
            # Validate required fields are present
            if eef_pos is None or eef_quat is None or gripper_qpos is None:
                raise ValueError(
                    f"Missing required robot state fields in raw observation. "
                    f"Got eef_pos={eef_pos is not None}, eef_quat={eef_quat is not None}, "
                    f"gripper_qpos={gripper_qpos is not None}"
                )
            return obs

        raise NotImplementedError(
            f"The observation type '{self.obs_type}' is not supported in LiberoEnv. "
            "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
        )

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._env.seed(seed)
        raw_obs = self._env.reset()
        if self.init_states and self._init_states is not None:
            self._env.set_init_state(self._init_states[self._init_state_id])
            raw_obs = self._env.env._get_observations()

        # After reset, objects may be unstable (slightly floating, intersecting, etc.).
        # Step the simulator with a no-op action for a few frames so everything settles.
        # Increasing this value can improve determinism and reproducibility across resets.
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())

        if self.control_mode == "absolute":
            for robot in self._env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self._env.robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env.check_success()
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": done,
                "is_success": is_success,
            }
        )
        observation = self._format_raw_obs(raw_obs)
        if terminated:
            info["final_info"] = {
                "task": self.task,
                "task_id": self.task_id,
                "done": bool(done),
                "is_success": bool(is_success),
            }
            self.reset()
        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        self._env.close()


def _make_env_fns(
    *,
    suite,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
    control_mode: str,
) -> list[Callable[[], LiberoEnv]]:
    """Build n_envs factory callables for a single (suite, task_id)."""

    def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
        local_kwargs = dict(kwargs)
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_length=episode_length,
            episode_index=episode_index,
            control_mode=control_mode,
            **local_kwargs,
        )

    fns: list[Callable[[], LiberoEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


# ---- Main API ----------------------------------------------------------------


def create_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    control_mode: str = "relative",
    episode_length: int | None = None,
    use_depth: bool = False,
    use_voxel: bool = False,
    camera_name_mapping: dict[str, str] | None = None,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized LIBERO environments with a consistent return shape.

    Returns:
        dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single suite or a comma-separated list of suites.
        - You may pass `task_ids` (list[int]) inside `gym_kwargs` to restrict tasks per suite.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)  # optional: limit to specific tasks
    # Add use_depth, use_voxel, and camera_name_mapping to gym_kwargs so they get passed to LiberoEnv.__init__
    gym_kwargs["use_depth"] = use_depth
    gym_kwargs["use_voxel"] = use_voxel
    if camera_name_mapping is not None:
        gym_kwargs["camera_name_mapping"] = camera_name_mapping

    camera_names = _parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    print(
        f"Creating LIBERO envs | suites={suite_names} | n_envs(per task)={n_envs} | init_states={init_states}"
    )
    if task_ids_filter is not None:
        print(f"Restricting to task_ids={task_ids_filter}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        total = len(suite.tasks)
        selected = _select_task_ids(total, task_ids_filter)
        if not selected:
            raise ValueError(f"No tasks selected for suite '{suite_name}' (available: {total}).")

        for tid in selected:
            fns = _make_env_fns(
                suite=suite,
                episode_length=episode_length,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
                control_mode=control_mode,
            )
            out[suite_name][tid] = env_cls(fns)
            print(f"Built vec env | suite={suite_name} | task_id={tid} | n_envs={n_envs}")

    # return plain dicts for predictability
    return {suite: dict(task_map) for suite, task_map in out.items()}
