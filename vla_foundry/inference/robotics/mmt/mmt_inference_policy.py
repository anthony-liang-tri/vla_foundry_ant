#!/usr/bin/env python3
"""
MMT ZZK API Inference Script for VLA Foundry robot policy.

This script runs a trained model on a robot controlled via the ZZK API.
The ZZK API communicates with robots running on Isaac Sim or real hardware.

Usage:
    # Use the shell script (recommended - contains all default settings)
    bash examples/deployment/run_mmt_inference_policy.sh

    # Or run directly with all required arguments
    python vla_foundry/inference/robotics/mmt/mmt_inference_policy.py \
        --checkpoint_dir experiments/2026_02_25-18_44_27-model_diffusion_policy-lr_0.0005-bsz_8 \
        --zzk_api_client_path /path/to/zzk_api_client_dir
"""

import argparse
import copy
import logging
import os
import signal
import sys
import time
from collections import deque

import numpy as np
import torch

from vla_foundry.data.preprocessing.image_utils import ImageResizingMethod, resize_and_crop_image
from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.file_utils import get_latest_checkpoint, load_ema_checkpoint, load_model_checkpoint, yaml_load
from vla_foundry.inference.robotics.mmt.action_handlers import (
    MmtActionMapper,
    is_position_action_field,
    load_lowdim_index_selection,
)
from vla_foundry.inference.robotics.mmt.field_layouts import (
    MMT_FIELD_LAYOUTS,
    ZZK_STATE_COLS,
    ZZK_STATE_ROWS,
    build_runtime_layouts,
)
from vla_foundry.logger import setup_logging
from vla_foundry.models import create_model
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml


class ZzkPolicyInference:
    """Runs policy inference using ZZK API for robot control."""

    def __init__(
        self,
        checkpoint_directory: str,
        checkpoint_name: str | None = None,
        robot_hostname: str = "localhost",
        robot_port: int = 8888,
        device: str = "cuda",
        num_flow_steps: int = 10,
        open_loop_steps: int = 4,
        language_instruction: str = "You are a helpful robot assistant finishing tasks to help people's daily lives.",
        zzk_api_client_path: str | None = None,
        enable_compliance: bool = False,
    ):
        """Initialize the inference system.

        Args:
            checkpoint_directory: Path to the model checkpoint directory
            robot_hostname: Hostname or IP of the robot (e.g., 'localhost', 'zzk0.local')
            robot_port: Port number for ZZK API (default: 8888)
            checkpoint_name: Specific checkpoint to load (default: latest)
            device: Device to run model on ('cuda' or 'cpu')
            num_flow_steps: Number of diffusion denoising steps
            open_loop_steps: Number of steps to execute before replanning
            TODO: Enable to change language_instruction every episode
            language_instruction: Task instruction for the robot
            zzk_api_client_path: Path to directory containing zzk_api_client.py and ctypes .so library
        """
        self.checkpoint_directory = checkpoint_directory
        self.robot_hostname = robot_hostname
        self.robot_port = robot_port
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_flow_steps = num_flow_steps
        self.open_loop_steps = open_loop_steps
        self.language_instruction = language_instruction
        self.zzk_api_client_path = zzk_api_client_path
        self.enable_compliance = enable_compliance

        # Load model configuration
        self.model_config_path = os.path.join(checkpoint_directory, "config.yaml")
        self.cfg = load_experiment_params_from_yaml(
            self.model_config_path, localize_params=not self.model_config_path.startswith("s3://")
        )

        # Determine checkpoint to load (EMA if enabled)
        self.ema_enabled = self.cfg.ema.enabled
        if checkpoint_name is None or checkpoint_name == "":
            checkpoint_name = get_latest_checkpoint(checkpoint_directory)
            if not checkpoint_name:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_directory}")
            checkpoint_name = os.path.basename(checkpoint_name)

        if not checkpoint_name.endswith(".pt"):
            checkpoint_name = f"{checkpoint_name}.pt"

        # Use EMA checkpoint if enabled in config
        if self.ema_enabled:
            # Replace "checkpoint_" with "ema_" to get EMA checkpoint path
            checkpoint_name = checkpoint_name.replace("checkpoint_", "ema_")
            if not checkpoint_name.startswith("ema_"):
                checkpoint_name = f"ema_{checkpoint_name}"

        self.checkpoint_path = os.path.join(checkpoint_directory, "checkpoints", checkpoint_name)
        logging.info(f"Loading checkpoint: {self.checkpoint_path}")

        # Create model
        self.model = create_model(self.cfg.model, load_pretrained=False)

        # Create RoboticsProcessor for data processing
        self.robotics_processor = RoboticsProcessor.from_pretrained(checkpoint_directory)

        # Load checkpoint (EMA or regular)
        if self.ema_enabled:
            load_ema_checkpoint(self.model, self.checkpoint_path)
        else:
            load_model_checkpoint(self.model, self.checkpoint_path)

        self.model.to(self.device)
        self.model.eval()
        logging.info(f"Model loaded on {self.device}")

        # Load preprocessing config
        if checkpoint_directory.startswith("s3://"):
            preprocessing_config_path = f"{checkpoint_directory.rstrip('/')}/preprocessing_configs.yaml"
        else:
            preprocessing_config_path = os.path.join(checkpoint_directory, "preprocessing_configs.yaml")
        self.preprocessing_config = yaml_load(preprocessing_config_path)
        self.image_size = self.preprocessing_config["resize_images_size"]
        self.image_resizing_method = ImageResizingMethod(
            self.preprocessing_config.get("image_resizing_method", "center_crop").lower()
        )
        self.lowdim_index_selection = load_lowdim_index_selection(self.preprocessing_config)
        self.runtime_layouts = build_runtime_layouts(MMT_FIELD_LAYOUTS, self.preprocessing_config)

        # Get data configuration
        self.image_names = self.cfg.data.image_names
        self.action_fields = list(self.cfg.data.action_fields)
        self.proprioception_fields = list(self.cfg.data.proprioception_fields)
        self.num_past_timesteps = self.cfg.data.lowdim_past_timesteps
        self.num_future_timesteps = self.cfg.data.lowdim_future_timesteps
        self.total_timesteps = self.num_past_timesteps + 1 + self.num_future_timesteps
        self.action_mapper = MmtActionMapper(self.lowdim_index_selection, self.runtime_layouts)

        # Detect if action fields are position (pose) commands
        self.use_position_commands = any(is_position_action_field(f) for f in self.action_fields)

        # Derive unique base camera names from temporal image names (e.g. "rgb_t-1" → "rgb")
        self.camera_base_names = sorted(set(name.rsplit("_t", 1)[0] for name in self.image_names))

        logging.info(f"Camera names: {self.image_names} (base: {self.camera_base_names})")
        logging.info(f"Action fields: {self.action_fields}")
        logging.info(f"Position command mode: {self.use_position_commands}")
        logging.info(f"Timesteps: past={self.num_past_timesteps}, future={self.num_future_timesteps}")

        # Initialize ZZK API client
        if not self.zzk_api_client_path:
            raise ValueError(
                "zzk_api_client_path is required. "
                "Provide via --zzk_api_client_path CLI argument "
                "or paths.zzk_api_client_path in config file."
            )

        # Import zzk_api_client dynamically (path can be specified via config or CLI)
        sys.path.insert(0, self.zzk_api_client_path)
        logging.info(f"Added zzk_api_client path to sys.path: {self.zzk_api_client_path}")

        try:
            import zzk_api_client
        except ImportError:
            logging.error(
                "Failed to import zzk_api_client. Make sure zzk_api_client_path is correct "
                "via --zzk_api_client_path CLI argument or paths.zzk_api_client_path in config."
            )
            raise

        logging.info(f"Using zzk_api_client path: {self.zzk_api_client_path}")
        self.zzk_client = zzk_api_client.ZzkApiClient(
            robot_hostname, port=robot_port, file_path=self.zzk_api_client_path
        )
        # Re-register SIGINT handler in case the C extension reset it
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        logging.info("Connected to ZZK API")

        # Initialize buffers
        self._missing_cameras_warned: set[str] = set()
        self.reset_buffers()

    def _get_field_dim(self, field_name: str) -> int:
        # Read the configured full dimension for a lowdim field from normalization stats.
        if self.robotics_processor.normalizer and self.robotics_processor.normalizer.stats:
            return self.robotics_processor.normalizer.get_field_dimension(field_name)
        raise ValueError(f"Could not determine dimension for field '{field_name}' without normalization statistics.")

    @staticmethod
    def _apply_index_selection(values: np.ndarray, selection: list[int] | None) -> np.ndarray:
        # Select the subset of lowdim entries configured during preprocessing.
        if selection is None:
            return values
        return values[np.asarray(selection, dtype=np.int64)]

    @staticmethod
    def _build_full_eef_pose(
        status: dict,
        state: np.ndarray,
        left_pose_row: int,
        right_pose_row: int,
    ) -> np.ndarray | None:
        # Pack left/right gripper scalars and 6D poses into the training-time full EEF layout.
        left_gripper = status.get("left_gripper_position")
        right_gripper = status.get("right_gripper_position")
        if left_gripper is None or right_gripper is None:
            return None
        left_gripper = float(left_gripper)
        right_gripper = float(right_gripper)
        left_pose = state[left_pose_row, 0:6]
        right_pose = state[right_pose_row, 0:6]
        return np.concatenate([[left_gripper], left_pose, [right_gripper], right_pose]).astype(np.float32)

    def _select_field_value_or_zero(self, field_name: str, full_values: np.ndarray) -> np.ndarray:
        # Apply the preprocessing selection when present; otherwise fall back to the full field value.
        selection = self.lowdim_index_selection.get(field_name)
        if selection is not None:
            return self._apply_index_selection(full_values, selection)
        expected_dim = self._get_field_dim(field_name)
        if len(full_values) == expected_dim:
            return full_values
        logging.warning(
            "Dimension mismatch for '%s': got %s, expected %s; returning zeros.",
            field_name,
            len(full_values),
            expected_dim,
        )
        return np.zeros(expected_dim, dtype=np.float32)

    def _zero_proprioception_value(self, field_name: str) -> np.ndarray:
        # Synthesize a zero vector with the configured dimension for missing runtime inputs.
        return np.zeros(self._get_field_dim(field_name), dtype=np.float32)

    def _get_optional_field(self, status: dict, key: str, expected_size: int) -> np.ndarray | None:
        # Read an optional vector field from ZZK status when available and well-formed.
        if key not in status or status[key] is None:
            return None
        values = np.asarray(status[key], dtype=np.float32).reshape(-1)
        if values.size != expected_size:
            logging.warning("Unexpected size for %s: got %s, expected %s.", key, values.size, expected_size)
            return None
        return values

    @staticmethod
    def _parse_zzk_state(status: dict) -> np.ndarray | None:
        # Parse the 11x6 state matrix from ZZK status.
        if "state" not in status:
            logging.warning("No state in ZZK status")
            return None
        state = status["state"]
        if state.size == 0:
            logging.warning("Empty state in ZZK status")
            return None
        expected_size = ZZK_STATE_ROWS * ZZK_STATE_COLS
        if state.size != expected_size:
            logging.warning(
                "Unexpected state size in ZZK status: got %s, expected %s.",
                state.size,
                expected_size,
            )
            return None
        return state.reshape(ZZK_STATE_ROWS, ZZK_STATE_COLS)

    def _extract_field_from_zzk(
        self,
        field_name: str,
        state: np.ndarray | None,
        status: dict,
    ) -> np.ndarray | None:
        # Extract a proprioception field from ZZK status using the layout's zzk_source definition.
        layout = self.runtime_layouts.get(field_name)
        if layout is None or "zzk_source" not in layout:
            return None
        source = layout["zzk_source"]

        if source["type"] == "remap_slice":
            parent_values = self._extract_field_from_zzk(source["parent"], state, status)
            if parent_values is None:
                return None
            return parent_values[np.asarray(source["indices"], dtype=np.int64)]

        if source["type"] == "eef_pose":
            if state is None:
                return None
            return self._build_full_eef_pose(
                status,
                state,
                source["left_pose_row"],
                source["right_pose_row"],
            )

        if source["type"] == "single_arm_pose":
            if state is None:
                return None
            gripper = status.get(f"{source['side']}_gripper_position")
            if gripper is None:
                return None
            gripper = float(gripper)
            pose = state[source["pose_row"], 0:6]
            return np.concatenate([[gripper], pose]).astype(np.float32)

        if source["type"] == "state_row":
            if state is None:
                return None
            return state[source["row"]].astype(np.float32)

        if source["type"] == "status_field":
            return self._get_optional_field(status, source["key"], source["expected_size"])

        if source["type"] == "status_field_slice":
            full = self._get_optional_field(status, source["key"], source["end"])
            if full is None:
                return None
            return full[source["start"] : source["end"]]

        logging.warning("Unknown zzk_source type '%s' for field '%s'.", source["type"], field_name)
        return None

    def reset_buffers(self):
        """Reset observation and action buffers."""
        self.image_buffer = deque(maxlen=self.num_past_timesteps + 1)
        self.action_buffer = deque(maxlen=self.total_timesteps)
        self.proprioception_buffer = deque(maxlen=self.num_past_timesteps + 1)

        # Initialize with dummy data
        for _ in range(self.num_past_timesteps + 1):
            self.image_buffer.append({})
            self.proprioception_buffer.append({})

        for _ in range(self.total_timesteps):
            self.action_buffer.append({})

        logging.info("Buffers reset")

    def extract_obs_from_zzk_status(self, status: dict) -> dict[str, np.ndarray]:
        """Extract observation dictionary from ZZK API status.

        Args:
            status: Status dictionary from ZZK API

        Returns:
            Dictionary with observations separated by modality:
            - "images": dict[camera_name, np.ndarray] - Image data
            - "proprioception": dict[field_name, np.ndarray] - State data
            - "timestamp": int - Observation timestamp
        """
        obs = {
            "images": {},
            "proprioception": {},
            "timestamp": status.get("timestamp", 0),
        }

        # Extract images for each base camera (derived from image_names in __init__)
        for camera_name in self.camera_base_names:
            if camera_name in status:
                image = status[camera_name]
                if image is not None and image.size > 0:
                    resized_image = resize_and_crop_image(
                        image,
                        self.image_size,
                        resize_method=self.image_resizing_method,
                    )
                    obs["images"][camera_name] = np.array(resized_image)
                    logging.debug(f"Loaded image {camera_name}: shape={obs['images'][camera_name].shape}")
                else:
                    logging.warning(f"Empty or invalid image for {camera_name}")
                    obs["images"][camera_name] = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            else:
                # Warn once per missing camera to avoid log flooding; always insert
                # a zero image to keep the observation shape consistent.
                if camera_name not in self._missing_cameras_warned:
                    logging.warning(f"Camera {camera_name} not in status (may not be available)")
                    self._missing_cameras_warned.add(camera_name)
                else:
                    logging.debug(f"Camera {camera_name} still not in status; using zero image.")
                obs["images"][camera_name] = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)

        # Extract proprioception
        state = self._parse_zzk_state(status)

        for field_name in self.proprioception_fields:
            value = self._extract_field_from_zzk(field_name, state, status)
            if value is None:
                logging.warning("Missing proprioception field '%s' in ZZK status; using zeros.", field_name)
                obs["proprioception"][field_name] = self._zero_proprioception_value(field_name)
            else:
                obs["proprioception"][field_name] = self._select_field_value_or_zero(field_name, value)

        return obs

    def collect_images_from_buffer(self) -> dict[str, np.ndarray]:
        """Collect images from buffer with temporal ordering.

        Returns:
            Dictionary mapping image_name to image array
        """
        images_dict = {}
        image_buffer_list = list(self.image_buffer)

        for image_name in self.image_names:
            # Parse image name: camera_name_t<timestep>
            try:
                camera_base, timestep_str = image_name.rsplit("_t", 1)
                temporal_offset = int(timestep_str)
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Invalid image name format: '{image_name}'. "
                    f"Expected format: 'camera_name_t<timestep>' (e.g., 'rgb_t-1', 'rgb_t0'). "
                    f"Check your data configuration: cfg.data.image_names"
                ) from e

            # Calculate buffer index from temporal offset
            buffer_idx = len(image_buffer_list) - 1 + temporal_offset

            # Get image from buffer at correct temporal position
            if 0 <= buffer_idx < len(image_buffer_list):
                image_obs = image_buffer_list[buffer_idx]
                if camera_base in image_obs and image_obs[camera_base] is not None:
                    images_dict[image_name] = image_obs[camera_base]
                    logging.debug(f"Loaded {image_name} from buffer[{buffer_idx}] using camera '{camera_base}'")
                else:
                    images_dict[image_name] = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
                    logging.warning(f"Missing camera '{camera_base}' in buffer[{buffer_idx}], using zero image")
            else:
                images_dict[image_name] = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
                logging.debug(f"Buffer index {buffer_idx} out of range [0, {len(image_buffer_list)}), using zero image")

        return images_dict

    def collect_proprioception_from_buffer(self) -> dict[str, list[np.ndarray]]:
        """Collect proprioception from buffer.

        Returns:
            Dictionary mapping field_name to list of arrays
        """
        proprioception = {}
        for field in self.proprioception_fields:
            proprioception[field] = []
            for obs in self.proprioception_buffer:
                if field in obs and obs[field] is not None:
                    proprioception[field].append(obs[field])

            # Fill with zeros if needed
            if len(proprioception[field]) > 0:
                field_dim = len(proprioception[field][0])
            else:
                raise RuntimeError(
                    f"Proprioception field '{field}' has no data in buffer. Buffer may not be initialized properly."
                )

            while len(proprioception[field]) < (self.num_past_timesteps + 1):
                proprioception[field].append(np.zeros(field_dim, dtype=np.float32))

        return proprioception

    def collect_actions_from_buffer(self) -> dict[str, list[np.ndarray]]:
        """Collect actions from buffer.

        Returns:
            Dictionary mapping field_name to list of arrays
        """

        actions = {}
        for field in self.action_fields:
            actions[field] = []
            for action in self.action_buffer:
                if field in action and action[field] is not None:
                    actions[field].append(action[field])

            # Fill with zeros if needed
            if len(actions[field]) > 0:
                field_dim = len(actions[field][0])
            else:
                field_dim = self._get_field_dim(field)
                logging.debug(
                    f"Action field '{field}' has no data in buffer (first step), "
                    f"using field_dim={field_dim} from normalization stats"
                )

            while len(actions[field]) < self.total_timesteps:
                actions[field].append(np.zeros(field_dim, dtype=np.float32))

        return actions

    def build_lowdim_dict(
        self, actions: dict[str, list[np.ndarray]], proprioception: dict[str, list[np.ndarray]]
    ) -> dict[str, torch.Tensor]:
        """Build lowdim dictionary from actions and proprioception.

        Args:
            actions: Dictionary of action fields
            proprioception: Dictionary of proprioception fields

        Returns:
            Dictionary with torch tensors
        """
        lowdim_dict = {}

        # Add action fields (full timesteps: past + current + future)
        for field in self.action_fields:
            if field in actions and actions[field]:
                lowdim_dict[field] = torch.tensor(np.stack(actions[field]), dtype=torch.float32)

        # Add proprioception fields. If a field shares a key with an action,
        # overwrite the first (past + current) timesteps with actual
        # observation values so proprioception reflects the real robot state.
        for field in self.proprioception_fields:
            if field in proprioception and proprioception[field]:
                prop_tensor = torch.tensor(np.stack(proprioception[field]), dtype=torch.float32)
                if field in lowdim_dict:
                    # Overwrite the first num_past+1 timesteps with observation.
                    num_prop = prop_tensor.shape[0]
                    lowdim_dict[field][:num_prop] = prop_tensor
                else:
                    lowdim_dict[field] = prop_tensor

        return lowdim_dict

    def prepare_model_input(self) -> dict[str, torch.Tensor]:
        """Prepare model input from buffered observations.

        Returns:
            Dictionary with model input tensors
        """
        # Collect data from buffers
        images_dict = self.collect_images_from_buffer()
        proprioception = self.collect_proprioception_from_buffer()
        actions = self.collect_actions_from_buffer()

        # Build lowdim dictionary
        lowdim_dict = self.build_lowdim_dict(actions, proprioception)

        # Create processor input
        processor_input = {
            "images": [images_dict],
            "lowdim": [lowdim_dict],
            "language_instruction": [self.language_instruction],
            "metadata": [
                {
                    "anchor_relative_idx": self.num_past_timesteps,
                    "original_anchor_relative_idx": self.num_past_timesteps,
                }
            ],
        }

        # Process inputs
        processed = self.robotics_processor.process_inputs(processor_input, image_names=self.image_names)
        processed = self.robotics_processor.add_action_and_proprioception_fields(
            processed, action_fields=self.action_fields, proprioception_fields=self.proprioception_fields
        )

        # Create past mask
        past_mask = torch.zeros(1, self.total_timesteps, dtype=torch.bool)
        past_mask[:, : self.num_past_timesteps] = True
        processed["past_mask"] = past_mask

        return processed

    def generate_actions(self) -> np.ndarray:
        """Generate actions using the model.

        Returns:
            Action sequence as numpy array
        """
        model_input = self.prepare_model_input()

        # Move to device
        for key in model_input:
            if isinstance(model_input[key], torch.Tensor):
                model_input[key] = model_input[key].to(self.device)

        # Filter to tensor keys only to avoid forwarding non-model fields (e.g. raw images, metadata)
        model_input_tensors = {k: v for k, v in model_input.items() if isinstance(v, torch.Tensor)}

        # Generate actions
        with torch.no_grad():
            model_output = self.model.generate_actions(
                **model_input_tensors,
                num_inference_steps=self.num_flow_steps,
            )

        # Denormalize actions - need to denormalize each field separately
        # model_output is the normalized action tensor [B, T, D]
        # We need to split it by action fields and denormalize each one
        if self.robotics_processor.normalizer and self.robotics_processor.normalizer.enabled:
            # Split the action tensor by field dimensions
            action_start_idx = 0
            denormalized_parts = []

            for field_name in self.action_fields:
                field_dim = self.robotics_processor.normalizer.get_field_dimension(field_name)
                field_actions = model_output[:, :, action_start_idx : action_start_idx + field_dim]

                # Denormalize this field
                denorm_field = self.robotics_processor.normalizer.denormalize_tensor(
                    field_actions.cpu(), field_name, anchor_timestep=self.num_past_timesteps
                )
                denormalized_parts.append(denorm_field)
                action_start_idx += field_dim

            # Concatenate denormalized parts
            denormalized_actions = torch.cat(denormalized_parts, dim=-1)
        else:
            denormalized_actions = model_output.cpu()

        return denormalized_actions.numpy()

    def update_action_buffer(self, predicted_actions: np.ndarray):
        """Update action buffer with predicted actions.

        Args:
            predicted_actions: Predicted action sequence [batch, timesteps, action_dim]
        """
        # Extract actions from the prediction (batch=0)
        actions_np = predicted_actions[0]  # [timesteps, action_dim]

        # Clear buffer and replace with new predictions
        self.action_buffer.clear()

        for t in range(self.total_timesteps):
            action_dict = {}
            action_start_idx = 0
            for field_name in self.action_fields:
                field_dim = self._get_field_dim(field_name)
                action_dict[field_name] = actions_np[t, action_start_idx : action_start_idx + field_dim]
                action_start_idx += field_dim

            self.action_buffer.append(action_dict)

        logging.debug(f"Updated action buffer with {self.total_timesteps} predicted actions")

    def step_observations(self, zzk_status: dict) -> None:
        """Update observation buffers with new ZZK status.

        Args:
            zzk_status: Status dictionary from ZZK API
        """
        # Extract observations from ZZK status
        obs = self.extract_obs_from_zzk_status(zzk_status)

        # Update buffers with appropriate modality data
        self.image_buffer.append(obs["images"])
        self.proprioception_buffer.append(obs["proprioception"])

        logging.debug(
            f"Stepped observations: images={len(self.image_buffer)}, proprioception={len(self.proprioception_buffer)}"
        )

    def step_action(self) -> dict:
        """Get current action from buffer, shift buffer, and convert to ZZK format.

        Returns:
            ZZK action dictionary
        """
        # Get current action (at num_past_timesteps index)
        current_action_dict = self.action_buffer[self.num_past_timesteps]

        # Shift buffer: remove oldest, replicate last action
        last_action = copy.deepcopy(self.action_buffer[-1])
        self.action_buffer.popleft()
        self.action_buffer.append(last_action)

        # Convert to ZZK format
        zzk_action = {}
        debug_parts = []
        unsupported_fields = []
        for field_name in self.action_fields:
            if field_name not in current_action_dict:
                continue

            handler = self.action_mapper.action_field_handlers.get(field_name)
            if handler is None:
                unsupported_fields.append(field_name)
                continue

            action_values = np.asarray(current_action_dict[field_name], dtype=np.float32)
            handler(action_values, zzk_action, debug_parts)

        if unsupported_fields:
            logging.warning("Ignoring unsupported action fields for ZZK command conversion: %s", unsupported_fields)

        if self.enable_compliance:
            for arm_key in ("left_arm", "right_arm", "left_arm_at_gripper_tip", "right_arm_at_gripper_tip"):
                if arm_key in zzk_action:
                    padding = 12 - len(zzk_action[arm_key])
                    if padding > 0:
                        zzk_action[arm_key] = zzk_action[arm_key] + [0.0] * padding

        if debug_parts:
            logging.debug("Step action: %s", ", ".join(debug_parts))
        return zzk_action

    def step_position_action(self) -> list[tuple[dict, str, dict]]:
        """Get current action from buffer as position commands.

        Returns:
            List of (position_action_dict, tcp, gripper_action) tuples.
            Both dicts are merged and sent together via send_position_command.
            position_action_dict carries the arm pose and gripper_action carries
            the gripper target as a scalar; the server routes it to the correct
            joint axis using its kinematic model.
        """
        current_action_dict = self.action_buffer[self.num_past_timesteps]

        # Shift buffer
        last_action = copy.deepcopy(self.action_buffer[-1])
        self.action_buffer.popleft()
        self.action_buffer.append(last_action)

        commands = []
        for field_name in self.action_fields:
            if field_name not in current_action_dict:
                continue
            action_values = np.asarray(current_action_dict[field_name], dtype=np.float32)
            if is_position_action_field(field_name):
                pos_cmd, tcp, gripper_cmd = self.action_mapper.build_position_command(field_name, action_values)
                commands.append((pos_cmd, tcp, gripper_cmd))
            else:
                logging.warning("Non-position field '%s' in position mode; skipping.", field_name)
        return commands

    def _wait_for_motion_complete(self, timeout: float = 1.0, poll_interval: float = 0.05) -> dict:
        """Poll ZZK status until motion completes, keeping observation buffers fresh.

        Checks both navigation_running (position commands) and
        parts_actions_running (velocity commands).

        Returns the latest zzk_status.
        """
        start = time.time()
        while time.time() - start < timeout:
            zzk_status = self.zzk_client.receive_status()
            self.step_observations(zzk_status)
            if not self.zzk_client.navigation_running and not self.zzk_client.parts_actions_running:
                return zzk_status
            time.sleep(poll_interval)
        logging.warning("Motion did not complete within %.1fs timeout", timeout)
        return zzk_status

    def run_episode(self, max_steps: int = 500) -> bool:
        """Run a single episode.

        Args:
            max_steps: Maximum number of steps per episode

        Returns:
            True if episode completed successfully
        """
        logging.info(f"Starting episode with max_steps={max_steps}")
        self.reset_buffers()

        if self.use_position_commands:
            return self._run_episode_position(max_steps)
        return self._run_episode_velocity(max_steps)

    def _run_episode_velocity(self, max_steps: int) -> bool:
        """Run episode with velocity commands at fixed control rate."""
        control_period = 0.2  # 5Hz
        for step in range(max_steps):
            step_start = time.time()

            # Get robot status
            zzk_status = self.zzk_client.receive_status()

            # Update observation buffers
            self.step_observations(zzk_status)

            # Replan every open_loop_steps
            if step % self.open_loop_steps == 0:
                logging.info(f"Step {step}: Generating new actions...")
                predicted_actions = self.generate_actions()
                self.update_action_buffer(predicted_actions)

            # Get current action from buffer and convert to ZZK format
            zzk_action = self.step_action()
            logging.info(f"Step {step} zzk_action: {zzk_action}")

            # Sleep to maintain control rate, then send action
            sleep_time = max(0.0, control_period - (time.time() - step_start))
            time.sleep(sleep_time)

            observation_timestamp = zzk_status.get("timestamp", 0)
            tcp = "gripper_tip" if any("at_gripper_tip" in f for f in self.action_fields) else "arm_tip"
            self.zzk_client.send_command([zzk_action], observation_timestamp, tool_center_point=tcp)

        logging.info("Episode completed")
        return True

    def _run_episode_position(self, max_steps: int) -> bool:
        """Run episode with position commands, waiting for each to complete."""
        # Fill observation buffer before first action to avoid zero-image predictions.
        logging.info("Warming up observation buffer (%d frames)...", self.num_past_timesteps + 1)
        for i in range(self.num_past_timesteps + 1):
            zzk_status = self.zzk_client.receive_status()
            self.step_observations(zzk_status)
            logging.info("Warmup frame %d/%d received.", i + 1, self.num_past_timesteps + 1)

        step = 0
        while step < max_steps:
            # Get fresh observation
            zzk_status = self.zzk_client.receive_status()
            self.step_observations(zzk_status)

            # Generate new action chunk
            logging.info(f"Step {step}: Generating new actions...")
            predicted_actions = self.generate_actions()
            self.update_action_buffer(predicted_actions)

            # Execute each action in the chunk sequentially
            for _ in range(self.open_loop_steps):
                if step >= max_steps:
                    break

                # Get fresh observation before each action
                zzk_status = self.zzk_client.receive_status()
                self.step_observations(zzk_status)

                # Get position command from buffer
                commands = self.step_position_action()
                if not commands:
                    logging.warning("No position commands at step %d; skipping.", step)
                    step += 1
                    continue

                # Send all position commands and record each target for the wait.
                sent_targets: list[tuple[np.ndarray, int]] = []
                for pos_cmd, tcp, gripper_cmd in commands:
                    # In position mode, gripper commands are sent as part of
                    # the position payload as a scalar target; the server
                    # routes it to the correct joint axis using its kinematic
                    # model.
                    combined_pos_cmd = {**pos_cmd, **gripper_cmd}
                    logging.info(f"Step {step} position_cmd: {combined_pos_cmd} tcp={tcp}")
                    self.zzk_client.send_position_command(combined_pos_cmd, tool_center_point=tcp)

                    arm_key = next(iter(pos_cmd))
                    target_xyz = np.array(pos_cmd[arm_key]["pose"][:3])
                    ref_frame = pos_cmd[arm_key].get("reference_frame", 0)
                    side = "left" if "left" in arm_key else "right"
                    if tcp == "gripper_tip":
                        state_row = 9 if side == "left" else 10
                        if ref_frame == 0:
                            state_row = 7 if side == "left" else 8
                    else:
                        state_row = 5 if side == "left" else 6
                        if ref_frame == 0:
                            state_row = 2 if side == "left" else 4
                    sent_targets.append((target_xyz, state_row))

                # Wait for all arms to reach their targets (pose-based, not flag-based).
                for _ in range(20):  # max ~1s (20 * 50ms)
                    zzk_status = self.zzk_client.receive_status()
                    self.step_observations(zzk_status)
                    state = np.array(zzk_status["state"]).reshape(-1, 6)
                    max_dist = max(np.linalg.norm(t_xyz - state[row, :3]) for t_xyz, row in sent_targets)
                    if max_dist < 0.005:  # 5mm threshold
                        break
                    time.sleep(0.05)

                step += 1

        logging.info("Episode completed")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Run policy inference via ZZK API", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to model checkpoint directory")

    # Model settings
    parser.add_argument(
        "--checkpoint_name", type=str, default=None, help="Specific checkpoint name (default: auto-detect latest)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num_flow_steps", type=int, default=10, help="Number of diffusion denoising steps")
    parser.add_argument("--open_loop_steps", type=int, default=4, help="Number of steps to execute before replanning")

    # Robot settings
    parser.add_argument("--robot_hostname", type=str, default="localhost", help="Robot hostname or IP")
    parser.add_argument("--robot_port", type=int, default=8888, help="Robot port for ZZK API")

    # Task settings
    parser.add_argument(
        "--language_instruction",
        type=str,
        default="You are a helpful robot assistant finishing tasks to help people's daily lives.",
        help="Task instruction for the robot",
    )
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--max_steps_per_episode", type=int, default=500, help="Maximum steps per episode")

    # ZZK API path
    parser.add_argument(
        "--zzk_api_client_path",
        type=str,
        required=True,
        help="Path to directory containing zzk_api_client.py and ctypes .so library",
    )

    # Compliance
    parser.add_argument(
        "--enable_compliance", action="store_true", default=False, help="Enable compliance mode for arm actions"
    )

    # Logging
    parser.add_argument(
        "--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=None, level=getattr(logging, args.log_level.upper()))

    logging.info(f"{'=' * 60}")
    logging.info("MMT ZZK Policy Inference")
    logging.info(f"{'=' * 60}")
    logging.info(f"Checkpoint: {args.checkpoint_dir}")
    logging.info(f"Robot: {args.robot_hostname}:{args.robot_port}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Episodes: {args.num_episodes} x {args.max_steps_per_episode} steps")
    logging.info(f"{'=' * 60}")

    # Create inference system
    inference_system = ZzkPolicyInference(
        checkpoint_directory=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        robot_hostname=args.robot_hostname,
        robot_port=args.robot_port,
        device=args.device,
        num_flow_steps=args.num_flow_steps,
        open_loop_steps=args.open_loop_steps,
        language_instruction=args.language_instruction,
        zzk_api_client_path=args.zzk_api_client_path,
        enable_compliance=args.enable_compliance,
    )

    # Run episodes
    num_success = 0
    for episode in range(args.num_episodes):
        logging.info(f"\n{'=' * 50}")
        logging.info(f"Episode {episode + 1}/{args.num_episodes}")
        logging.info(f"{'=' * 50}")

        success = inference_system.run_episode(max_steps=args.max_steps_per_episode)
        if success:
            num_success += 1

        logging.info(f"Episode {episode + 1} completed. Success rate: {num_success}/{episode + 1}")

    logging.info(f"\n{'=' * 50}")
    logging.info(f"Final Results: {num_success}/{args.num_episodes} episodes successful")
    logging.info(f"Success rate: {100 * num_success / args.num_episodes:.1f}%")


if __name__ == "__main__":
    # Use the OS default SIGINT handler (immediate process termination).
    # Python's signal handlers only run between bytecode instructions, so they
    # cannot interrupt blocking C extension calls (e.g. zzk_api_client.receive_status).
    # SIG_DFL lets the kernel kill the process directly on Ctrl+C.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
