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
        --zzk_api_client_py_path /path/to/zzk_api_client.py \
        --zzk_api_client_ctypes_library_path /path/to/zzk_api_ctypes_client.so
"""

import argparse
import copy
import logging
import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch

from vla_foundry.data.preprocessing.image_utils import resize_image
from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.file_utils import get_latest_checkpoint, load_ema_checkpoint, load_model_checkpoint, yaml_load
from vla_foundry.logger import setup_logging
from vla_foundry.models import create_model
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml


class ZzkPolicyInference:
    """Runs policy inference using ZZK API for robot control."""

    def __init__(
        self,
        checkpoint_directory: str,
        checkpoint_name: Optional[str] = None,
        robot_hostname: str = "localhost",
        robot_port: int = 8888,
        device: str = "cuda",
        num_flow_steps: int = 10,
        open_loop_steps: int = 4,
        language_instruction: str = "You are a helpful robot assistant finishing tasks to help people's daily lives.",
        zzk_api_client_py_path: Optional[str] = None,
        zzk_api_client_ctypes_library_path: Optional[str] = None,
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
            zzk_api_client_py_path: Path to directory containing zzk_api_client.py
            zzk_api_client_ctypes_library_path: Path to directory containing zzk_api_ctypes_client.so
        """
        self.checkpoint_directory = checkpoint_directory
        self.robot_hostname = robot_hostname
        self.robot_port = robot_port
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_flow_steps = num_flow_steps
        self.open_loop_steps = open_loop_steps
        self.language_instruction = language_instruction
        self.zzk_api_client_py_path = zzk_api_client_py_path
        self.zzk_api_client_ctypes_library_path = zzk_api_client_ctypes_library_path

        # Load model configuration
        self.model_config_path = os.path.join(checkpoint_directory, "config.yaml")
        self.cfg = load_experiment_params_from_yaml(
            self.model_config_path, localize_params=not self.model_config_path.startswith("s3://")
        )

        # Determine checkpoint to load (EMA if enabled)
        self.ema_enabled = self.cfg.ema.enabled
        if checkpoint_name is None or checkpoint_name == "":
            checkpoint_name = get_latest_checkpoint(checkpoint_directory)
            if checkpoint_name:
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

        # Get data configuration
        self.image_names = self.cfg.data.image_names
        self.action_fields = list(self.cfg.data.action_fields)
        self.proprioception_fields = list(self.cfg.data.proprioception_fields)
        self.num_past_timesteps = self.cfg.data.lowdim_past_timesteps
        self.num_future_timesteps = self.cfg.data.lowdim_future_timesteps
        self.total_timesteps = self.num_past_timesteps + 1 + self.num_future_timesteps

        logging.info(f"Camera names: {self.image_names}")
        logging.info(f"Action fields: {self.action_fields}")
        logging.info(f"Timesteps: past={self.num_past_timesteps}, future={self.num_future_timesteps}")

        # Initialize ZZK API client
        # Import zzk_api_client dynamically (path can be specified via config or CLI)
        if zzk_api_client_py_path:
            sys.path.insert(0, zzk_api_client_py_path)
            logging.info(f"Added zzk_api_client path to sys.path: {zzk_api_client_py_path}")

        try:
            import zzk_api_client
        except ImportError:
            logging.error(
                "Failed to import zzk_api_client. Make sure zzk_api_client_py_path is provided "
                "via --zzk_api_client_py_path CLI argument or paths.zzk_api_client_py_path in config."
            )
            raise

        # Validate ctypes library path (can be specified via config or CLI)
        if not self.zzk_api_client_ctypes_library_path:
            raise ValueError(
                "zzk_api_client_ctypes_library_path is required. "
                "Provide via --zzk_api_client_ctypes_library_path CLI argument "
                "or paths.zzk_api_client_ctypes_library_path in config file."
            )

        logging.info(f"Using ctypes library path: {self.zzk_api_client_ctypes_library_path}")
        self.zzk_client = zzk_api_client.ZzkApiClient(
            robot_hostname, port=robot_port, file_path=self.zzk_api_client_ctypes_library_path
        )
        logging.info("Connected to ZZK API")

        # Initialize buffers
        self.reset_buffers()

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

    def extract_obs_from_zzk_status(self, status: Dict) -> Dict[str, np.ndarray]:
        """Extract observation dictionary from ZZK API status.

        Args:
            status: Status dictionary from ZZK API

        Returns:
            Dictionary with observations separated by modality:
            - "images": Dict[camera_name, np.ndarray] - Image data
            - "proprioception": Dict[field_name, np.ndarray] - State data
            - "timestamp": int - Observation timestamp
        """
        obs = {
            "images": {},
            "proprioception": {},
            "timestamp": status.get("timestamp", 0),
        }

        # Extract images based on configured camera names
        # ZZK API returns: "rgb"
        # VLA model expects: "rgb_t-1", "rgb_t0" (temporal naming)

        # Map ZZK camera names to base names
        zzk_camera_map = {
            "rgb": "rgb",  # Head camera
        }

        # Process each ZZK camera and store with base name
        for zzk_image_name, model_image_base_name in zzk_camera_map.items():
            if zzk_image_name in status:
                image = status[zzk_image_name]
                if image is not None and image.size > 0:
                    # Resize image to expected size (same as training data preprocessing)
                    resized_image = resize_image(image, self.image_size)
                    # Store in images dict
                    obs["images"][model_image_base_name] = np.array(resized_image)
                    logging.debug(
                        f"Loaded image {model_image_base_name}: shape={obs['images'][model_image_base_name].shape}"
                    )
                else:
                    logging.warning(f"Empty or invalid image for {zzk_image_name}")
                    obs["images"][model_image_base_name] = np.zeros(
                        (self.image_size[1], self.image_size[0], 3), dtype=np.uint8
                    )
            else:
                logging.debug(f"Camera {zzk_image_name} not in status (may not be available)")

        # Extract proprioception (chassis_T_eef_pose)
        # ZZK state format from zzk_api_ctypes_client.cc:
        # state[0][0:6] = local_T_chassis (x, y, z, rx, ry, rz)
        # state[1][2]   = left_gripper_position
        # state[2][0:6] = chassis_T_left_arm_tip (x, y, z, rx, ry, rz)
        # state[3][2]   = right_gripper_position
        # state[4][0:6] = chassis_T_right_arm_tip (x, y, z, rx, ry, rz)
        # state[5][0:6] = chest_T_left_arm_tip (x, y, z, rx, ry, rz)
        # state[6][0:6] = chest_T_right_arm_tip (x, y, z, rx, ry, rz)
        # state[7][0:6] = chassis_T_left_gripper_tip (x, y, z, rx, ry, rz)
        # state[8][0:6] = chassis_T_right_gripper_tip (x, y, z, rx, ry, rz)
        # state[9][0:6] = chest_T_left_gripper_tip (x, y, z, rx, ry, rz)
        # state[10][0:6] = chest_T_right_gripper_tip (x, y, z, rx, ry, rz)

        if "state" in status:
            state = status["state"]
            if state.size > 0:
                state = state.reshape(11, 6)

                # Extract chassis_T_eef_pose for each arm
                # Format: [gripper, x, y, z, rx, ry, rz] (7 dimensions)

                # Left arm
                left_gripper = state[1, 2]  # state[1][2]
                left_xyz = state[2, 0:3]  # state[2][0:3]
                left_rpy = state[2, 3:6]  # state[2][3:6]
                obs["proprioception"]["chassis_T_eef_pose"] = np.concatenate(
                    [[left_gripper], left_xyz, left_rpy]
                )  # 7 dimensions: [gripper, x, y, z, rx, ry, rz]

                logging.debug(
                    f"chassis_T_eef_pose (left): gripper={left_gripper:.3f}, "
                    f"xyz=({left_xyz[0]:.3f}, {left_xyz[1]:.3f}, {left_xyz[2]:.3f}), "
                    f"rpy=({left_rpy[0]:.3f}, {left_rpy[1]:.3f}, {left_rpy[2]:.3f})"
                )
            else:
                logging.warning("Empty state in ZZK status")
                obs["proprioception"]["chassis_T_eef_pose"] = np.zeros(7, dtype=np.float32)
        else:
            logging.warning("No state in ZZK status")
            obs["proprioception"]["chassis_T_eef_pose"] = np.zeros(7, dtype=np.float32)

        return obs

    def collect_images_from_buffer(self) -> Dict[str, np.ndarray]:
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

    def collect_proprioception_from_buffer(self) -> Dict[str, List[np.ndarray]]:
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

    def collect_actions_from_buffer(self) -> Dict[str, List[np.ndarray]]:
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
                field_dim = self.cfg.data.action_dim
                logging.debug(
                    f"Action field '{field}' has no data in buffer (first step), "
                    f"using action_dim={field_dim} from config"
                )

            while len(actions[field]) < self.total_timesteps:
                actions[field].append(np.zeros(field_dim, dtype=np.float32))

        return actions

    def build_lowdim_dict(
        self, actions: Dict[str, List[np.ndarray]], proprioception: Dict[str, List[np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        """Build lowdim dictionary from actions and proprioception.

        Args:
            actions: Dictionary of action fields
            proprioception: Dictionary of proprioception fields

        Returns:
            Dictionary with torch tensors
        """
        lowdim_dict = {}

        # Add action fields
        for field in self.action_fields:
            if field in actions and actions[field]:
                lowdim_dict[field] = torch.tensor(np.stack(actions[field]), dtype=torch.float32)

        # Add proprioception fields
        for field in self.proprioception_fields:
            if field in proprioception and proprioception[field]:
                lowdim_dict[field] = torch.tensor(np.stack(proprioception[field]), dtype=torch.float32)

        return lowdim_dict

    def prepare_model_input(self) -> Dict[str, torch.Tensor]:
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

        # Generate actions
        with torch.no_grad():
            model_output = self.model.generate_actions(
                input_ids=model_input.get("input_ids"),
                pixel_values=model_input.get("pixel_values"),
                actions=model_input.get("actions"),
                attention_mask=model_input.get("attention_mask"),
                num_inference_steps=self.num_flow_steps,
                past_mask=model_input["past_mask"],
                proprioception=model_input.get("proprioception"),
            )

        # Denormalize actions - need to denormalize each field separately
        # model_output is the normalized action tensor [B, T, D]
        # We need to split it by action fields and denormalize each one
        denormalized_actions = model_output.cpu()

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
            for field_name in self.action_fields:
                if field_name == "arm_action":
                    # arm_action is 7-dimensional
                    action_dict[field_name] = actions_np[t, :7]
                # Add other action fields here if needed

            self.action_buffer.append(action_dict)

        logging.debug(f"Updated action buffer with {self.total_timesteps} predicted actions")

    def step_observations(self, zzk_status: Dict) -> None:
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

    def step_action(self) -> Dict:
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
        if "arm_action" in current_action_dict:
            arm_values = current_action_dict["arm_action"]

            # Map to ZZK format: [vx, vy, vz, wx, wy, wz] for left_arm
            zzk_action["left_arm"] = [
                arm_values[1],  # vx
                arm_values[2],  # vy
                arm_values[3],  # vz
                arm_values[4],  # wx
                arm_values[5],  # wy
                arm_values[6],  # wz
            ]
            # Gripper: [0, 0, gripper_value, 0, 0, 0]
            zzk_action["left_gripper"] = [0, 0, arm_values[0].item(), 0, 0, 0]

        logging.debug(f"Step action: vx={arm_values[1]:.3f}, vy={arm_values[2]:.3f}, gripper={arm_values[0]:.3f}")
        return zzk_action

    def run_episode(self, max_steps: int = 500) -> bool:
        """Run a single episode.

        Args:
            max_steps: Maximum number of steps per episode

        Returns:
            True if episode completed successfully
        """
        logging.info(f"Starting episode with max_steps={max_steps}")
        self.reset_buffers()

        for step in range(max_steps):
            # Get robot status
            zzk_status = self.zzk_client.receive_status()

            # Update observation buffers
            self.step_observations(zzk_status)

            # Replan every open_loop_steps
            if step % self.open_loop_steps == 0:
                logging.info(f"Step {step}: Generating new actions...")
                predicted_actions = self.generate_actions()

                # Update action buffer with new predictions
                self.update_action_buffer(predicted_actions)

            # Get current action from buffer and convert to ZZK format
            zzk_action = self.step_action()

            # Send action to robot
            observation_timestamp = zzk_status.get("timestamp", 0)
            self.zzk_client.send_command([zzk_action], observation_timestamp)

            # Sleep to match 10Hz control rate
            time.sleep(0.1)

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

    # ZZK API paths
    parser.add_argument(
        "--zzk_api_client_py_path", type=str, default=None, help="Path to directory containing zzk_api_client.py"
    )
    parser.add_argument(
        "--zzk_api_client_ctypes_library_path", type=str, required=True, help="Path to zzk_api_ctypes_client.so library"
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
        zzk_api_client_py_path=args.zzk_api_client_py_path,
        zzk_api_client_ctypes_library_path=args.zzk_api_client_ctypes_library_path,
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
    main()
