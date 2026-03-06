#!/usr/bin/env python3
"""
Refactored data adapter for converting between robotics pipeline format and policy format.

The refactored design separates configuration, buffering, and conversion logic so the
main adapter class focuses on orchestrating the flow between observations, the robotics
processor, and policy-facing outputs.
"""

import copy
import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

import vla_foundry.visualizers.visualizer as vz
from vla_foundry.data.preprocessing.image_utils import ImageResizingMethod, resize_and_crop_image
from vla_foundry.data.preprocessing.utils import depth_images_to_point_cloud, depth_images_to_point_maps
from vla_foundry.data.robotics.utils import (
    calculate_relative_pose,
    pose_to_9d,
    to_pose_matrix,
)
from vla_foundry.inference.robotics.lbm_mapping import ActionMapping, ObservationMapping
from vla_foundry.inference.robotics.utils import (
    any_to_actual_map,
    center_crop,
    relative_to_absolute_map,
)


class PolicyDataAdapter:
    """Adapter for converting policy data to training format.

    It requires buffers to keep track of past observations and actions.
    It also requires to manage normalization of the data and adapting relative fields to the correct reference.
    For that, the buffers are always using the unnormalized absolute fields and the adapter is tracking the reference.
    When producing the model input format, the adapter converts absolute buffers to the relative format if needed and
    normalizes the results using the robotics processor normalizer.

    Args:
        robotics_processor: The robotics processor to process the data.
        data_config: The data configuration containing action_fields, proprioception_fields, and pose_groups.
        field_mapping_path: The path to the field mapping file that references the field names in the gym format.
        image_names: The names of the images in the observation (in the format of camera_name_t<timestep>).
        preprocessor_image_size: The size of the image preprocessor.
    """

    def __init__(
        self,
        robotics_processor,
        data_config,
        field_mapping_path: str,
        image_names: list[str],
        preprocessor_image_size: tuple[int, int],
        preprocessor_image_resize_method: ImageResizingMethod = ImageResizingMethod.CENTER_CROP,
        num_past_timesteps: int = 1,
        num_future_timesteps: int = 14,
        image_indices: tuple[int, ...] = (-1, 0),
    ):
        self.robotics_processor = robotics_processor
        self.data_config = data_config
        self.action_fields = list(data_config.action_fields)
        self.relative_action_fields = [field for field in self.action_fields if field.endswith("_relative")]
        self.proprioception_fields = list(data_config.proprioception_fields)
        self.relative_proprioception_fields = [
            field for field in self.proprioception_fields if field.endswith("_relative")
        ]
        self.pose_groups = list(data_config.pose_groups)

        # Create lookup table for fast pose group access during inference
        self.pose_group_lookup = {}
        for pose_group in self.pose_groups:
            self.pose_group_lookup[pose_group["position_key"]] = pose_group
            self.pose_group_lookup[pose_group["rotation_key"]] = pose_group
        self.num_past_timesteps = num_past_timesteps
        self.num_future_timesteps = num_future_timesteps
        self.image_indices = image_indices
        self.image_names = image_names
        self.field_mapping = ObservationMapping(field_mapping_path, image_names)
        self.action_mapping = ActionMapping(
            field_mapping_path,
            self.action_fields,
            self.robotics_processor,
            pose_groups=self.pose_groups,
            num_past_timesteps=num_past_timesteps,
        )
        self.action_dim = self.action_mapping.action_dim
        self.language_instruction = "Do the task"

        self.preprocessor_image_size = preprocessor_image_size
        self.preprocessor_image_resize_method = preprocessor_image_resize_method
        self.image_crop_size = self.data_config.augmentation.image.crop.shape

        self.total_action_timesteps = self.num_past_timesteps + 1 + self.num_future_timesteps

        self.action_buffer = []
        self.proprioception_buffer = []
        self.image_buffer = []
        self.raw_obs_buffer = []  # Store raw observations for generating point clouds
        self.reference = {}
        self.reference_initialized = False

    def initialize_action_buffer(self, observation) -> None:
        logging.debug("Initializing action buffer")
        action = {}
        for field in self.action_fields:
            # Get the absolute field name (strip _relative suffix if present)
            absolute_field = relative_to_absolute_map(field)
            # Read from the actual robot state
            actual_absolute_field = any_to_actual_map(absolute_field)
            robot_data = self.field_mapping.get_field(observation, actual_absolute_field)
            # Store with the absolute field name as key
            action[absolute_field] = np.asarray(robot_data, dtype=np.float64)

        self.action_buffer = [copy.deepcopy(action) for _ in range(self.total_action_timesteps)]

    def initialize_proprioception_buffer(self, observation) -> None:
        logging.debug("Initializing proprioception buffer")
        if not self.proprioception_fields:
            self.proprioception_buffer = []
            return

        proprioception = {}
        for field in self.proprioception_fields:
            absolute_field = relative_to_absolute_map(field)
            actual_field = any_to_actual_map(absolute_field)
            robot_data = self.field_mapping.get_field(observation, actual_field)
            proprioception[absolute_field] = np.asarray(robot_data, dtype=np.float64)

        buffer_size = self.num_past_timesteps + 1
        self.proprioception_buffer = [copy.deepcopy(proprioception) for _ in range(buffer_size)]

    def initialize_image_buffer(self, observation) -> None:
        logging.debug("Initializing image buffer")
        if len(self.image_names) == 0:
            self.image_buffer = []
            return

        images = self.field_mapping.get_all_images(observation)
        max_time = -np.inf
        min_time = np.inf
        for image_name in self.image_names:
            _, timestep_str = image_name.rsplit("_t", 1)
            timestep = int(timestep_str)
            max_time = max(max_time, timestep)
            min_time = min(min_time, timestep)

        range_images = int(max_time - min_time + 1)
        for _ in range(range_images):
            self.image_buffer.append(copy.deepcopy(images))

    def initialize_raw_obs_buffer(self, observation) -> None:
        logging.debug("Initializing raw observation buffer")
        # Store observations for generating point clouds from depth images
        # Buffer size matches image buffer size
        max_time = -np.inf
        min_time = np.inf
        for image_name in self.image_names:
            _, timestep_str = image_name.rsplit("_t", 1)
            timestep = int(timestep_str)
            max_time = max(max_time, timestep)
            min_time = min(min_time, timestep)

        range_obs = int(max_time - min_time + 1)
        for _ in range(range_obs):
            self.raw_obs_buffer.append(observation)

    def reset(self, initial_observation):
        logging.debug("Resetting data adapter")
        self.action_buffer = []
        self.proprioception_buffer = []
        self.image_buffer = []
        self.raw_obs_buffer = []
        self.reference_initialized = False
        self.initialize_action_buffer(initial_observation)
        self.initialize_proprioception_buffer(initial_observation)
        self.initialize_image_buffer(initial_observation)
        self.initialize_raw_obs_buffer(initial_observation)
        self.update_reference(initial_observation)
        self.past_mask = torch.zeros(1, self.num_past_timesteps + 1 + self.num_future_timesteps, dtype=torch.bool)

    def step_proprioception(self, observation) -> None:
        if not self.proprioception_fields:
            return

        proprioception = {}
        for field in self.proprioception_fields:
            absolute_field = relative_to_absolute_map(field)
            actual_field = any_to_actual_map(absolute_field)
            robot_data = self.field_mapping.get_field(observation, actual_field)
            proprioception[absolute_field] = np.asarray(robot_data, dtype=np.float64)
        self.proprioception_buffer.append(proprioception)
        if len(self.proprioception_buffer) > self.num_past_timesteps + 1:
            self.proprioception_buffer.pop(0)

    def step_action(self):
        current_action_dict = copy.deepcopy(self.action_buffer[self.num_past_timesteps])
        last_action = copy.deepcopy(self.action_buffer[-1])
        self.action_buffer.pop(0)
        self.action_buffer.append(last_action)

        output = self.action_mapping.create_pose_and_gripper(current_action_dict)
        vz.log_robot_gym_poses_and_grippers("current_action_arm_poses", output)
        return output

    def update_reference(self, observation):
        """
        The data adapter has buffers for the past actions and proprioception that are kept in absolute format.
        If the model needs relative fields input or if we need to interpret the relative action output of the model,
        we need to compute them using a reference observation.
        This function updates the reference with the given observation.
        """
        logging.debug("Updating reference")
        for field in self.action_fields:
            absolute_actual_field = relative_to_absolute_map(any_to_actual_map(field))
            robot_data = self.field_mapping.get_field(observation, absolute_actual_field)
            self.reference[absolute_actual_field] = np.asarray(robot_data, dtype=np.float64)
        vz.log_robot_gym_poses_and_grippers(
            "reference_arm_poses", self.field_mapping.create_pose_and_gripper(self.reference)
        )
        for field in self.proprioception_fields:
            absolute_actual_field = relative_to_absolute_map(any_to_actual_map(field))
            robot_data = self.field_mapping.get_field(observation, absolute_actual_field)
            self.reference[absolute_actual_field] = np.asarray(robot_data, dtype=np.float64)
        vz.log_robot_gym_poses_and_grippers(
            "reference_proprioception", self.field_mapping.create_pose_and_gripper(self.reference)
        )

    def preprocess_images(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        logging.debug(f"Preprocessing images resize {self.preprocessor_image_size} crop {self.image_crop_size}")
        processed: dict[str, np.ndarray] = {}
        for camera_name, image in images.items():
            resized = resize_and_crop_image(
                image,
                target_size=self.preprocessor_image_size,
                resize_method=self.preprocessor_image_resize_method,
            )
            cropped = center_crop(resized, self.image_crop_size[0], self.image_crop_size[1])
            if isinstance(cropped, Image.Image):
                cropped = np.array(cropped)
            processed[camera_name] = cropped
        return processed

    def step_image(self, observation) -> None:
        logging.debug("Stepping image")
        images = self.field_mapping.get_all_images(observation)
        vz.log_images("observation_images", images)
        processed_images = self.preprocess_images(images)
        self.image_buffer.append(processed_images)
        self.image_buffer.pop(0)

    def step_raw_obs_buffer(self, observation) -> None:
        logging.debug("Stepping raw observation buffer")
        self.raw_obs_buffer.append(observation)
        if len(self.raw_obs_buffer) > len(self.image_buffer):
            self.raw_obs_buffer.pop(0)

    def step_task(self, observation) -> None:
        logging.debug(f"Stepping task to instruction {observation.language_instruction}")
        self.language_instruction = observation.language_instruction
        vz.log_text("language_instruction", self.language_instruction)

    def step_past_mask(self) -> None:
        logging.debug("Stepping past mask")
        if self.num_past_timesteps == 0:
            return

        if self.num_past_timesteps > 1:
            # Roll the past mask to the left as we are stepping the buffer to track the real past actions in the buffer
            self.past_mask[:, : self.num_past_timesteps - 1] = self.past_mask[:, 1 : self.num_past_timesteps].clone()
        self.past_mask[:, self.num_past_timesteps - 1] = True

    def step_observations(
        self,
        observation,
    ):
        self.step_proprioception(observation)
        self.step_image(observation)
        self.step_raw_obs_buffer(observation)
        self.step_task(observation)
        self.step_past_mask()
        vz.log_robot_gym_poses_and_grippers("current_pose", observation.robot.actual)

    def get_images_for_processor(self) -> dict[str, np.ndarray]:
        logging.debug("Getting images for processor")
        images: dict[str, np.ndarray] = {}
        for image_name in self.image_names:
            camera_name, timestep_str = image_name.rsplit("_t", 1)
            timestep = int(timestep_str)
            timestep = len(self.image_buffer) - 1 + timestep
            buffer_t = self.image_buffer[timestep]
            logging.debug(f"Getting image {camera_name} at timestep {timestep} of buffer for {image_name}")
            if camera_name in buffer_t:
                images[image_name] = buffer_t[camera_name]
        processed_images = {}
        for name, image in images.items():
            image = np.asarray(image, dtype=np.float32)
            image_min = image.min()
            image_max = image.max()
            if image_max == image_min:
                processed_images[name] = np.zeros_like(image, dtype=np.uint8)
                continue
            scaled = (image - image_min) / (image_max - image_min)
            processed_images[name] = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
        vz.log_images("processed_images", processed_images)
        return images

    def get_lowdim_for_processor(self) -> dict[str, torch.Tensor]:
        logging.debug("Getting lowdim for processor")
        lowdim = self._stack_fields(
            self.action_buffer,
            self.action_fields,
            self.relative_action_fields,
        )

        if self.proprioception_fields:
            lowdim.update(
                self._stack_fields(
                    self.proprioception_buffer,
                    self.proprioception_fields,
                    self.relative_proprioception_fields,
                )
            )

        return lowdim

    def _to_relative(self, buffer: list[dict], field: str, timesteps_slice: slice | None = None) -> np.ndarray:
        """Convert absolute field to relative based on field type.

        Automatically detects if field belongs to a pose group and applies appropriate conversion.

        Args:
            buffer: Buffer containing absolute field data
            field: Field name (with _relative suffix)
            timesteps_slice: Optional slice to apply to buffer

        Returns:
            Relative values as numpy array
        """
        absolute_field = relative_to_absolute_map(field)

        # Check if this field belongs to a pose group
        pose_group = self.pose_group_lookup.get(absolute_field)
        if pose_group is not None:
            # Handle pose group fields (xyz + rot_6d)
            xyz_key = pose_group["position_key"]
            rot_6d_key = pose_group["rotation_key"]

            # Calculate relative pose
            relative_xyz, relative_rot_6d = self._pose_to_relative(buffer, xyz_key, rot_6d_key, timesteps_slice)

            # Return the appropriate component
            if absolute_field == xyz_key:
                return relative_xyz
            else:  # rot_6d_key
                return relative_rot_6d

        # Handle non-pose relative fields (e.g., joint positions)
        values = [np.asarray(entry[absolute_field], dtype=np.float64) for entry in buffer]
        if timesteps_slice is not None:
            values = values[timesteps_slice]
        stacked = np.stack(values, axis=0)

        actual_field = any_to_actual_map(absolute_field)
        reference = self.reference.get(actual_field)
        if reference is None:
            raise KeyError(f"Reference for '{actual_field}' not initialized")

        if "joint_position" in field:
            return self._wrap_to_pi(stacked - reference)
        else:
            # TODO: This does not current support generic subtraction
            raise ValueError(f"Unsupported relative field type for '{field}'")

    def _stack_fields(
        self,
        buffer: list[dict],
        fields: list[str],
        relative_fields: list[str],
        timesteps_slice: slice | None = None,
    ) -> dict[str, torch.Tensor]:
        """Stack fields from buffer into tensors, applying relative conversions as needed.

        Note: If both xyz_relative and rot_6d_relative from the same pose group are in fields,
        _pose_to_relative() will be called twice. This is acceptable for code clarity - the
        alternative would require pre-processing pose groups which adds complexity. The duplicate
        calculation is minimal compared to model inference time. Consider refactoring in the future.
        """
        stacked_fields: dict[str, torch.Tensor] = {}
        if not buffer or not fields:
            return stacked_fields

        for field in fields:
            absolute_field = relative_to_absolute_map(field)

            if field in relative_fields:
                # Apply relative conversion
                stacked = self._to_relative(buffer, field, timesteps_slice)
            else:
                # Absolute field - extract and stack
                values = [np.asarray(entry[absolute_field], dtype=np.float64) for entry in buffer]
                if timesteps_slice is not None:
                    values = values[timesteps_slice]
                stacked = np.stack(values, axis=0)

            # Convert to 2D tensor
            stacked = stacked.astype(np.float32)
            if stacked.ndim == 1:
                stacked = stacked[:, None]
            if stacked.ndim != 2:
                raise ValueError(f"Expected stacked tensor to be 2D for field '{field}', got shape {stacked.shape}")

            logging.debug(f"Stacked shape: {stacked.shape} {field}")
            stacked_fields[field] = torch.as_tensor(stacked, dtype=torch.float32)

        return stacked_fields

    def _pose_to_relative(
        self, buffer: list[dict], xyz_key: str, rot_6d_key: str, timesteps_slice: slice | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert absolute pose fields to relative pose.

        Args:
            buffer: Buffer containing absolute pose data
            xyz_key: Key for position field (absolute)
            rot_6d_key: Key for rotation field (absolute)
            timesteps_slice: Optional slice to apply to buffer

        Returns:
            Tuple of (relative_xyz, relative_rot_6d) as numpy arrays
        """
        # Extract and stack values from buffer
        values_xyz = [np.asarray(entry[xyz_key], dtype=np.float64) for entry in buffer]
        values_rot = [np.asarray(entry[rot_6d_key], dtype=np.float64) for entry in buffer]
        if timesteps_slice is not None:
            values_xyz = values_xyz[timesteps_slice]
            values_rot = values_rot[timesteps_slice]
        xyz_stacked = np.stack(values_xyz, axis=0)
        rot_6d_stacked = np.stack(values_rot, axis=0)

        # Get reference pose
        xyz_actual = any_to_actual_map(xyz_key)
        rot_6d_actual = any_to_actual_map(rot_6d_key)
        reference_xyz = self.reference.get(xyz_actual)
        reference_rot_6d = self.reference.get(rot_6d_actual)

        if reference_xyz is None or reference_rot_6d is None:
            raise KeyError(f"Reference missing for pose: xyz={xyz_actual}, rot_6d={rot_6d_actual}")

        # Calculate relative pose using pose matrices
        reference_pose_matrix = to_pose_matrix(reference_xyz, reference_rot_6d)
        current_pose_matrices = to_pose_matrix(xyz_stacked, rot_6d_stacked)
        relative_pose_matrices = calculate_relative_pose(current_pose_matrices, reference_pose_matrix)
        relative_xyz, relative_rot_6d = pose_to_9d(relative_pose_matrices)

        return relative_xyz, relative_rot_6d

    @staticmethod
    def _wrap_to_pi(delta: np.ndarray) -> np.ndarray:
        return (delta + np.pi) % (2 * np.pi) - np.pi

    def get_point_cloud_for_processor(self) -> np.ndarray:
        """
        Generate point clouds from depth images in the observation buffer.
        Returns: (T, N, C) point cloud array where T=timesteps, N=num_points, C=6 (xyzrgb)
        """
        logging.debug("Generating point clouds for processor")
        if not self.data_config.use_point_cloud or not self.raw_obs_buffer:
            return None

        point_clouds_timesteps = []
        # Generate point cloud for each unique timestep needed
        # Note: image_names can have multiple cameras at the same timestep
        # Example: ["cam0_t-1", "cam1_t-1", "cam0_t0", "cam1_t0"]
        # We only need to generate point clouds once per unique timestep
        for obs_idx, image_name in enumerate(self.image_names):
            _, timestep_str = image_name.rsplit("_t", 1)
            timestep = int(timestep_str)

            # Skip if we already processed this timestep (handles multiple cameras)
            if obs_idx > 0:
                _, prev_timestep_str = self.image_names[obs_idx - 1].rsplit("_t", 1)
                prev_timestep = int(prev_timestep_str)
                if timestep == prev_timestep:
                    continue

            # Get observation for this timestep from buffer
            # raw_obs_buffer stores one entry per unique timestep, not per camera
            # Convert relative timestep (e.g., -1, 0) to absolute buffer index
            # Formula: buffer_idx = len(buffer) - 1 + relative_timestep
            # Example: buffer_len=2, timestep=-1 → buffer_idx=0, timestep=0 → buffer_idx=1
            buffer_idx = len(self.raw_obs_buffer) - 1 + timestep
            if buffer_idx < 0 or buffer_idx >= len(self.raw_obs_buffer):
                logging.warning(
                    f"Buffer index {buffer_idx} out of range for raw_obs_buffer length "
                    f"{len(self.raw_obs_buffer)} (timestep={timestep})"
                )
                buffer_idx = max(0, min(buffer_idx, len(self.raw_obs_buffer) - 1))
            observation = self.raw_obs_buffer[buffer_idx]

            # Extract depth, RGB, intrinsics, extrinsics for all cameras
            depth_images = self.field_mapping.get_all_depth_images(observation)
            if depth_images is None:
                logging.warning("No depth images available in observation, cannot generate point clouds")
                return None

            rgb_images = self.field_mapping.get_all_images(observation)
            intrinsics = self.field_mapping.get_all_intrinsics(observation)
            extrinsics = self.field_mapping.get_all_extrinsics(observation)

            # Generate point cloud for this timestep
            point_cloud = depth_images_to_point_cloud(
                depth_images=depth_images,
                rgb_images=rgb_images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_points=self.data_config.point_cloud_num_points,
                filter_ground_plane=True,
            )
            if point_cloud is None:
                logging.warning("Failed to generate point cloud for timestep")
                return None

            point_clouds_timesteps.append(point_cloud)

        # Stack along time dimension: (T, N, C)
        stacked_point_cloud = np.stack(point_clouds_timesteps, axis=0)
        return stacked_point_cloud

    def get_point_maps_for_processor(self) -> dict[str, np.ndarray]:
        """
        Generate point maps from depth images in the observation buffer.
        Returns dict with keys like "wrist_t0", "wrist_t-1" etc.
        """
        logging.debug("Generating point maps for processor")
        if not self.data_config.use_point_cloud or not self.observation_buffer:
            return None

        point_maps_dict = {}
        # Generate point maps for each image in image_names
        for obs_idx, image_name in enumerate(self.image_names):
            # Parse camera name and timestep offset from image_name
            # image_name format: "camera_name_t<offset>" e.g., "wrist_t0", "overhead_t-1"
            if "_t" not in image_name:
                continue

            camera_name, t_offset_str = image_name.rsplit("_t", 1)

            # Get observation for this timestep
            buffer_idx = obs_idx
            if buffer_idx >= len(self.observation_buffer):
                buffer_idx = len(self.observation_buffer) - 1
            observation = self.observation_buffer[buffer_idx]

            # Extract depth images and intrinsics for all cameras using field_mapping
            depth_images = self.field_mapping.get_all_depth_images(observation)
            if depth_images is None or camera_name not in depth_images:
                logging.debug(f"No depth image for {camera_name}, skipping point maps")
                continue

            intrinsics = self.field_mapping.get_all_intrinsics(observation)
            if intrinsics is None or camera_name not in intrinsics:
                logging.debug(f"No intrinsics for {camera_name}, skipping point maps")
                continue

            # Generate point map for this specific camera
            camera_depth = {camera_name: depth_images[camera_name]}
            camera_intrinsics = {camera_name: intrinsics[camera_name]}

            point_maps = depth_images_to_point_maps(
                depth_images=camera_depth,
                intrinsics=camera_intrinsics,
                depth_scale=1000.0,
                min_depth=0.001,
                max_depth=3.0,
            )

            if point_maps is None or camera_name not in point_maps:
                logging.debug(f"Failed to generate point map for {camera_name}")
                continue

            # Store with key format: camera_t<offset>
            point_maps_dict[image_name] = point_maps[camera_name]

        return point_maps_dict if point_maps_dict else None

    def get_processor_input(self) -> dict[str, Any]:
        logging.debug("Getting processor input")
        processor_input = {
            "images": [self.get_images_for_processor()],
            "lowdim": [self.get_lowdim_for_processor()],
            "metadata": [
                {
                    "anchor_relative_idx": self.num_past_timesteps,
                    "original_anchor_relative_idx": self.num_past_timesteps,
                }
            ],
            "language_instruction": [self.language_instruction],
        }

        # Add point cloud if enabled
        if self.data_config.use_point_cloud:
            point_cloud = self.get_point_cloud_for_processor()
            processor_input["point_cloud"] = [point_cloud] if point_cloud is not None else None

            # Add point maps
            point_maps = self.get_point_maps_for_processor()
            processor_input["point_maps"] = [point_maps] if point_maps is not None else None

        return processor_input

    def get_model_input(self, observation) -> dict[str, torch.Tensor]:
        logging.debug("Getting model input")
        self.update_reference(observation)
        processor_input = self.get_processor_input()
        processed = self.robotics_processor.process_inputs(processor_input, image_names=self.image_names)
        processed = self.robotics_processor.add_action_and_proprioception_fields(
            processed,
            action_fields=self.action_fields,
            proprioception_fields=self.proprioception_fields,
        )
        processed["past_mask"] = self.past_mask.clone()
        return processed

    def update_action(self, observation, model_output: torch.Tensor):
        logging.debug("Updating action buffer with fresh predictions")
        model_output = model_output.cpu()
        action_list = self.action_mapping.from_action_model(
            model_output,
            self.robotics_processor.normalizer,
            self.reference,
        )
        # Update the action buffer with the new actions from the model
        self.action_buffer = [copy.deepcopy(action) for action in action_list]
        vz.log_robot_gym_action_predictions(
            "action_predictions", [self.action_mapping.create_pose_and_gripper(action) for action in self.action_buffer]
        )
