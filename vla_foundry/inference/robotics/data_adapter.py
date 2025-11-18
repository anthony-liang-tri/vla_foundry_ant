#!/usr/bin/env python3
"""
Refactored data adapter for converting between robotics pipeline format and policy format.

The refactored design separates configuration, buffering, and conversion logic so the
main adapter class focuses on orchestrating the flow between observations, the robotics
processor, and policy-facing outputs.
"""

import copy
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from vla_foundry.data.preprocessing.image_utils import resize_image
from vla_foundry.data.robotics.utils import (
    rot_6d_to_relative,
    xyz_to_relative,
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
        data_config: The data configuration.
        field_mapping_path: The path to the field mapping file that references the field names in the gym format.
        image_names: The names of the images in the observation (in the format of camera_name_t<timestep>).
        preprocessor_image_size: The size of the image preprocessor.
    """

    def __init__(
        self,
        robotics_processor,
        data_config,
        field_mapping_path: str,
        image_names: List[str],
        preprocessor_image_size: Tuple[int, int],
        num_past_timesteps: int = 1,
        num_future_timesteps: int = 14,
        image_indices: Tuple[int, ...] = (-1, 0),
    ):
        self.robotics_processor = robotics_processor
        self.data_config = data_config
        self.action_fields = list(data_config.action_fields)
        self.relative_action_fields = [field for field in self.action_fields if field.endswith("_relative")]  # noqa: E501
        self.proprioception_fields = list(data_config.proprioception_fields)
        # TODO: Add proprioception support
        # self.relative_proprioception_fields = [
        #     field for field in self.proprioception_fields if field.endswith("_relative")
        # ]
        self.num_past_timesteps = num_past_timesteps
        self.num_future_timesteps = num_future_timesteps
        self.image_indices = image_indices
        self.image_names = image_names
        self.field_mapping = ObservationMapping(field_mapping_path, image_names)
        self.action_mapping = ActionMapping(
            field_mapping_path, self.action_fields, self.robotics_processor, num_past_timesteps=num_past_timesteps
        )
        self.action_dim = self.action_mapping.action_dim
        self.language_instruction = "Do the task"

        self.preprocessor_image_size = preprocessor_image_size
        self.image_crop_size = self.data_config.augmentation.image.random_crop.shape

        self.total_action_timesteps = self.num_past_timesteps + 1 + self.num_future_timesteps

        self.action_buffer = []
        # self.proprioception_buffer = []
        self.image_buffer = []
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
        proprioception = {}
        for field in self.proprioception_fields:
            absolute_field = relative_to_absolute_map(field)
            robot_data = self.field_mapping.get_field(observation, absolute_field)
            proprioception[field] = robot_data
        for _ in range(self.num_past_timesteps + 1):
            self.proprioception_buffer.append(proprioception.copy())

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

    def reset(self, initial_observation):
        logging.debug("Resetting data adapter")
        self.action_buffer = []
        # self.proprioception_buffer = []
        self.image_buffer = []
        self.reference_initialized = False
        self.initialize_action_buffer(initial_observation)
        # self.initialize_proprioception_buffer(initial_observation)
        self.initialize_image_buffer(initial_observation)
        self.update_reference(initial_observation)
        self.past_mask = torch.zeros(1, self.num_past_timesteps + 1 + self.num_future_timesteps, dtype=torch.bool)

    def step_proprioception(self, observation) -> None:
        proprioception = {}
        for field in self.proprioception_fields:
            absolute_field = relative_to_absolute_map(field)
            robot_data = self.field_mapping.get_field(observation, absolute_field)
            proprioception[field] = robot_data
        self.proprioception_buffer.append(proprioception)
        self.proprioception_buffer.pop(0)

    def step_action(self):
        current_action_dict = copy.deepcopy(self.action_buffer[self.num_past_timesteps])
        last_action = copy.deepcopy(self.action_buffer[-1])
        self.action_buffer.pop(0)
        self.action_buffer.append(last_action)

        return self.action_mapping.create_pose_and_gripper(current_action_dict)

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

    def preprocess_images(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        logging.debug(f"Preprocessing images resize {self.preprocessor_image_size} crop {self.image_crop_size}")
        processed: Dict[str, np.ndarray] = {}
        for camera_name, image in images.items():
            resized = resize_image(image, self.preprocessor_image_size)
            cropped = center_crop(resized, self.image_crop_size[0], self.image_crop_size[1])
            if isinstance(cropped, Image.Image):
                cropped = np.array(cropped)
            processed[camera_name] = cropped
        return processed

    def step_image(self, observation) -> None:
        logging.debug("Stepping image")
        images = self.field_mapping.get_all_images(observation)
        processed_images = self.preprocess_images(images)
        self.image_buffer.append(processed_images)
        self.image_buffer.pop(0)

    def step_task(self, observation) -> None:
        logging.debug(f"Stepping task to instruction {observation.language_instruction}")
        self.language_instruction = observation.language_instruction

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
        # self.step_proprioception(observation)
        self.step_image(observation)
        self.step_task(observation)
        self.step_past_mask()

    def get_images_for_processor(self) -> Dict[str, np.ndarray]:
        logging.debug("Getting images for processor")
        images: Dict[str, np.ndarray] = {}
        for image_name in self.image_names:
            camera_name, timestep_str = image_name.rsplit("_t", 1)
            timestep = int(timestep_str)
            timestep = len(self.image_buffer) - 1 + timestep
            buffer_t = self.image_buffer[timestep]
            logging.debug(f"Getting image {camera_name} at timestep {timestep} of buffer for {image_name}")
            if camera_name in buffer_t:
                images[image_name] = buffer_t[camera_name]
        return images

    def get_lowdim_for_processor(self) -> Dict[str, torch.Tensor]:
        logging.debug("Getting lowdim for processor")
        lowdim: Dict[str, torch.Tensor] = {}

        for field in self.action_fields:
            field_values = []
            for _, action_t in enumerate(self.action_buffer):
                absolute_field = relative_to_absolute_map(field)
                field_values.append(np.asarray(action_t[absolute_field], dtype=np.float64))

            stacked = np.stack(field_values, axis=0)

            if field in self.relative_action_fields:
                absolute_actual_field = any_to_actual_map(absolute_field)
                reference = self.reference[absolute_actual_field]

                if "rot_6d" in field:
                    stacked = rot_6d_to_relative(stacked, reference)
                elif "xyz" in field or "gripper" in field:
                    stacked = xyz_to_relative(stacked, reference)
                else:
                    raise ValueError(f"Unsupported relative field type for '{field}'")

            stacked = stacked.astype(np.float32)
            if stacked.ndim == 1:
                stacked = stacked[:, None]
            if stacked.ndim != 2:
                raise ValueError(f"Expected stacked tensor to be 2D for field '{field}', got shape {stacked.shape}")
            tensor = torch.as_tensor(stacked, dtype=torch.float32)
            logging.debug(f"Stacked shape: {stacked.shape} {field}")

            lowdim[field] = tensor

        return lowdim

    def get_processor_input(self) -> Dict[str, Any]:
        logging.debug("Getting processor input")
        return {
            "images": [self.get_images_for_processor()],
            "lowdim": [self.get_lowdim_for_processor()],
            "language_instruction": [self.language_instruction],
        }

    def get_model_input(self, observation) -> Dict[str, torch.Tensor]:
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
