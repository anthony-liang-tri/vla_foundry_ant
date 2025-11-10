import logging
import os

import draccus
import torch

from vla_foundry.data.processor import get_processor
from vla_foundry.data.robotics.normalization import RoboticsNormalizer
from vla_foundry.file_utils import json_load
from vla_foundry.params.data_params import RoboticsDataParams


class RoboticsProcessor:
    """
    This class handles tokenization and normalization of robotics data.
    It also handles image loading and processing.
    """

    def __init__(self, data_params: RoboticsDataParams):
        self.data_params = data_params
        self.vlm_processor = get_processor(data_params)

        # Normalize contained entirely within the processor
        statistics_entries = [json_load(stats_path) for stats_path in data_params.dataset_statistics]
        if self.data_params.normalization.enabled:
            self.normalizer = RoboticsNormalizer(
                normalization_params=self.data_params.normalization,
                statistics_data=statistics_entries,
            )
        else:
            self.normalizer = None

    def save(self, experiment_path: str):
        with open(os.path.join(experiment_path, "config_processor.yaml"), "w") as f:
            draccus.dump(self.data_params, f)

    @classmethod
    def load(cls, config_path: str):
        return cls(RoboticsDataParams.from_file(config_path))

    @classmethod
    def from_pretrained(cls, config_path: str):
        return cls(RoboticsDataParams.from_file(os.path.join(config_path, "config_processor.yaml")))

    def add_action_and_proprioception_fields(self, batch, action_fields=None, proprioception_fields=None):
        # Pre-extract concatenated actions if action fields are provided
        if action_fields:
            action_data = []
            for key in action_fields:
                if key in batch["lowdim"]:
                    action_data.append(batch["lowdim"][key])
                else:
                    raise KeyError(f"Action field '{key}' missing from lowdim data")

            batch["actions"] = torch.cat(action_data, dim=-1)  # [B, T, D]

        if proprioception_fields:
            proprioception_data = []
            num_past_steps = batch.get("metadata", [{}])[0].get("anchor_relative_idx", 0)
            if num_past_steps > 0:
                for key in proprioception_fields:
                    proprioception_data.append(batch["lowdim"][key][:, :num_past_steps])
                batch["proprioception"] = torch.cat(proprioception_data, dim=-1)

        return batch

    def apply_chat_template(self, num_images, instruction):
        """
        Wrapper around vlm_processor's HF apply_chat_template method.
        Takes in number of images and instruction and adds keywords like <image> or <human></human> to the instruction.
        """
        # Apply chat template if available
        if self.vlm_processor.chat_template:
            content = [{"type": "image"} for _ in range(num_images)]
            content.append({"type": "text", "text": instruction})
            messages = [{"role": "user", "content": content}]
            instruction = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        elif self.vlm_processor.tokenizer and self.vlm_processor.tokenizer.chat_template:
            content = [{"type": "image"} for _ in range(num_images)]
            content.append({"type": "text", "text": instruction})
            messages = [{"role": "user", "content": content}]
            instruction = self.vlm_processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # No chat template support, use instruction as-is
            # Add image tokens for PaliGemma processor if we have images
            if num_images > 0:
                image_tokens = "<image> " * num_images
                instruction = image_tokens + instruction

        return instruction

    def process_inputs(self, batch, image_names, max_text_seq_len=None):
        """Tokenizes the text and converts the image to pixel_values
        Args:
            batch: Batch of samples to convert to tensors.
            image_names: Automatically generated from camera_names and image_indices in the data_params.
        """
        batch_text, batch_images = [], []
        for sample_images, instruction in zip(batch["images"], batch["language_instruction"], strict=False):
            if image_names is None or len(image_names) == 0:
                image_names = list(sample_images.keys())
                logging.warning(
                    "WARNING: Using sample_images.keys() to detect camera names. No guarantee of consistent ordering."
                    f"Sample keys: {list(sample_images.keys())}"
                )
            sample_images = [sample_images[k] for k in image_names if k in sample_images]
            instruction = self.apply_chat_template(len(sample_images), instruction)

            batch_text.append(instruction)
            if len(sample_images) > 0:
                batch_images.append(sample_images)

        # If no images, set batch_images to None
        if len(batch_images) == 0:
            batch_images = None

        # Run processor on entire batch
        processed = self.vlm_processor(images=batch_images, text=batch_text, padding=True, return_tensors="pt")

        processed_batch = batch.copy()
        processed_batch["input_ids"] = processed["input_ids"]
        processed_batch["attention_mask"] = processed["attention_mask"]
        if "pixel_values" in processed:
            c, h, w = processed["pixel_values"].shape[-3:]
            processed_batch["pixel_values"] = processed["pixel_values"].reshape(len(batch_images), -1, c, h, w)
        processed_batch["camera_names"] = self.data_params.camera_names
        processed_batch["images"] = batch_images
        processed_batch["lowdim"] = {}
        for k in batch["lowdim"][0]:
            if isinstance(batch["lowdim"][0][k][0], str):
                continue
            values = [sample_lowdim[k] for sample_lowdim in batch["lowdim"]]
            processed_batch["lowdim"][k] = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in values])

        # Normalize each field individually
        if self.normalizer:
            anchor_timestep = self.data_params.lowdim_past_timesteps
            # Normalize each lowdim field
            for field_name, tensor in processed_batch["lowdim"].items():
                if isinstance(tensor, torch.Tensor) and field_name in self.normalizer.include_fields:
                    processed_batch["lowdim"][field_name] = self.normalizer.normalize_tensor(
                        tensor, field_name, anchor_timestep=anchor_timestep
                    )

        return processed_batch
