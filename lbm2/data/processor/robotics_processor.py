import os

import draccus
import torch

from lbm2.data.processor import get_processor
from lbm2.data.robotics.normalization import RoboticsNormalizer
from lbm2.file_utils import json_load
from lbm2.params.data_params import LBMDataParams


class RoboticsProcessor:
    """
    This class handles tokenization and normalization of robotics data.
    It also handles image loading and processing.
    """

    def __init__(self, data_params: LBMDataParams):
        self.data_params = data_params
        self.vlm_processor = get_processor(data_params)

        # Normalize contained entirely within the processor
        self.statistics = [json_load(s) for s in data_params.dataset_statistics]
        if self.data_params.normalization.enabled:
            self.normalizer = RoboticsNormalizer(dataset_config=self.data_params, statistics_data=self.statistics)
        else:
            self.normalizer = None

    def save(self, experiment_path: str):
        with open(os.path.join(experiment_path, "config_processor.yaml"), "w") as f:
            draccus.dump(self.data_params, f)

    @classmethod
    def load(cls, config_path: str):
        return cls(LBMDataParams.from_file(config_path))

    @classmethod
    def from_pretrained(cls, config_path: str):
        return cls(LBMDataParams.from_file(os.path.join(config_path, "config_processor.yaml")))

    def add_action_and_proprioception_fields(self, batch, action_fields=None, proprioception_fields=None):
        # Pre-extract concatenated actions if action fields are provided
        if action_fields:
            action_data = []
            for key in action_fields:
                if key in batch["lowdim"]:
                    action_data.append(batch["lowdim"][key])
            batch["actions"] = torch.cat(action_data, dim=-1)  # [B, T, D]

        if proprioception_fields:
            proprioception_data = []
            num_past_steps = batch.get("metadata", [{}])[0].get("anchor_relative_idx", 0)
            if num_past_steps > 0:
                for key in proprioception_fields:
                    proprioception_data.append(batch["lowdim"][key][:, :num_past_steps])
                batch["proprioception"] = torch.cat(proprioception_data, dim=-1)

        return batch

    def process_inputs(self, batch, num_images=None, max_text_seq_len=None):
        """Convert with padding for specific sequence fields in lowdim data too.
        Args:
            batch: Batch of samples to convert to tensors.
            processor: Processor to use for tokenization.
        """
        batch_text, batch_images = [], []
        for sample_images, instruction in zip(batch["images"], batch["language_instruction"], strict=False):
            camera_names = list(sample_images.keys())
            if num_images is not None and num_images > 0:
                camera_names = camera_names[:num_images]

            sample_images = [sample_images[k] for k in camera_names]
            sample_num_images = len(sample_images)

            # Apply chat template if available
            if self.vlm_processor.chat_template:
                content = [{"type": "image"} for _ in range(sample_num_images)]
                content.append({"type": "text", "text": instruction})
                messages = [{"role": "user", "content": content}]
                instruction = self.vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            elif self.vlm_processor.tokenizer and self.vlm_processor.tokenizer.chat_template:
                content = [{"type": "image"} for _ in range(sample_num_images)]
                content.append({"type": "text", "text": instruction})
                messages = [{"role": "user", "content": content}]
                instruction = self.vlm_processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                # No chat template support, use instruction as-is
                # Add image tokens for PaliGemma processor if we have images
                if sample_num_images > 0:
                    image_tokens = "<image> " * sample_num_images
                    instruction = image_tokens + instruction

            batch_text.append(instruction)
            batch_images.append(sample_images)

        # Run processor on entire batch
        processed = self.vlm_processor(images=batch_images, text=batch_text, padding=True, return_tensors="pt")

        processed_batch = batch.copy()
        processed_batch["input_ids"] = processed["input_ids"]
        processed_batch["attention_mask"] = processed["attention_mask"]
        c, h, w = processed["pixel_values"].shape[-3:]
        processed_batch["pixel_values"] = processed["pixel_values"].reshape(len(batch_images), -1, c, h, w)
        processed_batch["camera_names"] = camera_names
        processed_batch["images"] = batch_images
        processed_batch["lowdim"] = {}
        for k in batch["lowdim"][0]:
            if isinstance(batch["lowdim"][0][k][0], str):
                continue
            values = [sample_lowdim[k] for sample_lowdim in batch["lowdim"]]
            processed_batch["lowdim"][k] = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in values])

        normalized_batch = self.normalizer.normalize_batch(processed_batch) if self.normalizer else processed_batch
        return normalized_batch
