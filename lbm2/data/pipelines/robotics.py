import os
import random

import numpy as np
import webdataset as wds

from lbm2.data.augmentations.base import Augmentations
from lbm2.data.pipelines.base import BaseWebDatasetPipeline
from lbm2.data.processor.robotics_processor import RoboticsProcessor
from lbm2.data.utils import deterministic_shuffle, log_and_continue
from lbm2.params.data_params import RoboticsDataParams


def filter_robotics_sample(sample):
    """Filter to ensure sample has required robotics data components."""
    has_lowdim = any(k.endswith("lowdim.npz") for k in sample)
    has_metadata = any(k.endswith("metadata.json") for k in sample)
    has_images = any(k.endswith(".jpg") for k in sample)
    return has_lowdim and has_metadata and has_images


def select_language_instruction(language_instructions, instruction_types):
    """Select a random language instruction from the specified types."""
    if not language_instructions or not instruction_types:
        return ""

    # Collect all instructions from the specified types
    available_instructions = []
    for instruction_type in instruction_types:
        if instruction_type in language_instructions:
            if isinstance(language_instructions[instruction_type], str):
                language_instructions[instruction_type] = [language_instructions[instruction_type]]
            available_instructions.extend(language_instructions[instruction_type])
    return random.choice(available_instructions) if available_instructions else ""


def extract_robotics_fields(
    sample,
    language_instruction_types=None,
    action_fields=None,
    proprioception_fields=None,
    intrinsics_fields=None,
    extirnsics_fields=None,
):
    """Extract robotics fields from sample."""
    if extirnsics_fields is None:
        extirnsics_fields = []
    if intrinsics_fields is None:
        intrinsics_fields = []
    if proprioception_fields is None:
        proprioception_fields = []
    if action_fields is None:
        action_fields = []

    images, data = {}, {}
    for key, value in sample.items():
        if key.endswith(".jpg"):
            # Extract camera name and timestep from key (format: {sample_id}.{camera}_{timestep}.jpg)
            img_key = key.split(".")[-2]  # e.g., "wrist_camera_t-1"
            images[img_key] = np.array(value)
        else:
            suffix_map = ["lowdim.npz", "metadata.json", "language_instructions.json"]
            for suffix in suffix_map:
                if key.endswith(suffix):
                    data[suffix] = value

    instruction = select_language_instruction(data.get("language_instructions.json"), language_instruction_types)

    lowdim_data = data.get("lowdim.npz")

    return {
        "images": images,
        "lowdim": {key: lowdim_data.get(key) for key in action_fields + proprioception_fields},
        "past_mask": lowdim_data.get("past_mask"),
        "future_mask": lowdim_data.get("future_mask"),
        "metadata": data.get("metadata.json", {}),
        "intrinsics": {key: lowdim_data.get(key) for key in intrinsics_fields},
        "extrinsics": {key: lowdim_data.get(key) for key in extirnsics_fields},
        "language_instruction": instruction,
        "language_instruction_full": data.get("language_instructions.json", {}),
    }


class RoboticsPipeline(BaseWebDatasetPipeline):
    def __init__(self, modality, data_params: RoboticsDataParams, batch_size: int):
        super().__init__(modality, data_params, batch_size)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.data_params = data_params
        self.robotics_processor = RoboticsProcessor(data_params)
        self.augmentations = Augmentations(data_params.augmentation)

    def __len__(self):
        """Return the number of samples in the dataset (cached)."""
        if not hasattr(self, "_cached_num_samples"):
            num_samples = 0
            for i in range(len(self.data_params.dataset_manifest)):
                num_samples += self.data_params.dataset_manifest[i]["num_sequences"]
            self._cached_num_samples = num_samples
        return self._cached_num_samples

    def create_pipeline(self, datastring, checkpoint_num):
        pipeline = [
            wds.SimpleShardList(datastring),
            deterministic_shuffle(
                bufsize=self.data_params.shuffle_buffer_size,
                initial=self.data_params.shuffle_initial,
                seed=self.data_params.seed,
                epoch=checkpoint_num,
            ),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.select(filter_robotics_sample),
            wds.map(
                lambda sample: self.augmentations.apply_transforms(sample),
                handler=log_and_continue,
            ),
            wds.map(
                lambda sample: extract_robotics_fields(
                    sample,
                    language_instruction_types=self.data_params.language_instruction_types,
                    action_fields=self.data_params.action_fields,
                    proprioception_fields=self.data_params.proprioception_fields,
                    intrinsics_fields=self.data_params.intrinsics_fields,
                    extirnsics_fields=self.data_params.extirnsics_fields,
                ),
                handler=log_and_continue,
            ),
            wds.batched(self.batch_size, partial=False),
            wds.map(
                lambda batch: self.robotics_processor.process_inputs(
                    batch, num_images=self.data_params.num_images, max_text_seq_len=None
                ),
                handler=log_and_continue,
            ),
            wds.map(
                lambda batch: self.robotics_processor.add_action_and_proprioception_fields(
                    batch,
                    action_fields=self.data_params.action_fields,
                    proprioception_fields=self.data_params.proprioception_fields,
                ),
                handler=log_and_continue,
            ),
            wds.map(lambda batch: {**batch, "images": None}, handler=log_and_continue),  # Save memory
        ]

        return pipeline

    def save_configs(self, experiment_path: str):
        # Save normalizer config
        # Can be loaded with RoboticsNormalizer.load(config_path, statistics_path)
        self.robotics_processor.normalizer.save(experiment_path)

        # Save processor config
        # Can be loaded with RoboticsProcessor.load(config_path)
        self.robotics_processor.save(experiment_path)
