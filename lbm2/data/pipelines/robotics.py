import os
import random

import numpy as np
import webdataset as wds

from lbm2.data.pipelines.base import BaseWebDatasetPipeline
from lbm2.data.processor import get_processor
from lbm2.data.processor.robotics_processor import RoboticsProcessor
from lbm2.data.robotics.normalization import RoboticsNormalizer
from lbm2.data.utils import deterministic_shuffle, log_and_continue
from lbm2.file_utils import json_load
from lbm2.params.data_params import LBMDataParams


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


def extract_robotics_fields(sample, language_instruction_types=None):
    """Extract robotics fields from sample."""
    # Extract data by file type
    images, data = {}, {}
    for key, value in sample.items():
        if key.endswith(".jpg"):
            # Extract camera name and timestep from key (format: {sample_id}.{camera}_{timestep}.jpg)
            img_key = key.split(".")[-2]  # e.g., "wrist_camera_t-1"
            images[img_key] = np.array(value)
        else:
            suffix_map = [
                "lowdim.npz",
                "masks.npz",
                "metadata.json",
                "intrinsics.npz",
                "extrinsics.npz",
                "language_instructions.json",
            ]
            for suffix in suffix_map:
                if key.endswith(suffix):
                    data[suffix] = value

    instruction = select_language_instruction(data.get("language_instructions.json"), language_instruction_types)

    return {
        "images": images,
        "lowdim": data.get("lowdim.npz"),
        "masks": data.get("masks.npz", {}),
        "metadata": data.get("metadata.json", {}),
        "intrinsics": data.get("intrinsics.npz", {}),
        "extrinsics": data.get("extrinsics.npz", {}),
        "language_instruction": instruction,
        "language_instruction_full": data.get("language_instructions.json", {}),
    }


class LBMPipeline(BaseWebDatasetPipeline):
    def __init__(self, modality, data_configs: LBMDataParams, batch_size: int):
        super().__init__(modality, data_configs, batch_size)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.data_configs = data_configs
        self.vlm_processor = get_processor(data_configs)
        self.robotics_processor = RoboticsProcessor()

        # Initialize normalizer
        self.statistics = [json_load(s) for s in data_configs.dataset_statistics]
        self.normalization_config = data_configs.normalization
        if self.data_configs.normalization.enabled:
            self.normalizer = RoboticsNormalizer(dataset_config=self.data_configs, statistics_data=self.statistics)
        else:
            self.normalizer = None

    def __len__(self):
        """Return the number of samples in the dataset (cached)."""
        if not hasattr(self, "_cached_num_samples"):
            num_samples = 0
            for i in range(len(self.data_configs.dataset_manifest)):
                num_samples += self.data_configs.dataset_manifest[i]["num_sequences"]
            self._cached_num_samples = num_samples
        return self._cached_num_samples

    def create_pipeline(self, datastring, checkpoint_num):
        pipeline = [
            wds.SimpleShardList(datastring),
            deterministic_shuffle(
                bufsize=self.data_configs.shuffle_buffer_size,
                initial=self.data_configs.shuffle_initial,
                seed=self.data_configs.seed,
                epoch=checkpoint_num,
            ),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.select(filter_robotics_sample),
            wds.map(
                lambda sample: extract_robotics_fields(
                    sample, language_instruction_types=self.data_configs.language_instruction_types
                ),
                handler=log_and_continue,
            ),
            wds.batched(self.batch_size, partial=False),
            wds.map(
                lambda batch: self.robotics_processor.tokenize_inputs(
                    batch, processor=self.vlm_processor, num_images=self.data_configs.num_images, max_text_seq_len=None
                ),
                handler=log_and_continue,
            ),
            wds.map(
                lambda batch: self.normalizer.normalize_batch(batch) if self.normalizer else batch,
                handler=log_and_continue,
            ),
            wds.map(
                lambda batch: self.robotics_processor.add_action_and_proprioception_fields(
                    batch,
                    action_fields=self.data_configs.action_fields,
                    proprioception_fields=self.data_configs.proprioception_fields,
                ),
                handler=log_and_continue,
            ),
            wds.map(lambda batch: {**batch, "images": None}, handler=log_and_continue),  # Save memory
        ]

        return pipeline
