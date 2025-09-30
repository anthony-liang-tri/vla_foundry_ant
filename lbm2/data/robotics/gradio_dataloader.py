#!/usr/bin/env python3
"""
Gradio-based Robotics Data Explorer

This tool provides a gradio web interface for exploring preprocessed robotics data
with interactive trajectory visualization overlaid on camera images.

Usage:
    python lbm2/data/robotics/data_explorer_gradio.py \
        --dataset-path /path/to/processed/dataset/ \
        --max-samples 100 \
        --port 7860

Features:
    - Modern web-based interface using Gradio
    - Interactive sliders and dropdowns
    - Real-time trajectory overlay on camera images
    - 3D trajectory visualization
    - Multiple camera support
    - Gripper state visualization
    - Sample metadata display
"""

import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import fsspec
import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm

from lbm2.data.dataloader import get_datastring_input, get_wds_dataloader
from lbm2.data.pipelines.robotics import extract_robotics_fields
from lbm2.data.robotics.normalization import RoboticsNormalizer
from lbm2.data.robotics.utils import _get_rotation_matrix, rot_6d_to_matrix
from lbm2.params.data_params import LBMDataParams


class RoboticsDataLoader:
    """Load and process robotics WebDataset data using the training pipeline."""

    def __init__(
        self,
        params: LBMDataParams,
        max_samples: int = -1,
        max_shards: int = -1,
        use_dataloader: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        augmentation_params: Optional = None,
    ):
        self.params = params
        self.max_samples = max_samples
        self.max_shards = max_shards
        self.use_dataloader = use_dataloader
        self.samples = []
        self.device = device  # Used for model predictions only
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        self.augmentation_params = augmentation_params
        # Initialize normalizer for denormalization if normalization is enabled
        self.normalizer = None
        if self.params.normalization.enabled:
            self.normalizer = RoboticsNormalizer(
                dataset_config=self.params, statistics_path=self.params.dataset_statistics
            )
            if self.normalizer:
                logging.info("RoboticsDataLoader: Normalizer initialized for denormalization")
            else:
                logging.warning("RoboticsDataLoader: Failed to initialize normalizer for denormalization")

    @staticmethod
    def _convert_image_tensor_to_numpy(img_tensor):
        """Convert image tensor to numpy array with proper pixel range handling."""
        img_numpy = img_tensor.permute(1, 2, 0).float().cpu().numpy()  # [H, W, C]

        # Handle different pixel value ranges
        min_val = img_numpy.min()
        max_val = img_numpy.max()
        img_numpy = img_numpy - min_val
        img_numpy = img_numpy / (max_val - min_val)
        img_numpy = img_numpy * 255
        img_numpy = img_numpy.astype(np.uint8)
        return img_numpy

    def load_samples(self) -> List[Dict[str, Any]]:
        """Fallback method to load samples directly from shards."""

        # Try to load manifest
        manifest = []
        try:
            for manifest_path in self.params.dataset_manifest:
                with fsspec.open(manifest_path, "r") as f:
                    manifest.extend([json.loads(line.strip()) for line in f if line.strip()])
        except Exception as e:
            print(f"Warning: Could not load manifest: {e}")
            return []

        # Apply shard limit
        if self.max_shards > 0:
            shards_to_load = manifest[: self.max_shards]
            print(f"Loading samples from {len(shards_to_load)} shards (limited from {len(manifest)} total)...")
        else:
            shards_to_load = manifest  # Load all available shards
            print(f"Loading samples from {len(shards_to_load)} shards...")

        for shard_info in tqdm(shards_to_load, desc="Loading shards"):
            base_path = self.params.dataset_manifest[0].rsplit("/", 1)[0]
            shard_path = os.path.join(base_path, f"{shard_info['shard']}.tar")

            try:
                # Handle S3 paths with pipe prefix for WebDataset
                wds_path = f"pipe:aws s3 cp {shard_path} -" if shard_path.startswith("s3://") else shard_path

                dataset = wds.WebDataset(wds_path).decode("pilrgb").map(extract_robotics_fields)

                for sample in dataset:
                    self.samples.append(sample)

                    if self.max_samples > 0 and len(self.samples) >= self.max_samples:
                        break

            except Exception as e:
                print(f"Warning: Failed to load shard {shard_path}: {e}")
                continue

            if self.max_samples > 0 and len(self.samples) >= self.max_samples:
                break

        return self.samples

    def load_samples_auto(self) -> List[Dict[str, Any]]:
        """Load samples using the preferred method (dataloader or direct file loading)."""
        if self.use_dataloader:
            print("🔧 Loading samples using dataloader pipeline...")
            return self.load_samples_from_dataloader()
        else:
            print("📁 Loading samples directly from files...")
            return self.load_samples()

    def load_samples_from_dataloader(self) -> List[Dict[str, Any]]:
        """Load samples using the data loader and denormalize the output."""
        # Create config for dataloader
        cfg = SimpleNamespace()
        cfg.distributed = SimpleNamespace(world_size=1, rank=0)
        cfg.vit = SimpleNamespace()
        # Create a copy of params with the correct batch size for the data explorer
        cfg.data = self.params
        cfg.hparams = SimpleNamespace(global_batch_size=1)
        cfg.augmentations = self.augmentation_params

        # Load all samples
        num_samples = -1

        # Use get_datastring_input to generate datastrings
        curr_shard_idx_per_dataset = [0]
        shard_shuffle_seed_per_dataset = [0]
        manifest_paths = self.params.dataset_manifest
        dataset_weighting = [1]  # Single dataset
        allow_multiple_epochs = False
        num_workers_per_gpu = 1
        world_size = 1

        (
            datastrings,
            num_samples_list_per_dataset,
            next_shard_idx_per_dataset,
            next_shard_shuffle_seed_per_dataset,
        ) = get_datastring_input(
            num_samples,
            curr_shard_idx_per_dataset,
            shard_shuffle_seed_per_dataset,
            manifest_paths,
            dataset_weighting,
            allow_multiple_epochs,
            num_workers_per_gpu,
            world_size,
        )
        print(f"Creating dataloader with datastrings: {datastrings}")
        print(f"Num samples per dataset: {num_samples_list_per_dataset}")

        # Create dataloader
        dataloader_info = get_wds_dataloader(
            datastrings=datastrings,
            num_samples_per_dataset=num_samples_list_per_dataset,
            checkpoint_num=0,
            cfg=cfg,
        )

        dataloader = dataloader_info.dataloader
        print(f"Dataloader created with {dataloader.num_batches} batches, {dataloader.num_samples} samples")
        print(f"Dataloader type: {type(dataloader)}")
        print(f"Dataloader has __iter__: {hasattr(dataloader, '__iter__')}")
        print(f"Dataloader has __len__: {hasattr(dataloader, '__len__')}")

        samples = []

        # Load samples from dataloader
        batch_count = 0
        try:
            print("Starting to iterate over dataloader...")

            for batch in dataloader:
                batch_count += 1

                if isinstance(batch, dict):
                    # Convert batch to individual samples
                    batch_samples = self._batch_to_samples(batch)
                    samples.extend(batch_samples)

                    # Stop if we have enough samples
                    if self.max_samples > 0 and len(samples) >= self.max_samples:
                        samples = samples[: self.max_samples]
                        break

                else:
                    print(f"WARNING: Batch is not a dict, it's a {type(batch)}")
            print(f"Loaded {len(samples)} samples")

        except Exception as e:
            print(f"Error loading samples from dataloader: {e}")
            import traceback

            traceback.print_exc()

        print(f"Finished loading. Total batches processed: {batch_count}, Total samples: {len(samples)}")

        return samples

    def load_samples_from_model_predictions(self, model, num_inference_steps=10, cfg=None):
        """Load samples using the data loader and denormalize the output."""
        # Create config for dataloader
        if cfg is None:
            cfg = SimpleNamespace()
            cfg.distributed = SimpleNamespace(world_size=1, rank=0)
            cfg.vit = SimpleNamespace()
            # Create a copy of params with the correct batch size for the data explorer
            cfg.data = self.params
            cfg.hparams = SimpleNamespace(global_batch_size=1)
            cfg.augmentations = self.augmentation_params

        # Load all samples
        num_samples = -1

        # Use get_datastring_input to generate datastrings
        curr_shard_idx_per_dataset = [0]
        shard_shuffle_seed_per_dataset = [0]
        manifest_paths = self.params.dataset_manifest
        dataset_weighting = [1]  # Single dataset
        allow_multiple_epochs = False
        num_workers_per_gpu = 1
        world_size = 1

        (
            datastrings,
            num_samples_list_per_dataset,
            next_shard_idx_per_dataset,
            next_shard_shuffle_seed_per_dataset,
        ) = get_datastring_input(
            num_samples,
            curr_shard_idx_per_dataset,
            shard_shuffle_seed_per_dataset,
            manifest_paths,
            dataset_weighting,
            allow_multiple_epochs,
            num_workers_per_gpu,
            world_size,
        )
        print(f"Creating dataloader with datastrings: {datastrings}")
        print(f"Num samples per dataset: {num_samples_list_per_dataset}")

        # Create dataloader
        dataloader_info = get_wds_dataloader(
            datastrings=datastrings, num_samples_per_dataset=num_samples_list_per_dataset, checkpoint_num=0, cfg=cfg
        )

        dataloader = dataloader_info.dataloader

        samples = []

        # Load samples from dataloader
        batch_count = 0
        try:
            print("Starting to iterate over dataloader...")

            for batch in dataloader:
                batch_count += 1

                if isinstance(batch, dict):
                    model.eval()
                    # Don't pass the ground truth actions to the model to avoid cheating
                    actions = batch["actions"].to(self.device) * batch["past_mask"][:, :, None].to(
                        self.device, dtype=self.dtype
                    ) + torch.randn_like(batch["actions"].to(self.device, dtype=self.dtype)) * (
                        batch["past_mask"][:, :, None].to(self.device, dtype=self.dtype) == 0
                    )
                    # actions = batch["actions"].to(self.device)
                    with torch.no_grad():
                        # Generate actions from model
                        predictions = model.generate_actions(
                            batch["input_ids"].to(self.device, dtype=torch.long),
                            batch["pixel_values"].to(self.device, dtype=self.dtype),
                            actions.to(self.device, dtype=self.dtype),
                            batch["attention_mask"].to(self.device, dtype=torch.bool),
                            past_mask=batch["past_mask"].to(self.device, dtype=torch.bool),
                            num_inference_steps=num_inference_steps,
                        )

                    # Replace actions with generated actions
                    start_idx = 0
                    for action_field in self.params.action_fields:
                        end_idx = start_idx + batch["lowdim"][action_field].shape[-1]
                        difference = (
                            (predictions[:, :, start_idx:end_idx] - batch["lowdim"][action_field].to(self.device))
                            .abs()
                            .mean()
                        )
                        print(f"Normalized difference of {action_field} is {difference}")
                        batch["lowdim"][action_field] = predictions[:, :, start_idx:end_idx]
                        start_idx = end_idx

                    # Convert batch to individual samples
                    batch_samples = self._batch_to_samples(batch)
                    samples.extend(batch_samples)

                    # Stop if we have enough samples
                    if self.max_samples > 0 and len(samples) >= self.max_samples:
                        samples = samples[: self.max_samples]
                        break

                else:
                    print(f"WARNING: Batch is not a dict, it's a {type(batch)}")
            print(f"Loaded {len(samples)} samples")

        except Exception as e:
            print(f"Error loading samples from dataloader: {e}")
            import traceback

            traceback.print_exc()

        print(f"Finished loading. Total batches processed: {batch_count}, Total samples: {len(samples)}")

        return samples

    def _denormalize_lowdim_data(self, lowdim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Denormalize lowdim data if normalizer is available."""
        if self.normalizer is None:
            return lowdim_data

        denormalized_lowdim = {}
        for field_name, tensor in lowdim_data.items():
            if isinstance(tensor, (np.ndarray, torch.Tensor)):
                # Convert numpy array to torch tensor for denormalization
                tensor_torch = torch.from_numpy(tensor)
                denormalized_tensor = self.normalizer.denormalize_tensor(tensor_torch, field_name)
                denormalized_lowdim[field_name] = denormalized_tensor.float().cpu().numpy()
            else:
                denormalized_lowdim[field_name] = tensor

        return denormalized_lowdim

    def _batch_to_samples(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert a batch to a list of individual samples."""
        samples = []
        batch_size = batch["input_ids"].shape[0]

        if batch_size == 0:
            return samples

        for i in range(batch_size):
            sample = {}

            # Extract metadata
            if "metadata" in batch and i < len(batch["metadata"]):
                sample["metadata"] = batch["metadata"][i]
            # Add __key__ propagation
            if "__key__" in batch and i < len(batch["__key__"]):
                sample["__key__"] = batch["__key__"][i]

            # Extract lowdim data
            if "lowdim" in batch:
                sample["lowdim"] = {}
                for key, value in batch["lowdim"].items():
                    if isinstance(value, torch.Tensor):
                        # Extract single sample from batch tensor
                        if value.dim() > 0:
                            sample["lowdim"][key] = value[i].float().cpu().numpy()
                        else:
                            sample["lowdim"][key] = value.float().cpu().numpy()
                    else:
                        sample["lowdim"][key] = value

                # Denormalize lowdim data if normalization is enabled
                if self.params.normalization.enabled and self.normalizer is not None:
                    sample["lowdim"] = self._denormalize_lowdim_data(sample["lowdim"])

            # Extract images - handle the processed pixel_values from the pipeline
            if "pixel_values" in batch and batch["pixel_values"] is not None:
                # The pipeline processes images into pixel_values tensor
                # Shape: [batch_size, num_images, channels, height, width]
                pixel_values = batch["pixel_values"]
                if pixel_values.dim() == 5:  # [B, N, C, H, W]
                    num_images = pixel_values.shape[1]
                    sample["images"] = {}

                    # Get camera names from metadata or batch
                    camera_names = batch.get("camera_names", [])
                    if not camera_names and "metadata" in sample:
                        camera_names = sample["metadata"].get("camera_names", [])

                    # Convert each image back to numpy and store with camera name
                    for img_idx in range(num_images):
                        if img_idx < len(camera_names):
                            camera_name = camera_names[img_idx]
                            # Convert tensor to numpy: [C, H, W] -> [H, W, C]
                            img_tensor = pixel_values[i, img_idx]  # [C, H, W]
                            img_numpy = RoboticsDataLoader._convert_image_tensor_to_numpy(img_tensor)
                            sample["images"][camera_name] = img_numpy
                        else:
                            # Fallback naming if camera names not available
                            sample["images"][f"camera_{img_idx}"] = img_tensor.permute(1, 2, 0).float().cpu().numpy()

            # Extract past and future masks
            if "past_mask" in batch:
                sample["past_mask"] = batch["past_mask"][i].float().cpu().numpy()
            if "future_mask" in batch:
                sample["future_mask"] = batch["future_mask"][i].float().cpu().numpy()

            # Extract actions
            if "actions" in batch:
                # Denormalize actions data if normalization is enabled
                if self.params.normalization.enabled and self.normalizer is not None:
                    sample["actions"] = (
                        self.normalizer.denormalize_actions_batch(batch["actions"])[i].float().cpu().numpy()
                    )
                else:
                    sample["actions"] = batch["actions"][i].float().cpu().numpy()

            # Extract calibration data (intrinsics and extrinsics)
            if "intrinsics" in batch:
                sample["intrinsics"] = {}
                for key, value in batch["intrinsics"][i].items():
                    if isinstance(value, torch.Tensor):
                        sample["intrinsics"][key] = value[i].float().cpu().numpy()
                    else:
                        sample["intrinsics"][key] = value

            if "extrinsics" in batch:
                sample["extrinsics"] = {}
                for key, value in batch["extrinsics"][i].items():
                    if isinstance(value, torch.Tensor):
                        sample["extrinsics"][key] = value[i].float().cpu().numpy()
                    else:
                        sample["extrinsics"][key] = value

            if "language_instructions" in batch:
                print(f"Found language instructions in batch: {batch['language_instructions'][i]}")
                sample["language_instructions"] = batch["language_instructions"][i]

            if "__key__" in batch and i < len(batch["__key__"]):
                sample["__key__"] = batch["__key__"][i]

            samples.append(sample)

        return samples

    @staticmethod
    def _batch_to_samples_static(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Static version of _batch_to_samples for use outside the class."""
        samples = []
        batch_size = len(batch.get("metadata", []))

        if batch_size == 0:
            return samples

        for i in range(batch_size):
            sample = {}

            # Extract metadata
            if "metadata" in batch and i < len(batch["metadata"]):
                sample["metadata"] = batch["metadata"][i]

            # Extract lowdim data
            if "lowdim" in batch:
                sample["lowdim"] = {}
                for key, value in batch["lowdim"].items():
                    if isinstance(value, torch.Tensor):
                        # Extract single sample from batch tensor
                        if value.dim() > 0:
                            sample["lowdim"][key] = value[i].float().cpu().numpy()
                        else:
                            sample["lowdim"][key] = value.float().cpu().numpy()
                    else:
                        sample["lowdim"][key] = value

            # Extract images - handle the processed pixel_values from the pipeline
            if "pixel_values" in batch and batch["pixel_values"] is not None:
                # The pipeline processes images into pixel_values tensor
                # Shape: [batch_size, num_images, channels, height, width]
                pixel_values = batch["pixel_values"]
                if pixel_values.dim() == 5:  # [B, N, C, H, W]
                    num_images = pixel_values.shape[1]
                    sample["images"] = {}

                    # Get camera names from metadata or batch
                    camera_names = batch.get("camera_names", [])
                    if not camera_names and "metadata" in sample:
                        camera_names = sample["metadata"].get("camera_names", [])

                    # Convert each image back to numpy and store with camera name
                    for img_idx in range(num_images):
                        if img_idx < len(camera_names):
                            camera_name = camera_names[img_idx]
                            # Convert tensor to numpy: [C, H, W] -> [H, W, C]
                            img_tensor = pixel_values[i, img_idx]  # [C, H, W]
                            img_numpy = RoboticsDataLoader._convert_image_tensor_to_numpy(img_tensor)
                            sample["images"][camera_name] = img_numpy
                        else:
                            # Fallback naming if camera names not available
                            img_tensor = pixel_values[i, img_idx]  # [C, H, W]
                            img_numpy = RoboticsDataLoader._convert_image_tensor_to_numpy(img_tensor)
                            sample["images"][f"camera_{img_idx}"] = img_numpy

            # Extract past and future masks
            if "past_mask" in batch:
                sample["past_mask"] = batch["past_mask"][i].float().cpu().numpy()
            if "future_mask" in batch:
                sample["future_mask"] = batch["future_mask"][i].float().cpu().numpy()

            # Extract actions
            if "actions" in batch:
                sample["actions"] = {}
                for key, value in batch["actions"].items():
                    if isinstance(value, torch.Tensor):
                        if value.dim() > 0:
                            sample["actions"][key] = value[i].float().cpu().numpy()
                        else:
                            sample["actions"][key] = value.float().cpu().numpy()
                    else:
                        sample["actions"][key] = value

            # Extract calibration data (intrinsics and extrinsics)
            if "intrinsics" in batch:
                sample["intrinsics"] = {}
                for key, value in batch["intrinsics"][i].items():
                    if isinstance(value, torch.Tensor):
                        sample["intrinsics"][key] = value[i].float().cpu().numpy()
                    else:
                        sample["intrinsics"][key] = value

            if "extrinsics" in batch:
                sample["extrinsics"] = {}
                for key, value in batch["extrinsics"][i].items():
                    if isinstance(value, torch.Tensor):
                        sample["extrinsics"][key] = value[i].float().cpu().numpy()
                    else:
                        sample["extrinsics"][key] = value

            samples.append(sample)

        return samples


class TrajectoryExtractor:
    """Extract robot trajectories from lowdim data, with instance state."""

    def __init__(
        self,
        rotation_interpretation: str = "XY_+-",
        camera_best_interpretations: Optional[Dict[str, str]] = None,
    ):
        if camera_best_interpretations is None:
            # Default camera best interpretation for LBM data to change the 6D rotation interpretation directions
            camera_best_interpretations = {
                "wrist_right_minus": "XY_+-",
                "wrist_right_plus": "XY_+-",
                "wrist_left_minus": "XY_+-",
                "wrist_left_plus": "XY_+-",
                "scene_right_0": "XY_+-",
                "scene_left_0": "XY_+-",
            }
        # Instance configuration
        self.rotation_interpretation = rotation_interpretation
        self._interpretations = {
            # Standard permutations with different signs
            "XY_++": ((0, 1, 2), (1, 1)),  # X=first3+, Y=next3+, Z=cross
            "XY_+-": ((0, 1, 2), (1, -1)),  # X=first3+, Y=next3-, Z=cross
            "XY_-+": ((0, 1, 2), (-1, 1)),  # X=first3-, Y=next3+, Z=cross
            "XY_--": ((0, 1, 2), (-1, -1)),  # X=first3-, Y=next3-, Z=cross
            # Swapped permutations
            "YX_++": ((1, 0, 2), (1, 1)),  # Y=first3+, X=next3+, Z=cross
            "YX_+-": ((1, 0, 2), (1, -1)),  # Y=first3+, X=next3-, Z=cross
            "YX_-+": ((1, 0, 2), (-1, 1)),  # Y=first3-, X=next3+, Z=cross
            "YX_--": ((1, 0, 2), (-1, -1)),  # Y=first3-, X=next3-, Z=cross
            # Other axis combinations (reduced set for speed)
            "XZ_++": ((0, 2, 1), (1, 1)),  # X=first3+, Z=next3+, Y=cross
            "XZ_+-": ((0, 2, 1), (1, -1)),  # X=first3+, Z=next3-, Y=cross
            "YZ_++": ((1, 2, 0), (1, 1)),  # Y=first3+, Z=next3+, X=cross
            "YZ_+-": ((1, 2, 0), (1, -1)),  # Y=first3+, Z=next3-, X=cross
        }
        self.wrist_rot_transform = np.array(
            [
                [-6.16650880e-01, -7.87227812e-01, 3.75015604e-03],
                [-7.87236259e-01, 6.16648793e-01, -1.82714716e-03],
                [-8.74148131e-04, -4.07897071e-03, -9.99991299e-01],
            ]
        )
        self.camera_best_interpretations = camera_best_interpretations or {}
        self._camera_best_configs = {}
        for cam_name, interpretation in self.camera_best_interpretations.items():
            self._camera_best_configs[cam_name] = self._interpretations[interpretation]
        self._best_interpretation_config = None  # Optionally set by user

    def set_camera_best_interpretations(self, camera_best_interpretations: Dict[str, str]) -> None:
        self.camera_best_interpretations = camera_best_interpretations
        self._camera_best_configs = {}
        for cam_name, interpretation in camera_best_interpretations.items():
            self._camera_best_configs[cam_name] = self._interpretations[interpretation]

    def set_rotation_interpretation(self, interpretation: str) -> None:
        # Validate format (should be like XY_++, YX_-+, etc.)
        if not (
            len(interpretation.split("_")) == 2
            and len(interpretation.split("_")[0]) == 2
            and len(interpretation.split("_")[1]) == 2
        ):
            interpretation = "XY_++"
        self.rotation_interpretation = interpretation

    def arm_xyz_to_gripper_xyz(
        self,
        arm_xyz: np.ndarray,
        arm_rot_6d: np.ndarray,
        gripper_offset: float = 0.22,
        interpretation: str = None,
        use_discovered_transform: bool = True,
        camera_name: str = None,
    ) -> np.ndarray:
        """Convert arm xyz and rot_6d to gripper xyz with different 6D interpretations."""
        # Use camera-specific interpretation if available
        if camera_name and self._camera_best_configs and camera_name in self._camera_best_configs:
            perm_config, signs = self._camera_best_configs[camera_name]
            rot_mat = rot_6d_to_matrix(arm_rot_6d, perm_config, signs)
        elif self._best_interpretation_config:
            # Fallback to global interpretation if available
            perm_config, signs = self._best_interpretation_config
            rot_mat = rot_6d_to_matrix(arm_rot_6d, perm_config, signs)
        else:
            # Use class-level setting if no interpretation specified
            if interpretation is None:
                interpretation = self.rotation_interpretation
            # Fallback: parse the interpretation name directly
            rot_mat = self._parse_and_compute_rotation(arm_rot_6d, interpretation)

        # Option 1: Z-direction (current assumption)
        gripper_offset_vector = np.array([0, 0, gripper_offset])

        # Calculate gripper in robot frame (pure robot coordinates)
        # Corrected order of multiplication for valid transformation: world->wrist @ wrist->camera
        gripper_xyz = arm_xyz + self.wrist_rot_transform @ rot_mat @ gripper_offset_vector

        return gripper_xyz

    def _parse_and_compute_rotation(self, arm_rot_6d: np.ndarray, interpretation: str) -> np.ndarray:
        """Parse interpretation string and compute rotation matrix."""
        # Default configuration
        perm_config = [0, 1, 2]  # XYZ
        signs = [1, 1]  # ++

        # Parse interpretation like "XY_++" or "YX_-+"
        if "_" in interpretation:
            parts = interpretation.split("_")
            if len(parts) == 2:
                axes_part, signs_part = parts

                # Parse axes
                if axes_part.upper() == "XY":
                    perm_config = [0, 1, 2]
                elif axes_part.upper() == "YX":
                    perm_config = [1, 0, 2]
                elif axes_part.upper() == "XZ":
                    perm_config = [0, 2, 1]
                elif axes_part.upper() == "YZ":
                    perm_config = [1, 2, 0]

                # Parse signs
                if len(signs_part) == 2:
                    signs = [1 if signs_part[0] == "+" else -1, 1 if signs_part[1] == "+" else -1]

        return self._rot_6d_to_rot_mat(arm_rot_6d, perm_config, signs)

    def rot_6d_to_rot_mat_alternative(self, rot_6d: np.ndarray, interpretation: str = "standard") -> np.ndarray:
        """Alternative 6D to rotation matrix conversion with different axis interpretations."""
        # Handle single vector or batch of vectors
        original_shape = rot_6d.shape
        if rot_6d.ndim == 1:
            rot_6d = rot_6d.reshape(1, -1)

        if interpretation == "standard":
            # Standard: first 3 = X-axis, next 3 = Y-axis
            x_raw = rot_6d[:, :3]
            y_raw = rot_6d[:, 3:6]
        elif interpretation == "swapped":
            # Swapped: first 3 = Y-axis, next 3 = X-axis
            y_raw = rot_6d[:, :3]
            x_raw = rot_6d[:, 3:6]
        elif interpretation == "xz":
            # X and Z: first 3 = X-axis, next 3 = Z-axis
            x_raw = rot_6d[:, :3]
            z_raw = rot_6d[:, 3:6]
            # Y will be computed as Z x X
            z = z_raw / np.linalg.norm(z_raw, axis=1, keepdims=True)
            x = x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)
            y = np.cross(z, x, axis=1)
            rot_mat = np.stack([x, y, z], axis=2)
            if len(original_shape) == 1:
                return rot_mat[0]
            else:
                return rot_mat
        elif interpretation == "yz":
            # Y and Z: first 3 = Y-axis, next 3 = Z-axis
            y_raw = rot_6d[:, :3]
            z_raw = rot_6d[:, 3:6]
            # X will be computed as Y x Z
            y = y_raw / np.linalg.norm(y_raw, axis=1, keepdims=True)
            z = z_raw / np.linalg.norm(z_raw, axis=1, keepdims=True)
            x = np.cross(y, z, axis=1)
            rot_mat = np.stack([x, y, z], axis=2)
            if len(original_shape) == 1:
                return rot_mat[0]
            else:
                return rot_mat
        else:
            raise ValueError(f"Unknown interpretation: {interpretation}")

        # Normalize first column
        x = x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)

        # Make second column orthogonal to first and normalize
        y = y_raw - np.sum(x * y_raw, axis=1, keepdims=True) * x
        y = y / np.linalg.norm(y, axis=1, keepdims=True)

        # Compute third column via cross product
        z = np.cross(x, y, axis=1)

        # Stack to form rotation matrices
        rot_mat = np.stack([x, y, z], axis=2)  # Shape: [batch_size, 3, 3]

        # Return original shape
        if len(original_shape) == 1:
            return rot_mat[0]
        else:
            return rot_mat

    def find_best_6d_interpretation(self, sample: Dict[str, Any], camera_name: str = "wrist_right_minus") -> str:
        """Find the best 6D rotation interpretation based on wrist-camera rotation consistency."""
        if "lowdim" not in sample:
            return "XY_++"

        lowdim = sample["lowdim"]
        right_6d_key = "robot__actual__poses__right::panda__rot_6d"

        if right_6d_key not in lowdim:
            return "XY_++"

        arm_6d = lowdim[right_6d_key]

        # Get camera extrinsics
        if "extrinsics" not in sample or camera_name not in sample["extrinsics"]:
            # Try alternative camera names
            alt_cameras = ["wrist_right_plus", "wrist_right_0", "scene_left_0"]
            for alt_cam in alt_cameras:
                if "extrinsics" in sample and alt_cam in sample["extrinsics"]:
                    camera_name = alt_cam
                    break
            else:
                return "XY_++"

        metadata = sample["metadata"]
        current_timestep = metadata["anchor_timestep"] - metadata["lowdim_start_timestep"]

        valid_interpretations = []

        for interp in self._interpretations:
            try:
                # Test rotation matrix creation using comprehensive approach
                wrist_rot = self._parse_and_compute_rotation(arm_6d[current_timestep], interp)

                # Check if it's a valid rotation matrix
                det = np.linalg.det(wrist_rot)
                is_orthogonal = np.allclose(wrist_rot @ wrist_rot.T, np.eye(3), atol=1e-6)

                if abs(det - 1.0) < 0.01 and is_orthogonal:
                    valid_interpretations.append(interp)

            except Exception:
                continue

        if not valid_interpretations:
            return "XY_++"

        # Return the first valid interpretation
        return valid_interpretations[0]

    def extrinsics_6d_interpretation_comparison(self, samples: List[Dict[str, Any]], max_samples: int = 10) -> str:
        """Comprehensively test all possible 6D rotation interpretations PER CAMERA and find the best ones."""

        # Limit number of samples for testing (for speed)
        test_samples = samples[: min(max_samples, len(samples))]

        # Test each interpretation and collect results per camera
        camera_results = {}  # camera_name -> [(interp_name, consistency_score, transforms)]

        for interp_name, (perm_config, signs) in self._interpretations.items():
            # Test this interpretation across all samples
            transforms_per_sample = {}

            for sample_idx, sample in enumerate(test_samples):
                # Get robot rotation with this interpretation
                robot_rotation = self._compute_robot_rotation_with_interpretation(sample, perm_config, signs)

                if robot_rotation is None:
                    continue

                # Get camera rotations and compute transformations
                sample_transforms = self._compute_camera_transformations(sample, robot_rotation)

                if sample_transforms:
                    transforms_per_sample[f"sample_{sample_idx}"] = sample_transforms

            # Analyze consistency PER CAMERA for this interpretation
            if len(transforms_per_sample) >= 2:
                # Get all camera names
                camera_names = set()
                for sample_transforms in transforms_per_sample.values():
                    camera_names.update(sample_transforms.keys())

                # Test each camera separately
                for cam_name in camera_names:
                    # Extract transformations for this specific camera
                    cam_transforms_per_sample = {}
                    for sample_id, sample_transforms in transforms_per_sample.items():
                        if cam_name in sample_transforms:
                            cam_transforms_per_sample[sample_id] = {cam_name: sample_transforms[cam_name]}

                    if len(cam_transforms_per_sample) >= 2:
                        consistency_score = self._analyze_transformation_consistency(cam_transforms_per_sample)

                        if cam_name not in camera_results:
                            camera_results[cam_name] = []
                        camera_results[cam_name].append((interp_name, consistency_score, cam_transforms_per_sample))

        camera_best_interpretations = {}
        camera_best_configs = {}
        all_transforms = {}
        for cam_name, cam_interpretation_results in camera_results.items():
            # Sort by consistency score (lower is better)
            cam_interpretation_results.sort(key=lambda x: x[1])

            # Store the best interpretation for this camera
            best_interp_name, best_score, best_transforms = cam_interpretation_results[0]
            camera_best_interpretations[cam_name] = best_interp_name

            # Find the configuration for the best interpretation
            best_config = None
            for interp_name, (perm_config, signs) in self._interpretations.items():
                if interp_name == best_interp_name:
                    best_config = (perm_config, signs)
                    break

            if best_config:
                camera_best_configs[cam_name] = best_config

            # Store transforms for this camera
            for sample_id, sample_transforms in best_transforms.items():
                if sample_id not in all_transforms:
                    all_transforms[sample_id] = {}
                all_transforms[sample_id].update(sample_transforms)

        return camera_best_interpretations, all_transforms

    def _compute_robot_rotation_with_interpretation(
        self, sample: Dict[str, Any], perm_config: List[int], signs: List[int]
    ) -> Optional[np.ndarray]:
        """Compute robot rotation matrix using specific axis permutation and signs."""
        if "lowdim" not in sample:
            return None

        lowdim = sample["lowdim"]
        right_6d_key = "robot__actual__poses__right::panda__rot_6d"

        if right_6d_key not in lowdim:
            return None

        arm_6d = lowdim[right_6d_key]
        metadata = sample["metadata"]
        current_timestep = metadata["anchor_timestep"] - metadata["lowdim_start_timestep"]

        if current_timestep >= len(arm_6d):
            return None

        rot_6d = arm_6d[current_timestep]

        # Apply the specific interpretation
        # Determine which vectors to use based on configuration
        if perm_config[0] == 0:  # First vector from first 3 elements
            vec1 = rot_6d[:3] * signs[0]
            vec2 = rot_6d[3:6] * signs[1]
        else:  # First vector from next 3 elements
            vec1 = rot_6d[3:6] * signs[0]
            vec2 = rot_6d[:3] * signs[1]

        # Normalize first vector
        x = vec1 / np.linalg.norm(vec1)

        # Make second vector orthogonal and normalize
        y = vec2 - np.dot(x, vec2) * x
        y = y / np.linalg.norm(y)

        # Compute third vector
        z = np.cross(x, y)

        rot_mat = _get_rotation_matrix(x, y, z, perm_config)
        # Verify it's a valid rotation matrix
        det = np.linalg.det(rot_mat)

        if det < 0:
            rot_mat = _get_rotation_matrix(x, y, -z, perm_config)

        if abs(det - 1.0) > 0.1:  # Allow some tolerance
            return None

        return rot_mat

    def _compute_camera_transformations(
        self, sample: Dict[str, Any], robot_rotation: np.ndarray
    ) -> Optional[Dict[str, np.ndarray]]:
        """Compute camera-robot transformations for all available cameras."""
        if "extrinsics" not in sample:
            return None

        metadata = sample["metadata"]
        current_timestep = metadata["anchor_timestep"] - metadata["lowdim_start_timestep"]

        transformations = {}
        camera_names = ["wrist_right_minus", "wrist_right_plus"]

        for cam_name in camera_names:
            if cam_name not in sample["extrinsics"]:
                continue

            try:
                cam_extrinsics = sample["extrinsics"][cam_name]

                if cam_extrinsics.ndim == 3:
                    if current_timestep >= cam_extrinsics.shape[0]:
                        continue
                    current_extrinsics = cam_extrinsics[current_timestep]
                else:
                    current_extrinsics = cam_extrinsics

                # Extract camera rotation (top-left 3x3)
                camera_rotation = current_extrinsics[:3, :3]

                # Compute transformation: Camera = T * Robot, so T = Camera * Robot^T
                transformation = camera_rotation @ robot_rotation.T
                transformations[cam_name] = transformation

            except Exception:
                continue

        return transformations if transformations else None

    def _analyze_transformation_consistency(self, transforms_per_sample: Dict[str, Dict[str, np.ndarray]]) -> float:
        """Analyze consistency of transformations across samples and return a score (lower is better)."""
        if len(transforms_per_sample) < 2:
            return float("inf")

        # Get all camera names
        camera_names = set()
        for sample_transforms in transforms_per_sample.values():
            camera_names.update(sample_transforms.keys())

        total_inconsistency = 0.0
        camera_count = 0

        for cam_name in camera_names:
            # Collect transformations for this camera
            cam_transforms = []
            for sample_transforms in transforms_per_sample.values():
                if cam_name in sample_transforms:
                    cam_transforms.append(sample_transforms[cam_name])

            if len(cam_transforms) < 2:
                continue

            # Calculate standard deviation of matrix elements
            cam_transforms = np.array(cam_transforms)
            std_per_element = np.std(cam_transforms, axis=0)
            max_std = np.max(std_per_element)

            total_inconsistency += max_std
            camera_count += 1

        if camera_count == 0:
            return float("inf")

        return total_inconsistency / camera_count

    def extract_trajectories(
        self, sample: Dict[str, Any], camera_name: str = None, include_desired: bool = False
    ) -> Dict[str, np.ndarray]:
        """Extract robot trajectories from a sample.

        Args:
            sample: Sample data dictionary
            camera_name: Camera name for interpretation
            include_desired: If True, include both actual and desired poses
        """
        trajectories = {}

        lowdim = sample["lowdim"]

        if camera_name is None and self.camera_best_interpretations:
            camera_name = list(self.camera_best_interpretations.keys())[0]  # Use first camera for interpretation

        # Extract left arm trajectory
        left_xyz_key = "robot__actual__poses__left::panda__xyz"
        left_gripper_key = "robot__actual__grippers__left::panda_hand"
        left_6d_key = "robot__actual__poses__left::panda__rot_6d"
        if left_xyz_key in lowdim:
            trajectories["left_arm_xyz"] = lowdim[left_xyz_key]
            trajectories["left_arm_gripper"] = lowdim[left_gripper_key]
            trajectories["left_arm_6d"] = lowdim[left_6d_key]
            trajectories["left_gripper_xyz"] = self.arm_xyz_to_gripper_xyz(
                trajectories["left_arm_xyz"], trajectories["left_arm_6d"], camera_name=camera_name
            )

            # Add desired poses if requested and available
            if include_desired:
                left_desired_xyz_key = "robot__desired__poses__left::panda__xyz"
                left_desired_gripper_key = "robot__desired__grippers__left::panda_hand"
                left_desired_6d_key = "robot__desired__poses__left::panda__rot_6d"

                if left_desired_xyz_key in lowdim:
                    trajectories["left_arm_xyz_desired"] = lowdim[left_desired_xyz_key]
                    trajectories["left_arm_gripper_desired"] = lowdim[left_desired_gripper_key]
                    trajectories["left_arm_6d_desired"] = lowdim[left_desired_6d_key]
                    trajectories["left_gripper_xyz_desired"] = self.arm_xyz_to_gripper_xyz(
                        trajectories["left_arm_xyz_desired"],
                        trajectories["left_arm_6d_desired"],
                        camera_name=camera_name,
                    )

        # Extract right arm trajectory
        right_xyz_key = "robot__actual__poses__right::panda__xyz"
        right_gripper_key = "robot__actual__grippers__right::panda_hand"
        right_6d_key = "robot__actual__poses__right::panda__rot_6d"
        if right_xyz_key in lowdim:
            trajectories["right_arm_xyz"] = lowdim[right_xyz_key]
            trajectories["right_arm_gripper"] = lowdim[right_gripper_key]
            trajectories["right_arm_6d"] = lowdim[right_6d_key]
            trajectories["right_gripper_xyz"] = self.arm_xyz_to_gripper_xyz(
                trajectories["right_arm_xyz"], trajectories["right_arm_6d"], camera_name=camera_name
            )

            # Add desired poses if requested and available
            if include_desired:
                right_desired_xyz_key = "robot__desired__poses__right::panda__xyz"
                right_desired_gripper_key = "robot__desired__grippers__right::panda_hand"
                right_desired_6d_key = "robot__desired__poses__right::panda__rot_6d"

                if right_desired_xyz_key in lowdim:
                    trajectories["right_arm_xyz_desired"] = lowdim[right_desired_xyz_key]
                    trajectories["right_arm_gripper_desired"] = lowdim[right_desired_gripper_key]
                    trajectories["right_arm_6d_desired"] = lowdim[right_desired_6d_key]
                    trajectories["right_gripper_xyz_desired"] = self.arm_xyz_to_gripper_xyz(
                        trajectories["right_arm_xyz_desired"],
                        trajectories["right_arm_6d_desired"],
                        camera_name=camera_name,
                    )

        return trajectories
