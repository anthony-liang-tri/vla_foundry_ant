import logging
import os
import re

import cv2
import fsspec
import numpy as np

from vla_foundry.data.preprocessing.robotics.converters.base import BaseRoboticsConverter
from vla_foundry.data.preprocessing.robotics.preprocess_masks import create_past_and_future_masks
from vla_foundry.data.preprocessing.robotics.preprocess_params import MMTPreprocessParams
from vla_foundry.data.preprocessing.utils import is_still_sample
from vla_foundry.data.robotics.cv_utils import intrinsics_4_to_3x3
from vla_foundry.data.robotics.utils import xyzrpy_to_T


def downsample_with_valid_depths(depth_image: np.ndarray, target_size: tuple[int, int], mask_threshold: float):
    """Downsample depth image while preserving valid depth pixels. Assumptions:
    1. Invalid depth pixels are represented as 0.
    2.  Target size is smaller than original size and follows (width, height) format."""
    if depth_image.shape[1] == target_size[0] and depth_image.shape[0] == target_size[1]:
        return depth_image
    mask = (depth_image > 0).astype(np.float32)
    resized_depth = cv2.resize(depth_image, target_size, interpolation=cv2.INTER_NEAREST)
    resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)
    resized_depth[resized_mask < mask_threshold] = 0
    return resized_depth


class MMTNPZConverter(BaseRoboticsConverter):
    """Converter for MMT datasets stored in npz format."""

    def __init__(self, cfg: MMTPreprocessParams):
        super().__init__(cfg)

        # Because MMT npz files all have basename of format ep{:04d}_t{:04d}.npz, episode IDs can collide
        # when multiple source directories are used. To avoid this, we added part of the directory path
        # to the episode ID.
        self.episode_id_prefix_offset = -1
        # Make sure the while loop will terminate
        if len(set(self.cfg.source_episodes)) != len(self.cfg.source_episodes):
            raise ValueError("Source episode directories must be unique to avoid episode ID collisions.")
        while True:
            prefixes = [
                os.path.join(*p.split(os.sep)[self.episode_id_prefix_offset :]) for p in self.cfg.source_episodes
            ]
            if len(set(prefixes)) != len(prefixes) or all([p.startswith("npz") for p in prefixes]):
                self.episode_id_prefix_offset -= 1
            else:
                break

        self.item_bbox_keys = ["item_bounding_boxes", "index_of_center_bbox"]

    def get_episode_id(self, episode_path):
        prefix_parts = os.path.dirname(episode_path).split(os.sep)[self.episode_id_prefix_offset :]
        basename = os.path.basename(episode_path).replace("_t0000.npz", "")
        full_parts = prefix_parts + [basename]
        episode_id = os.path.join(*full_parts)
        # Keep legacy naming for S3 output. For local output paths, sanitize separators.
        if not self.cfg.output_dir.startswith("s3://"):
            episode_id = episode_id.replace("/", "_").replace("\\", "_")
        return episode_id

    def get_language_instructions(self, sample_metadata: dict):
        # MMT datasets do not have language instructions
        return {"original": "You are a helpful robot assistant finishing tasks to help people's daily lives."}

    def discover_episodes(self, source_paths: list[str], max_episodes_to_process: int = -1) -> list[str]:
        """Because of the this output of this function is required to represent episodes, we choose to return
        only the npz files corresponding to timestep 0 for each episode and defer episode loading to later."""
        all_episodes = []
        for source_dir in source_paths:
            fs, _ = fsspec.core.url_to_fs(source_dir)
            fs_source_dir = source_dir.replace("s3://", "")
            if not fs.isdir(fs_source_dir):
                raise ValueError(f"Source path {source_dir} is not a directory.")

            t0_npz_files = fs.glob(os.path.join(fs_source_dir, "*_t0000.npz"))
            if source_dir.startswith("s3://"):
                t0_npz_files = ["s3://" + fname for fname in t0_npz_files]
            all_episodes.extend(t0_npz_files)
        all_episodes.sort()

        return all_episodes[:max_episodes_to_process] if max_episodes_to_process > 0 else all_episodes

    def load_episode_data(self, episode_path: str):
        fs, fs_path = fsspec.core.url_to_fs(episode_path)
        pattern = fs_path.replace("_t0000.npz", "_t*.npz")
        npz_files = fs.glob(pattern)
        if episode_path.startswith("s3://"):
            npz_files = ["s3://" + fname for fname in npz_files]
        npz_files.sort()

        data = {}
        for npz_file in npz_files:
            # Extract timestamp from filename ending in _t[\d+].npz
            filename = os.path.basename(npz_file)
            match = re.search(r"^.+?_t(\d+)\.npz$", filename)
            if match:
                timestamp = int(match.group(1))
                # Use fsspec to load the file (works for both local and S3)
                with fs.open(npz_file, "rb") as f:
                    npz_data = np.load(f)
                    # Create a copy of all arrays to avoid file handle dependencies
                    # Skip keys with corrupted/invalid shapes
                    arrays = {}
                    skipped_keys = []
                    for key in npz_data.files:
                        try:
                            arrays[key] = np.array(npz_data[key])
                        except (ValueError, RuntimeError):
                            skipped_keys.append(key)
                    if skipped_keys:
                        logging.getLogger(__name__).warning(f"{npz_file}: skipped corrupted keys: {skipped_keys}")
                    data[timestamp] = arrays
            else:
                raise ValueError(f"Filename {filename} does not match expected pattern.")

        return data

    def get_episode_length(self, episode_data: dict):
        return len(episode_data)

    def extract_camera_data(self, episode_data: dict):
        camera_data = {}
        for camera_name in self.cfg.camera_names:
            camera_data[camera_name] = []
            for t in sorted(episode_data.keys()):
                frame = episode_data[t][camera_name]
                if "depth" in camera_name:
                    # Resize depth here to preserve valid depths only
                    frame = downsample_with_valid_depths(
                        frame, self.resize_images_size, self.cfg.depth_resizing_mask_threshold
                    )
                camera_data[camera_name].append(frame)
            camera_data[camera_name] = np.stack(camera_data[camera_name])

        return camera_data

    def extract_lowdim_data(self, episode_data: dict):

        lowdim_data = {}
        # Determine which raw keys are available in the NPZ data.
        first_step_data = episode_data[min(episode_data.keys())]
        available_raw_keys = set(first_step_data.keys())

        all_keys = list(self.item_bbox_keys)
        # Include keys from flatten indices selection
        if self.cfg.mmt_lowdim_flatten_indices_selection:
            selection_keys = set(self.cfg.mmt_lowdim_flatten_indices_selection.keys()) - set(all_keys)
            all_keys = all_keys + list(selection_keys)
        user_requested_keys = set(all_keys)
        # Include remap source keys so they are loaded from NPZ
        if self.cfg.lowdim_field_remap:
            remap_source_keys = set(self.cfg.lowdim_field_remap.keys()) - user_requested_keys
            all_keys = all_keys + list(remap_source_keys)
        for key in all_keys:
            if key not in available_raw_keys:
                continue
            lowdim_data[key] = []
            for t in sorted(episode_data.keys()):
                data = episode_data[t][key].flatten()
                if key in self.cfg.mmt_lowdim_flatten_indices_selection:
                    data = [data[i] for i in self.cfg.mmt_lowdim_flatten_indices_selection[key]]
                lowdim_data[key].append(data)
            lowdim_data[key] = np.stack(lowdim_data[key])

        # Apply field remaps: derive new fields from existing source fields with index selection
        # Format: {to_name: (source_key, [indices])}
        if self.cfg.lowdim_field_remap:
            resolved = self.cfg._resolved_remap
            remapped_sources = set()
            for to_name, (source_key, indices) in resolved.items():
                if source_key not in lowdim_data:
                    logging.getLogger(__name__).warning(
                        "Remap source '%s' for '%s' not found in lowdim data; skipping.",
                        source_key,
                        to_name,
                    )
                    continue
                lowdim_data[to_name] = lowdim_data[source_key][:, indices]
                remapped_sources.add(source_key)
            # Remove source fields that were only added for remapping
            for source_key in remapped_sources:
                if (
                    source_key in lowdim_data
                    and source_key not in self.cfg.mmt_lowdim_flatten_indices_selection
                    and source_key not in user_requested_keys
                ):
                    del lowdim_data[source_key]

        return lowdim_data

    def extract_intrinsics_extrinsics_data(self, episode_data):
        intrinsics_cam_map = {
            "rgb": "cam_intrinsics",
            "depth": "cam_intrinsics",
            "rgb_left_wrist": "cam_intrinsics_left_wrist",
            "rgb_right_wrist": "cam_intrinsics_right_wrist",
            "depth_left_wrist": "cam_intrinsics_left_wrist",
            "depth_right_wrist": "cam_intrinsics_right_wrist",
        }
        # TODO: NPZ does not contain per-camera extrinsics (e.g. cam_pose_left_wrist).
        # Wrist cameras currently fall back to head camera extrinsics ("cam_pose"),
        # which is incorrect. Once NPZ recording provides per-camera extrinsics,
        # update this map accordingly.
        extrinsics_cam_map = {
            "rgb": "cam_pose",
            "depth": "cam_pose",
            "rgb_left_wrist": "cam_pose",
            "rgb_right_wrist": "cam_pose",
            "depth_left_wrist": "cam_pose",
            "depth_right_wrist": "cam_pose",
        }

        first_step_data = episode_data[min(episode_data.keys())]
        intrinsics_data, extrinsics_data = {}, {}
        for camera_name in self.cfg.camera_names:
            if camera_name not in intrinsics_cam_map or camera_name not in extrinsics_cam_map:
                raise ValueError(f"Camera name {camera_name} not recognized for intrinsics/extrinsics extraction.")
            intrinsics_raw_key = intrinsics_cam_map[camera_name]
            extrinsics_raw_key = extrinsics_cam_map[camera_name]
            has_intrinsics = intrinsics_raw_key in first_step_data
            if not has_intrinsics:
                logging.getLogger(__name__).warning(
                    "Intrinsics key '%s' not found in NPZ for camera '%s'; omitting from output.",
                    intrinsics_raw_key,
                    camera_name,
                )
            has_extrinsics = extrinsics_raw_key in first_step_data
            if not has_extrinsics:
                logging.getLogger(__name__).warning(
                    "Extrinsics key '%s' not found in NPZ for camera '%s'; omitting from output.",
                    extrinsics_raw_key,
                    camera_name,
                )
            if not has_intrinsics and not has_extrinsics:
                continue
            if has_intrinsics:
                intrinsics_data[camera_name] = []
            if has_extrinsics:
                extrinsics_data[camera_name] = []
            for t in sorted(episode_data.keys()):
                if has_intrinsics:
                    intrinsics_frame = episode_data[t][intrinsics_raw_key]
                    # Data preprocessing pipeline expects 3x3 intrinsics and 4x4 extrinsics
                    if intrinsics_frame.shape == (4,):
                        intrinsics_frame = intrinsics_4_to_3x3(intrinsics_frame)
                    assert intrinsics_frame.shape == (3, 3), f"Intrinsics shape mismatch: {intrinsics_frame.shape}"
                    intrinsics_data[camera_name].append(intrinsics_frame)
                if has_extrinsics:
                    extrinsics_frame = episode_data[t][extrinsics_raw_key]
                    if extrinsics_frame.shape == (6,):
                        extrinsics_frame = xyzrpy_to_T(extrinsics_frame).squeeze()
                    assert extrinsics_frame.shape == (4, 4), f"Extrinsics shape mismatch: {extrinsics_frame.shape}"
                    extrinsics_data[camera_name].append(extrinsics_frame)
            if has_intrinsics:
                intrinsics_data[camera_name] = np.stack(intrinsics_data[camera_name])
            if has_extrinsics:
                extrinsics_data[camera_name] = np.stack(extrinsics_data[camera_name])

        return intrinsics_data, extrinsics_data

    def extract_metadata_data(self, episode_data: dict):
        metadata_keys = ["depth_scale"]

        metadata = {}
        first_step_data = episode_data[min(episode_data.keys())]
        for key in metadata_keys:
            if key not in first_step_data:
                continue
            first_value = first_step_data[key]
            metadata[key] = first_value
            # Verify all timesteps have the same value
            for t in sorted(episode_data.keys()):
                if not np.array_equal(episode_data[t][key], first_value):
                    raise ValueError(f"Metadata key '{key}' has inconsistent values across timesteps in episode")

        return metadata

    def extract_sample_camera_calibration(
        self,
        episode_intrinsics: dict[str, np.ndarray],
        episode_extrinsics: dict[str, np.ndarray],
        valid_start: int,
        valid_end: int,
        past_padding: int,
        future_padding: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Extract camera calibration data for the lowdim sequence timespan."""
        sample_intrinsics = {}
        sample_extrinsics = {}

        # Extract intrinsics for the sequence timespan
        for camera_name, intrinsics_data in episode_intrinsics.items():
            if intrinsics_data.ndim == 3:  # Time-varying intrinsics (timesteps, 3, 3)
                # Extract the sequence from valid_start to valid_end (inclusive)
                valid_intrinsics = intrinsics_data[valid_start : valid_end + 1]

                # Apply padding to match lowdim data
                if past_padding > 0 or future_padding > 0:
                    # Use edge padding for camera calibration
                    pad_width = [(past_padding, future_padding)] + [(0, 0)] * (valid_intrinsics.ndim - 1)
                    valid_intrinsics = np.pad(valid_intrinsics, pad_width, mode="edge")

                sample_intrinsics[camera_name] = valid_intrinsics
            elif intrinsics_data.ndim == 2:  # Static intrinsics (3, 3)
                # For static calibration, repeat for the sequence length
                sequence_length = (valid_end - valid_start + 1) + past_padding + future_padding
                sample_intrinsics[camera_name] = np.tile(intrinsics_data[np.newaxis, :, :], (sequence_length, 1, 1))

        # Extract extrinsics for the sequence timespan
        for camera_name, extrinsics_data in episode_extrinsics.items():
            if extrinsics_data.ndim == 3:  # Time-varying extrinsics (timesteps, 4, 4)
                # Extract the sequence from valid_start to valid_end (inclusive)
                valid_extrinsics = extrinsics_data[valid_start : valid_end + 1]

                # Apply padding to match lowdim data
                if past_padding > 0 or future_padding > 0:
                    # Use edge padding for camera calibration
                    pad_width = [(past_padding, future_padding)] + [(0, 0)] * (valid_extrinsics.ndim - 1)
                    valid_extrinsics = np.pad(valid_extrinsics, pad_width, mode="edge")

                sample_extrinsics[camera_name] = valid_extrinsics
            elif extrinsics_data.ndim == 2:  # Static extrinsics (4, 4)
                # For static calibration, repeat for the sequence length
                sequence_length = (valid_end - valid_start + 1) + past_padding + future_padding
                sample_extrinsics[camera_name] = np.tile(extrinsics_data[np.newaxis, :, :], (sequence_length, 1, 1))

        return sample_intrinsics, sample_extrinsics

    def extract_sample_data(
        self,
        anchor_timestep: int,
        episode_path: str,
        episode_length: int,
        camera_data: dict[str, np.ndarray],
        lowdim_data: dict[str, np.ndarray],
        intrinsics_data: dict[str, np.ndarray],
        extrinsics_data: dict[str, np.ndarray],
        metadata_data: dict[str, np.ndarray],
        statistics_ray_actor,
        logger_actor,
    ):
        logger_actor.increment_total_potential_samples.remote()

        # Calculate windows
        lowdim_start = anchor_timestep - self.cfg.past_lowdim_steps
        lowdim_end = anchor_timestep + self.cfg.future_lowdim_steps

        # Check padding
        past_padding = max(0, -lowdim_start)
        future_padding = max(0, lowdim_end - episode_length + 1)

        if past_padding > self.cfg.max_padding_left or future_padding > self.cfg.max_padding_right:
            logger_actor.increment_padding_samples_filtered.remote()
            return None, None, None, None

        valid_start = max(0, lowdim_start)
        valid_end = min(episode_length - 1, lowdim_end)

        # Check if robot is stationary (e.g. to filter pauses)
        if self.cfg.filter_still_samples and is_still_sample(
            lowdim_data, valid_start, valid_end, self.cfg.still_threshold
        ):
            logger_actor.increment_still_samples_filtered.remote()
            return None, None, None, None

        # Extract images
        # Different than spartan and Lerobot converter, we reuse past_lowdim_steps to keep
        # inputs synchronized.
        sample_images = {}
        actual_image_timesteps = []

        for img_offset in range(-self.cfg.past_lowdim_steps, 1):
            img_timestep = np.clip(anchor_timestep + img_offset, 0, episode_length - 1)
            actual_image_timesteps.append(int(img_timestep))

            for camera_name, camera_images in camera_data.items():
                key = f"{camera_name}_t{img_offset}"
                sample_images[key] = camera_images[img_timestep]

        # Process lowdim data (which includes actions)
        sample_lowdim = {}
        for key, data in lowdim_data.items():
            valid_data = data[valid_start : valid_end + 1]
            if past_padding > 0 or future_padding > 0:
                valid_data = self.pad_fn(valid_data, past_padding, future_padding)
            sample_lowdim[key] = valid_data

        # Create masks
        past_mask, future_mask = create_past_and_future_masks(
            anchor_timestep, self.cfg.past_lowdim_steps, self.cfg.future_lowdim_steps, episode_length
        )
        sample_lowdim["past_mask"] = past_mask
        sample_lowdim["future_mask"] = future_mask

        # Extract sequence-specific camera calibration
        sample_intrinsics, sample_extrinsics = self.extract_sample_camera_calibration(
            intrinsics_data,
            extrinsics_data,
            valid_start,
            valid_end,
            past_padding,
            future_padding,
        )

        sample_metadata = {
            "camera_names": list(camera_data.keys()),
            "episode_id": self.get_episode_id(episode_path),
            "lowdim_start_timestep": lowdim_start,
            "lowdim_end_timestep": lowdim_end,
            "anchor_timestep": int(anchor_timestep),
        }
        for key, value in metadata_data.items():
            if isinstance(value, (list, np.ndarray)) and len(value) == episode_length:
                sample_metadata[key] = value[anchor_timestep]
            else:
                sample_metadata[key] = value

        # Build stats_sample for batched statistics update (don't send immediately)
        # Exclude bounding box keys from statistics calculation and hence avoid normalization during training
        stats_sample = (
            None
            if statistics_ray_actor is None
            else {
                "lowdim": {k: v for k, v in sample_lowdim.items() if k not in self.item_bbox_keys},
                "past_mask": past_mask,
                "future_mask": future_mask,
            }
        )

        # Add intrinsics, extrinsics, past_mask, future_mask to lowdim (after building stats_sample)
        for key, value in sample_intrinsics.items():
            sample_lowdim[f"original_intrinsics.{key}"] = value
        for key, value in sample_extrinsics.items():
            sample_lowdim[f"extrinsics.{key}"] = value
        sample_lowdim["past_mask"] = past_mask
        sample_lowdim["future_mask"] = future_mask

        language_instructions = self.get_language_instructions(sample_metadata)

        return sample_images, sample_lowdim, sample_metadata, language_instructions, None, None, stats_sample
