import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from vla_foundry.data.preprocessing.image_utils import init_jpeg_encoder
from vla_foundry.data.preprocessing.robotics.preprocess_masks import PaddingStrategy
from vla_foundry.data.preprocessing.utils import upload_sample_to_s3
from vla_foundry.data.robotics.utils import (
    calculate_relative_pose,
    pose_to_9d,
    to_pose_matrix,
)


class BaseRoboticsConverter:
    """
    Base class for all robotics converters.
    This class handles the logic for discovering episodes, loading episode data, and extracting the relevant fields.

    All converters must inherit from this class and implement the methods in this file.
    Some methods are already implemented in this file, and you can probably use them as is.
    Notably, preprocess_robotics_to_tar.py calls process_episode(), which is already defined in this file.
    You need to define all methods called within process_episode(), as well as any other auxiliary methods you need.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.resize_images_size = cfg.resize_images_size
        self.image_resizing_method = cfg.image_resizing_method
        self.image_indices = sorted(cfg.image_indices) if cfg.image_indices is not None else [-1, 0]

        # Initialize JPEG encoder
        init_jpeg_encoder(cfg.jpeg_quality)

        # Set padding function
        self.pad_fn = PaddingStrategy.get_pad_fn(cfg.padding_strategy)

    def discover_episodes(self, source_paths: list[str], max_episodes_to_process: int = -1) -> list[str]:
        """
        Given a list of source paths, return a list of all full episode paths in the directories.
        """
        raise NotImplementedError("Subclasses must implement discover_episodes()")

    def load_episode_data(self, episode_path: str) -> Any:
        """
        The output here will be a dictionary (or anything, really).
        No strict format for the keys. Return whatever is needed for the extract() methods below.
        This output dictionary will be passed to the extract_camera_data() and extract_lowdim_data() methods.
        """
        raise NotImplementedError("Subclasses must implement load_episode_data()")

    def get_episode_length(self, episode_data: Any) -> int:
        """
        Given the episode_data, return the number of timesteps in the episode.
        """
        raise NotImplementedError("Subclasses must implement get_episode_length()")

    def extract_camera_data(self, episode_data: Any):
        """
        Return a dictionary with camera names as keys and image data as values.
        Camera data can be images or bytes. Both are supported in upload_sample_to_s3.
        Can be as simple as `return episode_data["observations"]`
        The values here will cover all the timesteps, then the process_episode() will extract the specific frames.
        """
        return None

    def extract_lowdim_data(self, episode_data: Any):
        """
        Return a dictionary with lowdim keys as keys and lowdim data as values.
        lowdim covers all low dimensional numpy arrays, including actions, proprioception, intrinsics, extrinsics, etc.
        Can be as simple as `return episode_data["lowdim"]`
        The values here will cover all the timesteps, then the process_episode() will extract the specific frames.
        """
        return None

    def extract_intrinsics_extrinsics_data(self, episode_data: Any):
        """
        Return a dictionary with intrinsics and extrinsics keys as keys and data as values.
        Can be as simple as `return episode_data["intrinsics"], episode_data["extrinsics"]`
        This is optional. Can return None, None if not available.
        """
        return None, None

    def extract_metadata_data(self, episode_data: Any):
        """
        Return a either a dictionary or a SampleMetadata object.
        The values here are global values that are shared across all timesteps.
        Alternatively, they can be lists of values, one for each timestep (e.g. timestamps in seconds).
        """
        return None

    def extract_sample_data(
        self,
        anchor_timestep: int,
        episode_path: str,
        episode_length: int,
        camera_data: dict[str, Any],
        lowdim_data: dict[str, Any],
        intrinsics_data: dict[str, Any],
        extrinsics_data: dict[str, Any],
        metadata_data: dict[str, Any],
        statistics_ray_actor,
        logger_actor,
    ):
        """
        Takes in camera_data, lowdim_data, intrinsics_data, extrinsics_data, metadata_data.
        Uses anchor_timestep to extract the specific frames.

        Arguments:
        - anchor_timestep: the current timestep to extract the sample data for.
        - episode_path: the path to the current episode.
        - episode_length: the number of timesteps in the current episode. Returned from get_episode_length().
        - camera_data: a dict with camera names as keys and images (array or bytes) as values.
        Returned from extract_camera_data().
        - lowdim_data: a dict with lowdim keys as keys and lowdim data as values. Returned from extract_lowdim_data().
        - intrinsics_data: Returned from extract_intrinsics_extrinsics_data().
        - extrinsics_data: Returned from extract_intrinsics_extrinsics_data().
        - metadata_data: a dict or a SampleMetadata object (some fields can be blank).
        Returned from extract_metadata_data().

        Returns:
        - sample_images: a dictionary with camera names as keys and image data as values.
        - sample_lowdim: a dictionary with lowdim keys as keys and lowdim data as values.
        - sample_metadata: a dictionary or a SampleMetadata object (some fields can be blank).
        - language_instructions: a dictionary with keys "original", etc. and language instructions as values.
        - sample_point_cloud (optional): point cloud array (T, N, 6) with [x,y,z,r,g,b] or None
        IMPORTANT: Make sure to also update statistics data in this function, as well as the sample counts.
        You can use the statistics_ray_actor and the logger_actor to update the statistics and sample counts.
        """
        raise NotImplementedError("Subclasses must implement extract_sample_data()")

    def create_relative_lowdim_data(
        self, lowdim_data: dict[str, np.ndarray], reference_data: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Create relative coordinate data using configuration-based pose matching."""
        if not hasattr(self, "pose_groups") or not self.pose_groups:
            # No pose groups configured - return empty dict (no relative coordinates needed)
            return {}

        relative_data = {}
        for pose_group in self.pose_groups:
            xyz_key = pose_group["position_key"]
            rot_6d_key = pose_group["rotation_key"]

            xyz_data = lowdim_data[xyz_key]
            rot_6d_data = lowdim_data[rot_6d_key]
            reference_xyz = reference_data[xyz_key]
            reference_rot_6d = reference_data[rot_6d_key]

            # Create reference pose matrix
            reference_pose_matrix = to_pose_matrix(reference_xyz, reference_rot_6d)

            # Create pose matrices for all timesteps (vectorized)
            current_pose_matrices = to_pose_matrix(xyz_data, rot_6d_data)

            # Calculate relative poses (vectorized)
            relative_pose_matrices = calculate_relative_pose(current_pose_matrices, reference_pose_matrix)

            # Extract xyz and rot_6d from relative pose matrices (vectorized)
            relative_xyz, relative_rot_6d = pose_to_9d(relative_pose_matrices)

            # Store relative data with appropriate names
            relative_data[f"{xyz_key}_relative"] = relative_xyz
            relative_data[f"{rot_6d_key}_relative"] = relative_rot_6d

        return relative_data

    def get_episode_id(self, episode_path: str) -> str:
        """Get episode ID from episode path."""
        return os.path.basename(episode_path.rstrip("/"))

    def process_episode(self, episode_path: str, statistics_ray_actor, logger_actor) -> None:
        """
        Process an episode and return a dictionary of the processed episode.

        Here, "processing" an episode means:
        1. Take in episode_path
        2. Load the episode data
        3. Extract the relevant fields (camera data, lowdim data, intrinsics data, extrinsics data, and metadata data)
            - Other modalities should be added here as needed.
        4. For each timestep in the episode:
            - Extract the sample data for the current timestep
            - Upload the sample data to S3
        """
        try:
            episode_data = self.load_episode_data(episode_path)
            episode_length = self.get_episode_length(episode_data)
            camera_data = self.extract_camera_data(episode_data)
            lowdim_data = self.extract_lowdim_data(episode_data)
            intrinsics_data, extrinsics_data = self.extract_intrinsics_extrinsics_data(episode_data)
            metadata_data = self.extract_metadata_data(episode_data)

            # Use ThreadPoolExecutor with bounded queue to prevent memory blowup
            with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as executor:
                futures = set()
                results = []
                stats_samples_batch = []  # Collect stats samples for batched update

                for anchor_timestep in range(0, episode_length, self.cfg.stride):
                    # If we have max_workers futures in flight, wait for one to complete
                    # This bounds memory usage to ~max_workers samples
                    if len(futures) >= self.cfg.num_workers:
                        done_future = next(as_completed(futures))
                        futures.remove(done_future)
                        result = done_future.result()  # Raise any exceptions
                        results.append(result)

                    # Create sample_images, sample_lowdim, sample_metadata, language_instructions,
                    # and optionally point_clouds and stats_sample
                    result = self.extract_sample_data(
                        anchor_timestep,
                        episode_path,
                        episode_length,
                        camera_data,
                        lowdim_data,
                        intrinsics_data,
                        extrinsics_data,
                        metadata_data,
                        statistics_ray_actor,
                        logger_actor,
                    )

                    # Handle 4+ tuple returns (sample_point_clouds and stats_sample are optional 5th and 6th elements)
                    sample_images, sample_lowdim, sample_metadata, language_instructions, *extra = result
                    sample_point_cloud = extra[0] if len(extra) >= 1 else None
                    stats_sample = extra[1] if len(extra) >= 2 else None

                    if sample_images is None and sample_lowdim is None:
                        # Filtered out either by max_padding or still_samples
                        continue

                    # Collect stats sample for batched update
                    if stats_sample is not None:
                        stats_samples_batch.append(stats_sample)

                    sample_data = {
                        "images": sample_images,
                        "lowdim": sample_lowdim,
                        "metadata": sample_metadata,
                        "language_instructions": language_instructions,
                    }

                    # Add point cloud to sample data if provided
                    if sample_point_cloud is not None:
                        sample_data["point_cloud"] = sample_point_cloud

                    # Submit upload task
                    future = executor.submit(
                        upload_sample_to_s3,
                        sample_data=sample_data,
                        output_dir=self.cfg.output_dir,
                        episode_path=episode_path,
                        episode_id=self.get_episode_id(episode_path),
                        frame_idx=anchor_timestep,
                        jpeg_quality=self.cfg.jpeg_quality,
                        resize_images_size=self.resize_images_size,
                        image_resizing_method=self.image_resizing_method,
                    )
                    futures.add(future)

                # Wait for remaining uploads to complete and collect results
                for future in as_completed(futures):
                    result = future.result()  # Raise any exceptions
                    results.append(result)

            # Send batched statistics update (one call per episode instead of per sample)
            if statistics_ray_actor is not None and stats_samples_batch:
                statistics_ray_actor.merge_from_samples.remote(stats_samples_batch)

            return results

        except Exception as e:
            if self.cfg.fail_on_nan:
                raise e
            print(f"Warning: Failed to process episode {episode_path}: {e}")
            return None
