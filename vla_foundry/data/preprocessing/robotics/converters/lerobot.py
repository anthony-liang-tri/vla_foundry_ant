import os
import re
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import ray

from vla_foundry.data.preprocessing.robotics.converters.base import BaseRoboticsConverter
from vla_foundry.data.preprocessing.robotics.preprocess_masks import create_past_and_future_masks
from vla_foundry.data.preprocessing.utils import is_still_sample
from vla_foundry.file_utils import copy_to_temp_file, file_exists, json_load, jsonl_load, list_directory


def resolve_path(base_path: str, relative_path: str) -> str:
    # If relative_path is already absolute (starts with s3:// or /), return as-is
    if relative_path.startswith("s3://") or relative_path.startswith("/"):
        return relative_path
    return f"{base_path.rstrip('/')}/{relative_path.lstrip('/')}"


def detect_fps(info_path: str) -> float:
    """Automatically detect FPS from info.json file."""
    info = json_load(info_path)
    if "fps" in info:
        fps = float(info["fps"])
        print(f"Detected FPS from global setting: {fps}")
    else:
        fps = 30.0
        print(f"Warning: Could not detect FPS from info.json, using default FPS {fps}")
    return fps


def read_parquet(path: str) -> pd.DataFrame:
    if path.startswith("s3"):
        with copy_to_temp_file(path) as temp_parquet:
            return pq.read_table(temp_parquet).to_pandas()
    return pq.read_table(path).to_pandas()


def discover_parquet_files(root: str) -> list[str]:
    parquet_files = []
    for chunk in list_directory(root):
        if not chunk.startswith("chunk-"):
            continue
        chunk_path = f"{root.rstrip('/')}/{chunk}"
        for item in list_directory(chunk_path):
            if item.endswith(".parquet"):
                parquet_files.append(f"{chunk_path}/{item}")
    return sorted(parquet_files)


def task_list_contains(tasks: Any, task_filter: str | None) -> bool:
    if task_filter is None:
        return True
    if isinstance(tasks, str):
        return tasks == task_filter
    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
    return task_filter in list(tasks)


@ray.remote
def build_episode_lookup_chunk(chunk_dir: str) -> dict[int, str]:
    """Build episode lookup for a single chunk directory."""
    episode_lookup = {}
    for file in list_directory(chunk_dir):
        if file.endswith(".parquet") and "episode_" in file:
            match = re.search(r"episode_(\d+)\.parquet", file)
            if match:
                ep_num = int(match.group(1))
                episode_lookup[ep_num] = f"{chunk_dir.rstrip('/')}/{file}"
    return episode_lookup


def build_episode_lookup(data_chunks: list[str]) -> dict[int, str]:
    print(f"Building episode lookup dict from {len(data_chunks)} chunks...")
    # Process chunks in parallel
    chunk_futures = [build_episode_lookup_chunk.remote(chunk) for chunk in data_chunks]
    chunk_results = ray.get(chunk_futures)

    # Merge results
    episode_lookup = {}
    for chunk_result in chunk_results:
        episode_lookup.update(chunk_result)
    return episode_lookup


@ray.remote
def discover_episodes_chunk(chunk_dir: str) -> list[str]:
    episodes = []
    for file in list_directory(chunk_dir):
        if file.endswith(".parquet") and "episode_" in file:
            match = re.search(r"episode_(\d+)\.parquet", file)
            if match:
                episodes.append(f"{chunk_dir.rstrip('/')}/{file}")
    return episodes


def discover_image_columns(data_chunks: list[str], episode_file_pattern: str) -> list[str]:
    """Discover image columns by checking first available episode."""
    for chunk_dir in data_chunks:
        for episode_idx in range(10):
            episode_path = f"{chunk_dir.rstrip('/')}/{episode_file_pattern.format(episode_idx)}"
            if not file_exists(episode_path):
                continue

            # Read parquet to examine columns
            if episode_path.startswith("s3"):
                with copy_to_temp_file(episode_path) as temp_parquet:
                    df = pq.read_table(temp_parquet).to_pandas()
            else:
                df = pq.read_table(episode_path).to_pandas()

            # Find columns containing image bytes
            image_columns = []
            for col in df.columns:
                if df[col].dtype == object and len(df[col]) > 0:
                    first_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if isinstance(first_val, (dict, bytes)) and (isinstance(first_val, bytes) or "bytes" in first_val):
                        image_columns.append(col)

            if image_columns:
                print(f"Discovered image columns: {image_columns}")
                return image_columns

    print("Warning: Could not discover image columns")
    return []


def decode_video_frames(
    video_path: str,
    start_frame: int | None = None,
    num_frames: int | None = None,
    fps: float = 30.0,
) -> list[np.ndarray]:
    """Decode all frames from a video file, returning a list of (H, W, 3) RGB uint8 numpy arrays."""
    import av

    if video_path.startswith("s3"):
        with copy_to_temp_file(video_path) as local_path:
            return decode_video_frames(local_path, start_frame=start_frame, num_frames=num_frames, fps=fps)

    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        if start_frame is not None:
            if num_frames is None:
                raise ValueError("num_frames must be set when start_frame is set")
            start_seconds = start_frame / fps
            container.seek(int(start_seconds / float(stream.time_base)), stream=stream, backward=True)

        for frame in container.decode(stream):
            if start_frame is not None:
                frame_index = len(frames) if frame.time is None else round(float(frame.time) * fps)
                if frame_index < start_frame:
                    continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if num_frames is not None and len(frames) >= num_frames:
                break
    if num_frames is not None and len(frames) != num_frames:
        raise RuntimeError(f"Expected {num_frames} frames from {video_path}, decoded {len(frames)}")
    return frames


def discover_cameras(video_chunks: list[str]) -> dict[str, str]:
    """Discover available cameras by scanning video chunk directories.

    Returns a dict mapping camera directory name -> camera directory name.
    The directory name is the full name as it appears on disk (e.g., 'observation.image').
    """
    cameras = {}

    if not video_chunks:
        return cameras

    first_chunk = video_chunks[0]
    for item in list_directory(first_chunk):
        cameras[item] = item

    print(f"Discovered cameras: {list(cameras.keys())}")
    return cameras


@ray.remote
def build_video_lookup_chunk(chunk_dir: str, cameras: dict[str, str]) -> dict[tuple[int, str], str]:
    """Build video lookup for a single chunk directory."""
    video_lookup = {}
    for _camera_name, camera_path in cameras.items():
        camera_dir = f"{chunk_dir.rstrip('/')}/{camera_path}"
        files = list_directory(camera_dir)
        for file in files:
            if file.endswith(".mp4") and "episode_" in file:
                match = re.search(r"episode_(\d+)\.mp4", file)
                if match:
                    ep_num = int(match.group(1))
                    video_lookup[(ep_num, camera_path)] = f"{camera_dir}/{file}"
    return video_lookup


def build_video_lookup(video_chunks: list[str], cameras: dict[str, str]) -> dict[tuple[int, str], str]:
    print(f"Building video lookup dict from {len(video_chunks)} chunks and {len(cameras)} cameras...")
    # Process chunks in parallel
    chunk_futures = [build_video_lookup_chunk.remote(chunk, cameras) for chunk in video_chunks]
    chunk_results = ray.get(chunk_futures)

    # Merge results
    video_lookup = {}
    for chunk_result in chunk_results:
        video_lookup.update(chunk_result)
    return video_lookup


class LeRobotConverter(BaseRoboticsConverter):
    def __init__(self, cfg):
        super().__init__(cfg)

        source_dir_contents = list_directory(cfg.source_episodes[0])
        self.has_videos = "videos" in source_dir_contents

        self.meta_episodes_path = resolve_path(cfg.source_episodes[0], "meta/episodes.jsonl")
        self.compact_meta_episodes_path = resolve_path(cfg.source_episodes[0], "meta/episodes")
        self.info_path = resolve_path(cfg.source_episodes[0], "meta/info.json")
        self.tasks_path = resolve_path(cfg.source_episodes[0], "meta/tasks.jsonl")
        self.data_path = resolve_path(cfg.source_episodes[0], "data")
        self.videos_path = resolve_path(cfg.source_episodes[0], "videos")
        self.is_compact_v3 = not file_exists(self.meta_episodes_path)

        self.fps = detect_fps(self.info_path)

        if self.is_compact_v3:
            self.data_files = discover_parquet_files(self.data_path)
            self.entries_df = pd.concat(
                [read_parquet(path) for path in discover_parquet_files(self.compact_meta_episodes_path)],
                ignore_index=True,
            )
            self.entries_by_episode = {
                int(row["episode_index"]): row for row in self.entries_df.to_dict("records")
            }
            self.image_columns, self.cameras, self.video_lookup = [], {}, {}
            if self.has_videos:
                self.cameras = {camera_name: camera_name for camera_name in list_directory(self.videos_path)}
                if not cfg.camera_names:
                    raise ValueError(
                        "camera_names must be specified for video-based LeRobot datasets. "
                        f"Available cameras from video directories: {list(self.cameras.keys())}"
                    )
                unknown = set(cfg.camera_names) - set(self.cameras)
                if unknown:
                    raise KeyError(f"Camera(s) {unknown} not found in discovered cameras {list(self.cameras.keys())}")
            print(f"Loaded {len(self.entries_df)} compact LeRobot v3 episodes metadata")
        else:
            self.data_chunks = self.discover_chunks([self.data_path])
            if self.has_videos:
                self.video_chunks = self.discover_chunks([self.videos_path])
            else:
                self.video_chunks = []

            # Pre-build episode lookup. Maps from episode number to episode path.
            self.episode_lookup = build_episode_lookup(self.data_chunks)
            print(f"Built episode lookup with {len(self.episode_lookup)} episodes")

            # Pre-build image columns, cameras, and video lookup.
            self.image_columns, self.cameras, self.video_lookup = [], {}, {}
            if not self.has_videos:
                self.image_columns = discover_image_columns(self.data_chunks, "episode_{:06d}.parquet")
                if not self.image_columns or len(self.image_columns) == 0:
                    raise ValueError("No image columns found in parquet files and none specified with --image_columns")
            else:
                # Discover cameras once and reuse it for all shards.
                self.cameras = discover_cameras(self.video_chunks)
                assert len(self.cameras) > 0, "No cameras found"
                print(f"Discovered cameras: {list(self.cameras.keys())}")

                if not cfg.camera_names:
                    raise ValueError(
                        "camera_names must be specified for video-based LeRobot datasets. "
                        f"Available cameras from video directories: {list(self.cameras.keys())}"
                    )

                unknown = set(cfg.camera_names) - set(self.cameras)
                if unknown:
                    raise KeyError(f"Camera(s) {unknown} not found in discovered cameras {list(self.cameras.keys())}")

                # Pre-build video lookup once and reuse it for all shards.
                self.video_lookup = build_video_lookup(self.video_chunks, self.cameras)
                print(f"Built video lookup with {len(self.video_lookup)} video files")

            # Read episodes metadata
            self.entries = jsonl_load(self.meta_episodes_path)
            print(f"Loaded {len(self.entries)} episodes metadata")

    def get_language_instructions(self, sample_metadata) -> dict[str, str]:
        episode_index = sample_metadata["episode_index"]
        if self.is_compact_v3:
            tasks = self.entries_by_episode[int(episode_index)]["tasks"]
            if isinstance(tasks, np.ndarray):
                tasks = tasks.tolist()
            return {"original": list(tasks)[0]}
        return {"original": self.entries[episode_index]["tasks"][0]}

    def discover_chunks(self, source_paths: list[str]) -> list[str]:
        """Discover all chunk directories in the base path."""
        chunk_dirs = []
        for source_path in source_paths:
            for item in list_directory(source_path):
                if item.startswith("chunk-"):
                    chunk_path = f"{source_path.rstrip('/')}/{item}"
                    chunk_dirs.append(chunk_path)
        chunk_dirs.sort()  # Sort to ensure consistent ordering
        return chunk_dirs

    def discover_episodes(self, source_paths: list[str], max_episodes_to_process: int = -1) -> list[str]:
        if self.is_compact_v3:
            episodes_df = self.entries_df
            if self.cfg.task_filter is not None:
                episodes_df = episodes_df[
                    episodes_df["tasks"].apply(lambda tasks: task_list_contains(tasks, self.cfg.task_filter))
                ]
            if max_episodes_to_process > 0:
                episodes_df = episodes_df.head(max_episodes_to_process)
            return [f"compact_episode_{int(idx):06d}" for idx in episodes_df["episode_index"]]

        chunks = self.discover_chunks([os.path.join(source_paths[0], "data")])
        chunk_futures = [discover_episodes_chunk.remote(chunk) for chunk in chunks]
        chunk_results = ray.get(chunk_futures)

        all_episodes = []
        for chunk_result in chunk_results:
            all_episodes.extend(chunk_result)
        if self.cfg.task_filter is not None:
            allowed_episode_indices = {
                idx
                for idx, entry in enumerate(self.entries)
                if task_list_contains(entry["tasks"], self.cfg.task_filter)
            }
            all_episodes = [
                path
                for path in all_episodes
                if int(re.search(r"episode_(\d+)\.parquet", path).group(1)) in allowed_episode_indices
            ]
        if max_episodes_to_process > 0:
            all_episodes = all_episodes[:max_episodes_to_process]
        return all_episodes

    def get_episode_length(self, episode_data: Any) -> int:
        return len(episode_data)

    def load_episode_data(self, episode_path):
        if self.is_compact_v3:
            match = re.search(r"compact_episode_(\d+)", episode_path)
            if match is None:
                raise ValueError(f"Could not parse compact episode index from {episode_path}")
            episode_index = int(match.group(1))
            episode_row = self.entries_by_episode[episode_index]
            data_path = resolve_path(
                self.data_path,
                f"chunk-{int(episode_row['data/chunk_index']):03d}/"
                f"file-{int(episode_row['data/file_index']):03d}.parquet",
            )
            df = read_parquet(data_path)
            start = int(episode_row["dataset_from_index"])
            end = int(episode_row["dataset_to_index"])
            if "index" in df.columns:
                df = df[(df["index"] >= start) & (df["index"] < end)].copy()
            else:
                df = df.iloc[start:end].copy()
            if len(df) != int(episode_row["length"]):
                raise ValueError(
                    f"Compact episode {episode_index} expected {int(episode_row['length'])} rows, got {len(df)}"
                )
            df.attrs["compact_episode_row"] = episode_row
            return df

        df = read_parquet(episode_path)
        return df

    def extract_camera_data(self, episode_data: Any):
        """
        Here, camera_data values are bytes (inline images) or numpy arrays (video frames).
        """
        camera_data = {}

        if self.is_compact_v3 and self.has_videos:
            episode_row = episode_data.attrs["compact_episode_row"]
            num_rows = len(episode_data)
            for camera_name in self.cfg.camera_names:
                chunk_index = int(episode_row[f"videos/{camera_name}/chunk_index"])
                file_index = int(episode_row[f"videos/{camera_name}/file_index"])
                video_path = resolve_path(
                    self.videos_path,
                    f"{camera_name}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                )
                start_frame = round(float(episode_row[f"videos/{camera_name}/from_timestamp"]) * self.fps)
                camera_data[camera_name] = decode_video_frames(
                    video_path,
                    start_frame=start_frame,
                    num_frames=num_rows,
                    fps=self.fps,
                )
        elif self.has_videos:
            episode_index = int(episode_data["episode_index"].iloc[0])
            num_rows = len(episode_data)
            for camera_name in self.cfg.camera_names:
                video_path = self.video_lookup.get((episode_index, camera_name))
                if video_path is None:
                    print(f"Warning: No video found for episode {episode_index}, camera {camera_name}")
                    continue
                frames = decode_video_frames(video_path)
                if len(frames) != num_rows:
                    if len(frames) < num_rows:
                        raise ValueError(
                            f"Video {video_path} has {len(frames)} frames but parquet has {num_rows} rows. "
                            f"Video has fewer frames than expected — the dataset may be corrupted."
                        )
                    # Video has more frames than parquet rows, truncate to match
                    print(
                        f"Warning: Video has {len(frames)} frames but parquet has {num_rows} rows "
                        f"for episode {episode_index}, camera {camera_name}. Truncating video to match parquet."
                    )
                    frames = frames[:num_rows]
                camera_data[camera_name] = frames
        else:
            for camera_name in self.cfg.camera_names:
                images = episode_data[camera_name].to_list()
                if isinstance(images[0], dict) and "bytes" in images[0]:
                    images = [image["bytes"] for image in images]
                elif isinstance(images[0], bytes):
                    images = images
                else:
                    raise ValueError(f"Unsupported image data format in column {camera_name}")
                camera_data[camera_name] = images

        return camera_data

    def extract_lowdim_data(self, episode_data: Any):
        """
        Return a dictionary with lowdim keys as keys and lowdim data as values.
        """
        lowdim_cols = self.cfg.observation_keys + self.cfg.action_keys
        lowdim_data = {}
        lowdim_key_remap = self.cfg.lowdim_key_remap or {}
        for col in lowdim_cols:
            lowdim_data[lowdim_key_remap.get(col, col)] = np.stack(episode_data[col].to_numpy())
        return lowdim_data

    def extract_intrinsics_extrinsics_data(self, episode_data: Any):
        return None, None

    def extract_metadata_data(self, episode_data: Any):
        exclude_keys = self.cfg.observation_keys + self.cfg.action_keys + list(self.cfg.camera_names)
        metadata_data = {}
        for key, value in episode_data.items():
            if key not in exclude_keys:
                metadata_data[key] = value.to_numpy()
        return metadata_data

    def extract_sample_data(
        self,
        anchor_timestep: int,
        episode_path: str,
        episode_length: int,
        camera_data: dict[str, np.ndarray],
        lowdim_data: dict[str, np.ndarray],
        intrinsics_data: dict[str, np.ndarray],
        extrinsics_data: dict[str, np.ndarray],
        metadata_data: dict[str, Any],
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
        sample_images = {}
        actual_image_timesteps = []

        for img_offset in self.cfg.image_indices:
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

        # Build stats_sample for batched statistics update (don't send immediately)
        # Must be done before modifying sample_lowdim with masks
        stats_sample = (
            None
            if statistics_ray_actor is None
            else {
                "lowdim": {k: v.copy() for k, v in sample_lowdim.items()},  # Copy before modifying
                "past_mask": past_mask,
                "future_mask": future_mask,
            }
        )

        # Add masks to sample_lowdim (after building stats_sample)
        sample_lowdim["past_mask"] = past_mask
        sample_lowdim["future_mask"] = future_mask

        sample_metadata = {
            "camera_names": list(camera_data.keys()),
            "anchor_relative_idx": int(self.cfg.past_lowdim_steps),
        }
        for key, value in metadata_data.items():
            if isinstance(value, (list, np.ndarray)):
                sample_metadata[key] = value[anchor_timestep]
            else:
                sample_metadata[key] = value
        language_instructions = self.get_language_instructions(sample_metadata)

        return sample_images, sample_lowdim, sample_metadata, language_instructions, None, None, stats_sample
