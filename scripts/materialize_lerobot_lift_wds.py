#!/usr/bin/env python3
"""Materialize a TRI-ML/PretrainFinetune robosuite task as robotics WDS shards."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
from pathlib import Path

import av
import numpy as np
import pandas as pd

from vla_foundry.data.preprocessing.image_utils import (
    ImageResizingMethod,
    image_to_bytes,
    init_jpeg_encoder,
)
from vla_foundry.data.preprocessing.robotics.preprocess_masks import create_past_and_future_masks


REPO_ID = "TRI-ML/PretrainFinetune"
DEFAULT_TASK = "lift the red cube off the table"
CAMERAS = {
    "agentview": "videos/observation.images.agentview/chunk-000/file-000.mp4",
    "wrist": "videos/observation.images.wrist/chunk-000/file-000.mp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.home() / ".cache/huggingface/lerobot/TRI-ML/PretrainFinetune",
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--past-lowdim-steps", type=int, default=1)
    parser.add_argument("--future-lowdim-steps", type=int, default=14)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def decode_frames(video_path: Path, num_frames: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= num_frames:
                break
    if len(frames) != num_frames:
        raise RuntimeError(f"Expected {num_frames} frames from {video_path}, decoded {len(frames)}")
    return frames


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


def npz_bytes(values: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def window_lowdim(
    data: np.ndarray,
    anchor_idx: int,
    episode_length: int,
    past_steps: int,
    future_steps: int,
) -> np.ndarray:
    start = anchor_idx - past_steps
    end = anchor_idx + future_steps
    past_padding = max(0, -start)
    future_padding = max(0, end - episode_length + 1)
    valid_start = max(0, start)
    valid_end = min(episode_length - 1, end)
    window = data[valid_start : valid_end + 1]
    if past_padding or future_padding:
        window = np.pad(window, [(past_padding, future_padding), (0, 0)], mode="edge")
    expected = past_steps + future_steps + 1
    if window.shape[0] != expected:
        raise RuntimeError(f"Bad lowdim window length {window.shape[0]} != {expected}")
    return window.astype(np.float32, copy=False)


class MaskedStats:
    def __init__(self) -> None:
        self.sums: dict[str, np.ndarray] = {}
        self.sumsqs: dict[str, np.ndarray] = {}
        self.counts: dict[str, np.ndarray] = {}
        self.mins: dict[str, np.ndarray] = {}
        self.maxs: dict[str, np.ndarray] = {}

    def update(self, sample: dict[str, np.ndarray], mask: np.ndarray) -> None:
        mask_2d = mask[:, None]
        for key, values in sample.items():
            values64 = values.astype(np.float64, copy=False)
            if key not in self.sums:
                self.sums[key] = np.zeros_like(values64)
                self.sumsqs[key] = np.zeros_like(values64)
                self.counts[key] = np.zeros((values64.shape[0], 1), dtype=np.float64)
                self.mins[key] = np.full_like(values64, np.inf)
                self.maxs[key] = np.full_like(values64, -np.inf)

            masked = np.where(mask_2d, values64, 0.0)
            self.sums[key] += masked
            self.sumsqs[key] += masked * masked
            self.counts[key] += mask_2d
            self.mins[key] = np.minimum(self.mins[key], np.where(mask_2d, values64, np.inf))
            self.maxs[key] = np.maximum(self.maxs[key], np.where(mask_2d, values64, -np.inf))

    def as_json(self) -> dict[str, dict[str, list]]:
        output: dict[str, dict[str, list]] = {}
        for key in self.sums:
            counts = self.counts[key]
            safe_counts = np.where(counts > 0, counts, 1.0)
            means = self.sums[key] / safe_counts
            variance = np.where(
                counts > 1,
                (self.sumsqs[key] - counts * means * means) / np.where(counts > 1, counts - 1, 1.0),
                0.0,
            )
            variance = np.maximum(variance, 0.0)
            weights = counts[:, 0]
            weighted_mean = np.average(means, axis=0, weights=weights)
            mean_variance = np.average(variance, axis=0, weights=weights)
            variance_of_means = np.average((means - weighted_mean) ** 2, axis=0, weights=weights)

            output[key] = {
                "mean": weighted_mean.tolist(),
                "std": np.sqrt(np.maximum(mean_variance + variance_of_means, 0.0)).tolist(),
                "min": np.min(self.mins[key], axis=0).tolist(),
                "max": np.max(self.maxs[key], axis=0).tolist(),
                "mean_per_timestep": means.tolist(),
                "std_per_timestep": np.sqrt(variance).tolist(),
                "min_per_timestep": self.mins[key].tolist(),
                "max_per_timestep": self.maxs[key].tolist(),
                "count": counts[:, 0].astype(int).tolist(),
            }
        return output


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_dir} exists; pass --force to replace it")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    data = pd.read_parquet(args.source_root / "data/chunk-000/file-000.parquet")
    episodes = pd.read_parquet(args.source_root / "meta/episodes/chunk-000/file-000.parquet")
    task_episodes = episodes[episodes["tasks"].apply(lambda tasks: args.task in tasks)].copy()
    if task_episodes.empty:
        available_tasks = sorted({task for task_list in episodes["tasks"] for task in task_list})
        raise ValueError(f"No episodes found for task {args.task!r}. Available tasks: {available_tasks}")
    max_frame = int(task_episodes["dataset_to_index"].max())

    states = np.stack(data.iloc[:max_frame]["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(data.iloc[:max_frame]["action"].to_numpy()).astype(np.float32)
    camera_frames = {name: decode_frames(args.source_root / rel_path, max_frame) for name, rel_path in CAMERAS.items()}

    init_jpeg_encoder(args.jpeg_quality)
    resize_to = (args.image_size, args.image_size)
    stats = MaskedStats()
    manifest: list[dict[str, int | str]] = []
    shard_idx = 0
    shard_count = 0
    total_count = 0
    shard_tar: tarfile.TarFile | None = None

    def open_shard(index: int) -> tarfile.TarFile:
        return tarfile.open(args.output_dir / f"shard_{index:06d}.tar", mode="w")

    try:
        shard_tar = open_shard(shard_idx)
        for episode in task_episodes.itertuples(index=False):
            episode_index = int(episode.episode_index)
            episode_start = int(episode.dataset_from_index)
            episode_length = int(episode.length)

            for frame_idx in range(episode_length):
                if shard_count == args.samples_per_shard:
                    assert shard_tar is not None
                    shard_tar.close()
                    manifest.append({"shard": f"shard_{shard_idx:06d}", "num_sequences": shard_count})
                    shard_idx += 1
                    shard_count = 0
                    shard_tar = open_shard(shard_idx)

                assert shard_tar is not None
                global_idx = episode_start + frame_idx
                base = f"episode_{episode_index:06d}_frame_{frame_idx:06d}"

                for offset in (-1, 0):
                    image_idx = min(max(frame_idx + offset, 0), episode_length - 1)
                    global_image_idx = episode_start + image_idx
                    for camera_name in CAMERAS:
                        image_bytes, _ = image_to_bytes(
                            camera_frames[camera_name][global_image_idx],
                            quality=args.jpeg_quality,
                            target_size=resize_to,
                            resize_method=ImageResizingMethod.CENTER_CROP,
                        )
                        add_bytes(shard_tar, f"{base}.{camera_name}_t{offset}.jpg", image_bytes)

                lowdim = {
                    "state": window_lowdim(
                        states[episode_start : episode_start + episode_length],
                        frame_idx,
                        episode_length,
                        args.past_lowdim_steps,
                        args.future_lowdim_steps,
                    ),
                    "actions": window_lowdim(
                        actions[episode_start : episode_start + episode_length],
                        frame_idx,
                        episode_length,
                        args.past_lowdim_steps,
                        args.future_lowdim_steps,
                    ),
                }
                past_mask, future_mask = create_past_and_future_masks(
                    frame_idx,
                    args.past_lowdim_steps,
                    args.future_lowdim_steps,
                    episode_length,
                )
                stats.update(lowdim, np.logical_or(past_mask, future_mask))
                lowdim["past_mask"] = past_mask
                lowdim["future_mask"] = future_mask
                add_bytes(shard_tar, f"{base}.lowdim.npz", npz_bytes(lowdim))

                metadata = {
                    "repo_id": REPO_ID,
                    "episode_index": episode_index,
                    "frame_index": frame_idx,
                    "dataset_index": int(global_idx),
                    "task": args.task,
                    "camera_names": list(CAMERAS.keys()),
                    "anchor_relative_idx": args.past_lowdim_steps,
                }
                add_bytes(shard_tar, f"{base}.metadata.json", json.dumps(metadata, indent=2).encode("utf-8"))
                add_bytes(
                    shard_tar,
                    f"{base}.language_instructions.json",
                    json.dumps({"original": args.task}, indent=2).encode("utf-8"),
                )

                shard_count += 1
                total_count += 1
    finally:
        if shard_tar is not None:
            shard_tar.close()

    if shard_count:
        manifest.append({"shard": f"shard_{shard_idx:06d}", "num_sequences": shard_count})

    with open(args.output_dir / "manifest.jsonl", "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    with open(args.output_dir / "stats.json", "w") as f:
        json.dump(stats.as_json(), f)
    preprocessing_config = {
        "type": "lerobot_v3",
        "source_repo_id": REPO_ID,
        "task": args.task,
        "state_key": "observation.state",
        "action_key": "action",
        "camera_map": {
            "observation.images.agentview": "agentview",
            "observation.images.wrist": "wrist",
        },
        "camera_names": list(CAMERAS.keys()),
        "image_indices": [-1, 0],
        "past_lowdim_steps": args.past_lowdim_steps,
        "future_lowdim_steps": args.future_lowdim_steps,
        "resize_images_size": [args.image_size, args.image_size],
        "samples_per_shard": args.samples_per_shard,
        "max_episodes_to_process": -1,
    }
    with open(args.output_dir / "preprocessing_config.yaml", "w") as f:
        json.dump(preprocessing_config, f, indent=2)
    with open(args.output_dir / "processing_metadata.json", "w") as f:
        json.dump(
            {
                "source_root": str(args.source_root),
                "repo_id": REPO_ID,
                "task": args.task,
                "num_episodes": int(len(task_episodes)),
                "total_samples_created": total_count,
                "num_shards": len(manifest),
            },
            f,
            indent=2,
        )

    print(f"Wrote {total_count} samples across {len(manifest)} shards to {args.output_dir}")


if __name__ == "__main__":
    main()
