import datetime
import io
import json
import os
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional torch for GPU-accelerated resize
import torch

from lbm2.data.preprocessing.image_utils import image_to_bytes


class StreamingShardWriter:
    """Memory-efficient streaming shard writer with optional background S3 uploads and incremental updates."""

    def __init__(
        self,
        output_dir: str,
        samples_per_shard: int,
        jpeg_quality: int = 95,
        gpu_resize: bool = False,
        upload_workers: int = 16,
        enable_incremental_updates: bool = True,
        update_frequency: int = 5,  # Update metadata every N shards
        resume: bool = False,  # Whether to resume from existing progress
    ):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.jpeg_quality = jpeg_quality
        self.gpu_resize = gpu_resize and (torch is not None) and torch.cuda.is_available()
        self.current_shard_idx = 0
        self.current_shard_samples = 0
        self.current_shard_file = None
        self.current_shard_tar = None
        self.manifest_data = []
        self.enable_incremental_updates = enable_incremental_updates
        self.update_frequency = update_frequency
        self.resume = resume
        self.total_samples_written = 0
        self.processed_episodes = set()  # Track processed episodes for resume capability

        # Filtering statistics for recovery
        self.total_potential_samples = 0
        self.total_still_filtered = 0
        self.total_padding_filtered = 0

        # Setup output directory
        self.is_s3_output = output_dir.startswith("s3://")
        if self.is_s3_output:
            self.temp_dir = tempfile.mkdtemp()
            self.shard_dir = self.temp_dir
        else:
            self.shard_dir = output_dir
            os.makedirs(self.shard_dir, exist_ok=True)

        # Initialize manifest file for incremental updates
        self.manifest_path = os.path.join(self.shard_dir, "manifest.jsonl")
        self.progress_path = os.path.join(self.shard_dir, "processing_progress.json")

        # Try to resume from existing progress
        self._try_resume_from_existing()

        # Initialize async S3 upload machinery if needed
        if self.is_s3_output:
            self._init_s3_transfer(upload_workers)

    def _try_resume_from_existing(self):
        """Try to resume from existing processing progress."""
        if not self.enable_incremental_updates or not self.resume:
            return

        try:
            # Check if progress file exists
            if os.path.exists(self.progress_path):
                with open(self.progress_path, "r") as f:
                    progress = json.load(f)

                # Resume from where we left off
                self.current_shard_idx = progress.get("next_shard_idx", 0)
                self.total_samples_written = progress.get("total_samples_written", 0)
                self.processed_episodes = set(progress.get("processed_episodes", []))

                # Restore filtering statistics if available
                filtering_stats = progress.get("filtering_statistics", {})
                self.total_potential_samples = filtering_stats.get("total_potential", 0)
                self.total_still_filtered = filtering_stats.get("total_still_filtered", 0)
                self.total_padding_filtered = filtering_stats.get("total_padding_filtered", 0)

                # Load existing manifest data
                if os.path.exists(self.manifest_path):
                    self.manifest_data = []
                    with open(self.manifest_path, "r") as f:
                        for line in f:
                            if line.strip():
                                self.manifest_data.append(json.loads(line.strip()))

                print(
                    f"📂 Resuming from shard {self.current_shard_idx}, {self.total_samples_written} samples processed"
                )
                print(f"📂 {len(self.processed_episodes)} episodes already processed")
                if self.total_potential_samples > 0:
                    print(
                        f"📊 Filtering stats recovered: {self.total_potential_samples} potential, "
                        f"{self.total_still_filtered} still filtered, {self.total_padding_filtered} padding filtered"
                    )

        except Exception as e:
            print(f"Warning: Could not resume from existing progress: {e}")
            # Reset to start from beginning
            self.current_shard_idx = 0
            self.total_samples_written = 0
            self.manifest_data = []

    def _update_progress(self):
        """Update processing progress file."""
        if not self.enable_incremental_updates:
            return

        progress = {
            "next_shard_idx": self.current_shard_idx,
            "total_samples_written": self.total_samples_written,
            "processed_episodes": list(self.processed_episodes),
            "filtering_statistics": {
                "total_potential": self.total_potential_samples,
                "total_still_filtered": self.total_still_filtered,
                "total_padding_filtered": self.total_padding_filtered,
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }

        with open(self.progress_path, "w") as f:
            json.dump(progress, f, indent=2)

        # Upload progress file if using S3
        if self.is_s3_output:
            self._schedule_upload(self.progress_path, os.path.basename(self.progress_path))

    def _append_to_manifest(self, shard_entry: Dict[str, Any]):
        """Append a shard entry to the manifest file."""
        if not self.enable_incremental_updates:
            return

        # Append to manifest file
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(shard_entry, separators=(",", ":")) + "\n")

        # Upload updated manifest if using S3
        if self.is_s3_output:
            self._schedule_upload(self.manifest_path, os.path.basename(self.manifest_path))

    def update_metadata_files(
        self, metadata: Dict[str, Any], statistics: Optional[Dict[str, Any]] = None, statistics_state=None
    ):
        """Update metadata and statistics files incrementally."""
        if not self.enable_incremental_updates:
            return

        # Update processing metadata
        metadata_path = os.path.join(self.shard_dir, "processing_metadata.json")

        # Update current progress in metadata
        if "processing" in metadata:
            metadata["processing"]["total_samples_created"] = self.total_samples_written
            metadata["processing"]["shards_completed"] = len(self.manifest_data)
            metadata["processing"]["last_updated"] = datetime.datetime.now().isoformat()

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.is_s3_output:
            self._schedule_upload(metadata_path, os.path.basename(metadata_path))

        # Update statistics if provided
        if statistics:
            stats_path = os.path.join(self.shard_dir, "dataset_statistics.json")
            with open(stats_path, "w") as f:
                json.dump(statistics, f, indent=2)

            if self.is_s3_output:
                self._schedule_upload(stats_path, os.path.basename(stats_path))

        # Save statistics state for recovery if provided
        if statistics_state:
            self.save_statistics_state(statistics_state)

    def save_statistics_state(self, statistics_state):
        """Save the current statistics computation state for recovery."""
        if not self.enable_incremental_updates:
            return

        stats_state_path = os.path.join(self.shard_dir, "processing_statistics.json")
        statistics_state.save_state(stats_state_path)

        if self.is_s3_output:
            self._schedule_upload(stats_state_path, os.path.basename(stats_state_path))

    def update_filtering_statistics(self, potential_samples: int, still_filtered: int, padding_filtered: int):
        """Update filtering statistics for recovery."""
        self.total_potential_samples += potential_samples
        self.total_still_filtered += still_filtered
        self.total_padding_filtered += padding_filtered

    def mark_episode_completed(self, episode_path: str):
        """Mark an episode as completed for resume capability."""
        if self.enable_incremental_updates:
            episode_id = os.path.basename(episode_path.rstrip("/"))
            self.processed_episodes.add(episode_id)

    def is_episode_processed(self, episode_path: str) -> bool:
        """Check if an episode has already been processed."""
        if not self.enable_incremental_updates:
            return False
        episode_id = os.path.basename(episode_path.rstrip("/"))
        return episode_id in self.processed_episodes

    def get_unprocessed_episodes(self, all_episodes: List[str]) -> List[str]:
        """Filter out already processed episodes."""
        if not self.enable_incremental_updates:
            return all_episodes

        unprocessed = []
        skipped_count = 0

        for episode in all_episodes:
            if not self.is_episode_processed(episode):
                unprocessed.append(episode)
            else:
                skipped_count += 1

        if skipped_count > 0:
            print(f"📂 Skipping {skipped_count} already processed episodes")

        return unprocessed

    def _init_s3_transfer(self, upload_workers: int):
        s3_path_clean = self.output_dir[5:]
        self.s3_bucket = s3_path_clean.split("/")[0]
        self.s3_prefix = "/".join(s3_path_clean.split("/")[1:])
        if self.s3_prefix and not self.s3_prefix.endswith("/"):
            self.s3_prefix += "/"

        import boto3
        from boto3.s3.transfer import S3Transfer, TransferConfig

        self._s3_client = boto3.client("s3")
        self._s3_transfer = S3Transfer(
            self._s3_client,
            config=TransferConfig(
                use_threads=True,
                max_concurrency=min(64, max(8, upload_workers)),
                multipart_threshold=8 * 1024 * 1024,
            ),
        )
        self._upload_executor = ThreadPoolExecutor(max_workers=min(64, max(8, upload_workers)))
        self._upload_futures = []

    def _schedule_upload(self, local_file: str, relative_path: Optional[str] = None):
        if not self.is_s3_output:
            return
        if relative_path is None:
            relative_path = os.path.relpath(local_file, self.shard_dir)
        s3_key = f"{self.s3_prefix}{relative_path}"
        future = self._upload_executor.submit(self._s3_transfer.upload_file, local_file, self.s3_bucket, s3_key)
        self._upload_futures.append(future)

    def upload_path(self, local_file: str, relative_path: Optional[str] = None):
        """Public API to queue an upload of any file under `shard_dir`."""
        self._schedule_upload(local_file, relative_path)

    def wait_for_uploads(self):
        if not self.is_s3_output:
            return
        for f in self._upload_futures:
            f.result()
        self._upload_futures = []

    def _start_new_shard(self):
        """Start a new shard file."""
        if self.current_shard_tar:
            self.current_shard_tar.close()

        shard_name = f"shard_{self.current_shard_idx:08d}"
        self.current_shard_file = os.path.join(self.shard_dir, f"{shard_name}.tar")
        self.current_shard_tar = tarfile.open(self.current_shard_file, "w")  # noqa SIM115
        self.current_shard_samples = 0

    def add_sample(self, sample: Dict[str, Any], target_image_size: Optional[Tuple[int, int]] = None):
        """Add a sample to current shard."""
        # Start new shard if needed
        if self.current_shard_tar is None or (
            self.samples_per_shard > 0 and self.current_shard_samples >= self.samples_per_shard
        ):
            if self.current_shard_tar:
                # Close previous shard and update manifest
                closed_shard_path = self.current_shard_file
                self.current_shard_tar.close()

                # Create shard entry
                shard_entry = {
                    "shard": f"shard_{self.current_shard_idx:08d}",
                    "num_sequences": self.current_shard_samples,
                }
                self.manifest_data.append(shard_entry)

                # Incrementally update manifest file
                self._append_to_manifest(shard_entry)

                # Queue upload of the closed shard if writing to S3
                if self.is_s3_output and closed_shard_path is not None:
                    self._schedule_upload(closed_shard_path, os.path.basename(closed_shard_path))

                self.current_shard_idx += 1

                # Update progress after completing a shard
                self._update_progress()

            self._start_new_shard()

        sample_id = sample["metadata"].sample_id

        original_image_sizes = {}
        for img_key, img_data in sample["images"].items():
            img_bytes, original_image_size = image_to_bytes(img_data, self.jpeg_quality, target_size=target_image_size)
            info = tarfile.TarInfo(name=f"{sample_id}.{img_key}.jpg")
            info.size = len(img_bytes)
            self.current_shard_tar.addfile(tarinfo=info, fileobj=io.BytesIO(img_bytes))
            img_key_no_t = img_key.split("_t")[0]
            original_image_sizes[img_key_no_t] = original_image_size

        # Write low-dim data as NPZ
        lowdim_buf = io.BytesIO()
        np.savez_compressed(lowdim_buf, **sample["lowdim"])  # Use compression
        lowdim_buf.seek(0)
        info = tarfile.TarInfo(name=f"{sample_id}.lowdim.npz")
        info.size = len(lowdim_buf.getbuffer())
        self.current_shard_tar.addfile(tarinfo=info, fileobj=lowdim_buf)

        # Write masks as NPZ
        masks_buf = io.BytesIO()
        np.savez_compressed(masks_buf, past_mask=sample["past_mask"], future_mask=sample["future_mask"])
        masks_buf.seek(0)
        info = tarfile.TarInfo(name=f"{sample_id}.masks.npz")
        info.size = len(masks_buf.getbuffer())
        self.current_shard_tar.addfile(tarinfo=info, fileobj=masks_buf)

        # Write intrinsics as NPZ (if available)
        if "intrinsics" in sample and sample["intrinsics"]:
            intrinsics_buf = io.BytesIO()
            np.savez_compressed(intrinsics_buf, **sample["intrinsics"])
            intrinsics_buf.seek(0)
            info = tarfile.TarInfo(name=f"{sample_id}.intrinsics.npz")
            info.size = len(intrinsics_buf.getbuffer())
            self.current_shard_tar.addfile(tarinfo=info, fileobj=intrinsics_buf)

        # Write extrinsics as NPZ (if available)
        if "extrinsics" in sample and sample["extrinsics"]:
            extrinsics_buf = io.BytesIO()
            np.savez_compressed(extrinsics_buf, **sample["extrinsics"])
            extrinsics_buf.seek(0)
            info = tarfile.TarInfo(name=f"{sample_id}.extrinsics.npz")
            info.size = len(extrinsics_buf.getbuffer())
            self.current_shard_tar.addfile(tarinfo=info, fileobj=extrinsics_buf)

        if "language_instructions" in sample:
            language_instructions_buf = io.BytesIO()
            json_str = json.dumps(sample["language_instructions"])
            language_instructions_buf.write(json_str.encode("utf-8"))
            language_instructions_buf.seek(0)
            info = tarfile.TarInfo(name=f"{sample_id}.language_instructions.json")
            info.size = len(language_instructions_buf.getbuffer())
            self.current_shard_tar.addfile(tarinfo=info, fileobj=language_instructions_buf)

        metadata_dict = sample["metadata"].__dict__
        metadata_dict["original_image_sizes"] = original_image_sizes

        metadata_json = json.dumps(metadata_dict, separators=(",", ":")).encode("utf-8")  # Compact JSON
        metadata_buf = io.BytesIO(metadata_json)
        info = tarfile.TarInfo(name=f"{sample_id}.metadata.json")
        info.size = len(metadata_buf.getbuffer())
        self.current_shard_tar.addfile(tarinfo=info, fileobj=metadata_buf)

        self.current_shard_samples += 1
        self.total_samples_written += 1

    def finalize(self) -> Tuple[List[Dict[str, Any]], str]:
        """Finalize writing and return manifest."""
        if self.current_shard_tar:
            closed_shard_path = self.current_shard_file
            self.current_shard_tar.close()

            # Create final shard entry
            shard_entry = {"shard": f"shard_{self.current_shard_idx:08d}", "num_sequences": self.current_shard_samples}
            self.manifest_data.append(shard_entry)

            # Update manifest incrementally if enabled
            if self.enable_incremental_updates:
                self._append_to_manifest(shard_entry)

            if self.is_s3_output and closed_shard_path is not None:
                self._schedule_upload(closed_shard_path, os.path.basename(closed_shard_path))

        # Write complete manifest (for non-incremental mode or as backup)
        if not self.enable_incremental_updates:
            with open(self.manifest_path, "w") as f:
                for entry in self.manifest_data:
                    f.write(json.dumps(entry, separators=(",", ":")) + "\n")

            # Queue manifest upload as well
            if self.is_s3_output:
                self._schedule_upload(self.manifest_path, os.path.basename(self.manifest_path))

        # Final progress update
        self._update_progress()

        return self.manifest_data, self.shard_dir

    def cleanup(self):
        """Cleanup temporary files. Ensures all uploads are complete first."""
        if self.is_s3_output and hasattr(self, "temp_dir"):
            self.wait_for_uploads()
            import shutil

            shutil.rmtree(self.temp_dir)
