from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lbm2.params.base_params import BaseParams


@dataclass
class SampleMetadata:
    """Metadata for each preprocessed sample."""

    episode_id: str
    sample_id: str
    anchor_timestep: int
    anchor_relative_idx: int
    image_timesteps: List[int]
    lowdim_start_timestep: int
    lowdim_end_timestep: int
    past_padding: int
    future_padding: int
    camera_names: List[str]
    original_episode_length: int
    original_image_sizes: Dict[str, Tuple[int, int]]
    is_padded: bool


@dataclass(frozen=True)
class PreprocessParams(BaseParams):
    """Dataclass for preprocessing configuration parsed by draccus."""

    # Core I/O
    source_episodes: Optional[List[str]] = field(default=None)
    output_dir: Optional[str] = field(default=None)

    # Sampling/windowing
    past_lowdim_steps: int = field(default=5)
    future_lowdim_steps: int = field(default=20)
    image_indices: List[int] = field(default_factory=lambda: [-5, 0])
    stride: int = field(default=1)
    max_padding_left: int = field(default=5)
    max_padding_right: int = field(default=5)
    padding_strategy: str = field(default="copy")  # one of: copy, zero, reflect

    # Filtering
    filter_still_samples: bool = field(default=True)
    still_threshold: float = field(default=0.01)

    # Cameras
    camera_names: Optional[List[str]] = field(default=None)
    discard_keys: Optional[List[str]] = field(default=None)

    # Sharding / compression
    samples_per_shard: int = field(default=100)
    jpeg_quality: int = field(default=95)

    # Runtime
    num_workers: int = field(default=40)
    max_episodes: int = field(default=-1)
    fail_on_nan: bool = field(default=False)
    shuffle_buffer_size: int = field(default=1000)
    shuffle_input_files: bool = field(default=True)

    # Statistics and reproducibility
    no_statistics: bool = field(default=False)
    no_auto_tag: bool = field(default=False)

    # Incremental updates and resume capability
    enable_incremental_updates: bool = field(default=True)
    update_frequency: int = field(default=5)  # Update metadata every N shards
    resume: bool = field(default=False)  # Whether to resume from existing progress

    # Testing flags
    skip_git_tagging: bool = field(default=False)  # Skip git operations for testing

    # Image preprocessing
    resize_images_size: int = field(default=0)
    use_gpu_resize: bool = field(default=True)

    # Language annotations
    language_annotations_path: str = field(
        default="lbm2/data/preprocessing/lbm_language_annotations.yaml",
    )
