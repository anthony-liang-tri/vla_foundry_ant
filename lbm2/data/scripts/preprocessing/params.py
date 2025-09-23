from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lbm2.params.base_params import BaseParams


@dataclass
class SampleMetadata:
    """Metadata for each preprocessed sample."""

    episode_id: str
    sample_id: str
    anchor_timestep: Optional[int]
    anchor_relative_idx: Optional[int]
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
    past_lowdim_steps: int = field(default=1)
    future_lowdim_steps: int = field(default=14)
    image_indices: List[int] = field(default_factory=lambda: [-1, 0])
    stride: int = field(default=1)
    max_padding_left: int = field(default=1)
    max_padding_right: int = field(default=7)
    padding_strategy: str = field(default="copy")  # one of: copy, zero, reflect

    # Filtering
    filter_still_samples: bool = field(default=False)
    still_threshold: float = field(default=0.01)

    # Cameras
    camera_names: Optional[List[str]] = field(default=None)
    camera_discard_keys: Optional[List[str]] = field(default=None)

    # Sharding / compression
    samples_per_shard: int = field(default=1)
    jpeg_quality: int = field(default=95)

    # Runtime
    max_episodes_to_process: int = field(default=-1)
    fail_on_nan: bool = field(default=False)

    # Statistics and reproducibility
    compute_statistics: bool = field(default=True)
    auto_tag: bool = field(default=True)

    # Testing flags
    skip_git_tagging: bool = field(default=False)  # Skip git operations for testing

    # Image preprocessing
    resize_images_size: List[int] = field(default=None)

    # Language annotations
    language_annotations_path: str = field(
        default="lbm2/data/preprocessing/lbm_language_annotations.yaml",
    )

    # Ray configuration
    ray_address: str = field(default=None)  # Ray cluster address, default to auto-detect
    ray_num_cpus: int = field(default=None)  # Number of CPUs for Ray, default to auto-detect
