from dataclasses import dataclass, field
from typing import List, Optional

from vla_foundry.params.base_params import BaseParams


@dataclass(frozen=True)
class PreprocessParams(BaseParams):
    """Dataclass for preprocessing configuration parsed by draccus."""

    # Core I/O
    source_type: str = field(default=None)
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

    # Field names
    camera_names: Optional[List[str]] = field(default=None)
    data_discard_keys: Optional[List[str]] = field(default=None)

    # Sharding / parallelism
    samples_per_shard: int = field(default=128)
    num_workers: int = field(default=20)

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
    jpeg_quality: int = field(default=95)

    # Language annotations
    language_annotations_path: str = field(
        default="vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml",
    )

    action_fields_config_path: str = field(
        default="vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml",
    )

    validation_episodes_path: Optional[str] = field(default=None)

    # Ray configuration
    ray_address: str = field(default=None)  # Ray cluster address, default to auto-detect
    ray_num_cpus: int = field(default=None)  # Number of CPUs for Ray, default to auto-detect

    def __post_init__(self):
        super().__post_init__()

        # Validate required paths
        assert self.source_episodes is not None, "--source_episodes is required (or set in config_path)"
        assert self.output_dir is not None, "--output_dir is required (or set in config_path)"
