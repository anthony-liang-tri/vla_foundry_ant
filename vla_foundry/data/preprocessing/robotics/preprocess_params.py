from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import draccus

from vla_foundry.data.preprocessing.image_utils import ImageResizingMethod
from vla_foundry.params.base_params import BaseParams


def register_preprocess_params(key: str):
    """
    Registers a PreprocessParams subclass and sets its type attribute.
    Use decorator wrapper because draccus's class selection with --preprocess_params.type doesn't
    automatically populate the attribute cfg.preprocess_params.type
    """

    def decorator(cls):
        registered_cls = PreprocessParams.register_subclass(key)(cls)
        registered_cls._type = key
        return registered_cls

    return decorator


@dataclass(frozen=True)
class PreprocessParams(draccus.ChoiceRegistry, BaseParams):
    type: str = field(default=None)

    # Core I/O
    source_episodes: Optional[List[str]] = field(default=None)
    output_dir: Optional[str] = field(default=None)
    output_dir_fixed_path: Optional[str] = field(default="s3://tri-ml-datasets-uw2/vla_foundry_datasets_fixed/")

    # Sampling/windowing
    past_lowdim_steps: int = field(default=1)
    future_lowdim_steps: int = field(default=14)
    camera_names: Optional[List[str]] = field(default=None)
    image_indices: List[int] = field(default_factory=lambda: [-1, 0])
    stride: int = field(default=1)
    max_padding_left: int = field(default=1)
    max_padding_right: int = field(default=7)
    padding_strategy: str = field(default="copy")  # one of: copy, zero, reflect

    # Filtering
    filter_still_samples: bool = field(default=False)
    still_threshold: float = field(default=0.01)

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
    resize_images_size: Optional[List[int]] = field(default=None)
    image_resizing_method: ImageResizingMethod = ImageResizingMethod.CENTER_CROP
    jpeg_quality: int = field(default=95)

    # Depth and point cloud control
    use_depth_data: bool = field(default=False)  # Whether to use depth data (depth images + point clouds)
    point_cloud_num_points: int = field(default=4096)  # Total number of points (divided evenly across views)

    # Ray configuration
    ray_address: str = field(default=None)  # Ray cluster address, default to auto-detect
    ray_num_cpus: int = field(default=None)  # Number of CPUs for Ray, default to auto-detect

    # GPU allocation per worker when use_depth_data=True (for CUDA FPS point cloud processing)
    # Default: 0.25 GPU per worker (allows 4 workers per GPU)
    ray_num_gpus_per_worker: float = field(default=0.25)
    # Database logging
    db_logging: bool = field(default=True)  # Whether to log preprocessing to DynamoDB

    def __post_init__(self):
        super().__post_init__()

        if self.type is None:
            object.__setattr__(self, "type", getattr(self, "_type", None))

        # Validate required paths
        assert self.source_episodes is not None, "--source_episodes is required (or set in config_path)"
        assert self.output_dir is not None, "--output_dir is required (or set in config_path)"

        # Validate image resizing method (not a strictly necessary check due to enum)
        allowed_image_resizing_methods = set(ImageResizingMethod)
        if self.image_resizing_method not in allowed_image_resizing_methods:
            raise ValueError(
                f"image_resizing_method must be one of {[m.value for m in allowed_image_resizing_methods]}, "
                f"got {self.image_resizing_method.value}"
            )


@register_preprocess_params("spartan")
@dataclass(frozen=True)
class SpartanPreprocessParams(PreprocessParams):
    data_discard_keys: Optional[List[str]] = field(default=None)

    language_annotations_path: str = field(
        default="vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml",
    )
    action_fields_config_path: str = field(
        default="vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml",
    )
    validation_episodes_path: Optional[str] = field(default=None)

    # Depth filtering parameters
    min_depth: float = field(default=0.001)  # Minimum valid depth in meters (filters invalid/too-close points)
    max_depth: float = field(default=3.0)  # Maximum valid depth in meters (filters too-far/unreliable points)


@register_preprocess_params("lerobot")
@dataclass(frozen=True)
class LeRobotPreprocessParams(PreprocessParams):
    observation_keys: List[str] = field(default=None)
    action_keys: List[str] = field(default=None)


@dataclass(frozen=True)
class RangeSpec:
    """Dataclass for specifying a range with optional start, end, and step."""

    start: Optional[int] = field(default=None)
    end: Optional[int] = field(default=None)
    step: Optional[int] = field(default=None)


@register_preprocess_params("mmt_npz")
@dataclass(frozen=True)
class MMTPreprocessParams(PreprocessParams):
    """Dataclass for MMT-specific preprocessing configuration parsed by draccus."""

    mmt_lowdim_flatten_indices_selection: Optional[Dict[str, Union[int, List[int], List[RangeSpec]]]] = field(
        default_factory=dict
    )
    depth_resizing_mask_threshold: float = field(default=0.99)

    def __post_init__(self):
        super().__post_init__()

        if self.mmt_lowdim_flatten_indices_selection:
            selections = {}
            for k, v in self.mmt_lowdim_flatten_indices_selection.items():
                if isinstance(v, int):
                    v = [v]
                if not isinstance(v, list):
                    raise ValueError(
                        f"Invalid type for mmt_lowdim_flatten_indices_selection[{k}]: "
                        f"expected int or list, got {type(v)}"
                    )
                new_v = []
                for item in v:
                    if isinstance(item, RangeSpec):
                        if item.start is None or item.end is None:
                            raise ValueError(
                                f"RangeSpec for mmt_lowdim_flatten_indices_selection[{k}] must have start and end"
                            )
                        step = item.step if item.step is not None else 1
                        new_v.extend(list(range(item.start, item.end, step)))
                    elif isinstance(item, int):
                        new_v.append(item)
                    else:
                        raise ValueError(
                            f"Invalid item type in mmt_lowdim_flatten_indices_selection[{k}]: "
                            f"expected int or RangeSpec, got {type(item)}"
                        )
                selections[k] = new_v
            object.__setattr__(self, "mmt_lowdim_flatten_indices_selection", selections)


@register_preprocess_params("humanoid_everyday")
@dataclass(frozen=True)
class HumanoidEverydayPreprocessParams(PreprocessParams):
    """Dataclass for HumanoidEveryday (Unitree G1) dataset preprocessing configuration."""

    # Required fields — must be set via YAML config.
    use_depth_data: bool = field(default=None)
    resize_images_size: List[int] = field(default=None)
    depth_resolution: List[int] = field(default=None)  # Raw depth sensor resolution (H, W)

    # Filtering: restrict to specific tasks and/or episodes
    # e.g. task_filter: ["drag_a_white_board"] to only process that task
    task_filter: Optional[List[str]] = field(default=None)
    # e.g. episode_filter: ["episode_0", "episode_1"] to only process those episodes
    episode_filter: Optional[List[str]] = field(default=None)
    # e.g. embodiment_filter: ["h1"] to only process episodes with robot_type "h1"
    embodiment_filter: Optional[List[str]] = field(default=None)

    _REQUIRED_FIELDS = ("use_depth_data", "resize_images_size", "depth_resolution")

    def __post_init__(self):
        super().__post_init__()
        for field_name in self._REQUIRED_FIELDS:
            assert getattr(self, field_name) is not None, f"--{field_name} is required (or set in config_path)"


@register_preprocess_params("mcap")
@dataclass(frozen=True)
class MCAPPreprocessParams(PreprocessParams):
    """Configuration for MCAP ROS 2 bag preprocessing."""

    # Path to topics config (action_topics, state_topics, camera_topics, target_hz)
    action_fields_config_path: str = field(
        default="vla_foundry/config_presets/data/preprocessing/test/unitree_g1_mcap_topics.yaml",
    )

    # Path to camera names config (list of camera names)
    camera_names_config_path: str = field(
        default="vla_foundry/config_presets/data/preprocessing/test/unitree_g1_camera_names.yaml",
    )

    task_name: Optional[str] = field(
        default=None, metadata={"help": "Task name for language instruction (overrides config default_task_name)"}
    )


TYPE_MAPPER = {
    "spartan": SpartanPreprocessParams,
    "lerobot": LeRobotPreprocessParams,
    "mmt_npz": MMTPreprocessParams,
    "humanoid_everyday": HumanoidEverydayPreprocessParams,
    "mcap": MCAPPreprocessParams,
}
