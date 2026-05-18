from dataclasses import dataclass, field

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
    source_episodes: list[str] | None = field(default=None)
    output_dir: str | None = field(default=None)
    output_dir_fixed_path: str | None = field(default="s3://tri-ml-datasets-uw2/vla_foundry_datasets_fixed/")

    # Sampling/windowing
    past_lowdim_steps: int = field(default=1)
    future_lowdim_steps: int = field(default=14)
    camera_names: list[str] | None = field(default=None)
    # If True, skip episodes that don't have ALL requested cameras.
    # If False (default), process episodes with whatever cameras are available and warn about missing ones.
    skip_episodes_missing_cameras: bool = field(default=False)
    image_indices: list[int] = field(default_factory=lambda: [-1, 0])
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
    resize_images_size: list[int] | None = field(default=None)
    image_resizing_method: ImageResizingMethod = ImageResizingMethod.CENTER_CROP
    camera_rotations: dict[str, int] | None = field(default=None)
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
    data_discard_keys: list[str] | None = field(default=None)

    language_annotations_path: str = field(
        default="vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml",
    )
    action_fields_config_path: str = field(
        default="vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml",
    )
    validation_episodes_path: str | None = field(default=None)

    # Depth filtering parameters
    min_depth: float = field(default=0.001)  # Minimum valid depth in meters (filters invalid/too-close points)
    max_depth: float = field(default=3.0)  # Maximum valid depth in meters (filters too-far/unreliable points)


@register_preprocess_params("lerobot")
@dataclass(frozen=True)
class LeRobotPreprocessParams(PreprocessParams):
    observation_keys: list[str] = field(default=None)
    action_keys: list[str] = field(default=None)
    task_filter: str | None = field(default=None)
    lowdim_key_remap: dict[str, str] | None = field(default=None)


@dataclass(frozen=True)
class RangeSpec:
    """Dataclass for specifying a range with optional start, end, and step."""

    start: int | None = field(default=None)
    end: int | None = field(default=None)
    step: int | None = field(default=None)

    def to_dict(self):
        return {"start": self.start, "end": self.end, "step": self.step}


@dataclass(frozen=True)
class FieldRemapEntry:
    """A single remap target: source_field -> to (new name) with index selection."""

    to: str | None = field(default=None)
    indices: list[int | RangeSpec] = field(default_factory=list)

    def to_dict(self):
        return {"to": self.to, "indices": [i.to_dict() if hasattr(i, "to_dict") else i for i in self.indices]}


def _resolve_indices(items: list[int | RangeSpec], context: str) -> list[int]:
    """Resolve a list of ints and RangeSpecs into a flat list of indices."""
    result = []
    for item in items:
        if isinstance(item, RangeSpec):
            if item.start is None or item.end is None:
                raise ValueError(f"RangeSpec for {context} must have start and end")
            step = item.step if item.step is not None else 1
            result.extend(list(range(item.start, item.end, step)))
        elif isinstance(item, int):
            result.append(item)
        else:
            raise ValueError(f"Invalid item type in {context}: expected int or RangeSpec, got {type(item)}")
    return result


@register_preprocess_params("mmt_npz")
@dataclass(frozen=True)
class MMTPreprocessParams(PreprocessParams):
    """Dataclass for MMT-specific preprocessing configuration parsed by draccus."""

    mmt_lowdim_flatten_indices_selection: dict[str, int | list[int] | list[RangeSpec]] | None = field(
        default_factory=dict
    )
    lowdim_field_remap: dict[str, list[FieldRemapEntry]] | None = field(default=None)
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
                selections[k] = _resolve_indices(v, f"mmt_lowdim_flatten_indices_selection[{k}]")
            object.__setattr__(self, "mmt_lowdim_flatten_indices_selection", selections)

        if self.lowdim_field_remap:
            # Validate: remap source keys must not also have flatten index selection
            if self.mmt_lowdim_flatten_indices_selection:
                conflict = set(self.lowdim_field_remap.keys()) & set(self.mmt_lowdim_flatten_indices_selection.keys())
                if conflict:
                    raise ValueError(
                        f"Fields {conflict} are in both lowdim_field_remap and "
                        f"mmt_lowdim_flatten_indices_selection. Remap indices assume the "
                        f"original field layout, so flatten selection must not be applied "
                        f"to remap source fields."
                    )
            # Resolve into flat structure: {to_name: (source_key, [indices])}
            resolved = {}
            for source_key, entries in self.lowdim_field_remap.items():
                for entry in entries:
                    if not entry.to:
                        raise ValueError(f"lowdim_field_remap[{source_key}] entry must have a 'to' field")
                    if entry.to in resolved:
                        existing_source = resolved[entry.to][0]
                        raise ValueError(
                            f"Duplicate remap target '{entry.to}': "
                            f"defined in both '{existing_source}' and '{source_key}'"
                        )
                    resolved[entry.to] = (
                        source_key,
                        _resolve_indices(entry.indices, f"lowdim_field_remap[{source_key}] -> {entry.to}"),
                    )
            object.__setattr__(self, "_resolved_remap", resolved)
            # Replace lowdim_field_remap with plain dicts so that
            # vars(cfg) is serializable by both yaml.dump and json.dumps.
            plain_remap = {
                source_key: [
                    {"to": entry.to, "indices": _resolve_indices(entry.indices, source_key)} for entry in entries
                ]
                for source_key, entries in self.lowdim_field_remap.items()
            }
            object.__setattr__(self, "lowdim_field_remap", plain_remap)


@register_preprocess_params("humanoid_everyday")
@dataclass(frozen=True)
class HumanoidEverydayPreprocessParams(PreprocessParams):
    """Dataclass for HumanoidEveryday (Unitree G1) dataset preprocessing configuration."""

    # Required fields — must be set via YAML config.
    use_depth_data: bool = field(default=None)
    resize_images_size: list[int] = field(default=None)
    depth_resolution: list[int] = field(default=None)  # Raw depth sensor resolution (H, W)

    # Filtering: restrict to specific tasks and/or episodes
    # e.g. task_filter: ["drag_a_white_board"] to only process that task
    task_filter: list[str] | None = field(default=None)
    # e.g. episode_filter: ["episode_0", "episode_1"] to only process those episodes
    episode_filter: list[str] | None = field(default=None)
    # e.g. embodiment_filter: ["h1"] to only process episodes with robot_type "h1"
    embodiment_filter: list[str] | None = field(default=None)

    _REQUIRED_FIELDS = ("use_depth_data", "resize_images_size", "depth_resolution")

    def __post_init__(self):
        super().__post_init__()
        for field_name in self._REQUIRED_FIELDS:
            assert getattr(self, field_name) is not None, f"--{field_name} is required (or set in config_path)"


@register_preprocess_params("mcap")
@dataclass(frozen=True)
class MCAPPreprocessParams(PreprocessParams):
    """Configuration for MCAP ROS 2 bag preprocessing."""

    # Path to topics config (action_topics, state_topics, camera_topics)
    topics_to_fields_path: str = field(
        default="vla_foundry/config_presets/data/unitree_g1/g1_mcap_topics.yaml",
    )
    # Path to the action fields and slices config
    action_fields_config_path: str = field(
        default="vla_foundry/config_presets/data/unitree_g1/g1_action_fields.yaml",
    )
    # Path to the language annotations as a fallback if not available in the metadata
    language_annotations_path: str = field(
        default="vla_foundry/config_presets/data/unitree_g1/g1_language_annotations.yaml",
    )

    # Filtering: restrict to specific data sources, tasks, domains, and/or episodes
    # e.g. source_filter: ["teleop"] to only process teleop episodes
    source_filter: list[str] = field(default_factory=lambda: ["teleop"])
    # e.g., task_filter: ["do_something_useful"] to only process that task
    task_filter: list[str] | None = field(default=None)
    # e.g. domain_filter: ["sim"] to only process simulation episodes
    domain_filter: list[str] | None = field(default=None)


TYPE_MAPPER = {
    "spartan": SpartanPreprocessParams,
    "lerobot": LeRobotPreprocessParams,
    "mmt_npz": MMTPreprocessParams,
    "humanoid_everyday": HumanoidEverydayPreprocessParams,
    "mcap": MCAPPreprocessParams,
}
