from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import draccus

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
    jpeg_quality: int = field(default=95)

    # Depth and point cloud control
    use_depth_data: bool = field(default=False)  # Whether to use depth data (depth images + point clouds)
    point_cloud_num_points: int = field(default=50000)  # Number of points for downsampling

    # Ray configuration
    ray_address: str = field(default=None)  # Ray cluster address, default to auto-detect
    ray_num_cpus: int = field(default=None)  # Number of CPUs for Ray, default to auto-detect

    def __post_init__(self):
        super().__post_init__()

        if self.type is None:
            object.__setattr__(self, "type", getattr(self, "_type", None))

        # Validate required paths
        assert self.source_episodes is not None, "--source_episodes is required (or set in config_path)"
        assert self.output_dir is not None, "--output_dir is required (or set in config_path)"


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


TYPE_MAPPER = {
    "spartan": SpartanPreprocessParams,
    "lerobot": LeRobotPreprocessParams,
    "mmt_npz": MMTPreprocessParams,
}
