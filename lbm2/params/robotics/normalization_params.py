from dataclasses import dataclass, field
from typing import Dict

from lbm2.params.base_params import BaseParams


@dataclass(frozen=True)
class FieldNormalizationParams:
    """Configuration for a specific field's normalization.

    We create a dictionary of these objects in NormalizationParams which is not properly serialized by draccus.
    So we use dataclasses_json to make sure the result is still properly serializable.
    """

    method: str = field(default="std")  # "std", "percentile_5_95", "percentile_1_99" "min_max"
    scope: str = field(default="global")  # "global" or "per_timestep"
    epsilon: float = field(default=1e-8)

    def to_dict(self):
        return {
            "method": self.method,
            "scope": self.scope,
            "epsilon": self.epsilon,
        }

    def __reduce__(self):
        """Control how this object is pickled/serialized.

        This helps avoid the !!python/object tag in YAML output.
        """
        # Just return the data as a tuple of (class, args)
        # This will make it serialize as a plain mapping
        return (self.__class__, (self.method, self.scope, self.epsilon))

    # This is what PyYAML will use for representing the object
    def __repr__(self):
        return str(self.to_dict())


@dataclass(frozen=True)
class NormalizationParams(BaseParams):
    enabled: bool = field(default=True)

    # Default parameters to be used for all fields if not specified in field_configs
    method: str = field(default="std")  # "std", "percentile_5_95", "percentile_1_99" "min_max"
    scope: str = field(default="global")  # "global" or "per_timestep"
    epsilon: float = field(default=1e-8)
    include_fields: list[str] = field(default_factory=list)

    # Field-specific configurations (initialized in __post_init__)
    field_configs: Dict[str, FieldNormalizationParams] = field(default_factory=dict)

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "method": self.method,
            "scope": self.scope,
            "epsilon": self.epsilon,
            "include_fields": self.include_fields,
            "field_configs": {k: v.to_dict() for k, v in self.field_configs.items()},
        }

    def __post_init__(self):
        self.check_asserts()

    def check_asserts(self):
        if self.method not in ["std", "percentile_5_95", "percentile_1_99", "min_max"]:
            raise ValueError(f"Invalid normalization method: {self.method}")
        if self.scope not in ["global", "per_timestep"]:
            raise ValueError(f"Invalid normalization scope: {self.scope}")

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        # Currently we don't support normalization of intrinsics and extrinsics fields
        object.__setattr__(self, "include_fields", cfg.data.proprioception_fields + cfg.data.action_fields)
