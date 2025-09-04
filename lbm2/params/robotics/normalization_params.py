from dataclasses import dataclass, field
from typing import Dict

from lbm2.params.base_params import BaseParams


@dataclass(frozen=True)
class FieldNormalizationParams:
    """Configuration for a specific field's normalization.

    We create a dictionary of these objects in NormalizationParams which is not properly serialized by draccus.
    So we use dataclasses_json to make sure the result is still properly serializable.
    """

    method: str = field(default="std")  # "std", "percentile_5_95", "percentile_1_99"
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
    method: str = field(default="std")  # "std", "percentile_5_95", "percentile_1_99"
    scope: str = field(default="global")  # "global" or "per_timestep"
    epsilon: float = field(default=1e-8)

    # Field-specific configurations (initialized in __post_init__)
    field_configs: Dict[str, FieldNormalizationParams] = field(default_factory=dict)

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "method": self.method,
            "scope": self.scope,
            "epsilon": self.epsilon,
            "field_configs": {k: v.to_dict() for k, v in self.field_configs.items()},
        }
