"""
Normalization utilities for robotics data.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import draccus
import torch

from lbm2.file_utils import json_load
from lbm2.params.data_params import LBMDataParams
from lbm2.params.robotics.normalization_params import FieldNormalizationParams, NormalizationParams


class RoboticsNormalizer:
    """
    Normalizer for robotics data with configurable strategies.

    Supports:
    - Global normalization: normalize across all timesteps
    - Per-timestep normalization: normalize each timestep separately
    - Std-based normalization: use mean/std
    - Quantile-based normalization: use percentiles (e.g., 5th/95th)
    """

    def __init__(
        self,
        dataset_config: Union[Dict[str, Any], "LBMDataParams"],
        statistics_data: Optional[Dict[str, Any]] = None,
        statistics_path: Optional[str] = None,
    ):
        """
        Initialize normalizer.

        Args:
            dataset_config: LBMDataParams instance with field definitions and normalization settings
            statistics_data: Pre-loaded statistics dict
            statistics_path: Path to statistics JSON file
        """
        self.config = dataset_config

        self.enabled = self.config.normalization.enabled

        if not self.enabled:
            self.stats = None
            return

        # Load statistics
        if statistics_data is not None:
            self.stats = statistics_data
        elif statistics_path is not None:
            self.stats = self._load_statistics(statistics_path)
        else:
            logging.warning("No statistics provided - normalization will be disabled")
            self.enabled = False
            self.stats = None
            return

        if isinstance(self.stats, list):
            if len(self.stats) > 1:
                # TODO: Jean handle statistics merging
                raise ValueError("Merging statistics is not supported yet")
                # self.stats = merge_statistics(self.stats)
            self.stats = self.stats[0]

        # Parse configuration from dataclass
        self.method = self.config.normalization.method
        self.scope = self.config.normalization.scope
        self.field_configs = self.config.normalization.field_configs
        self.exclude_fields = set(self.config.exclude_fields)

        # Validate configuration
        if self.method not in ["std", "percentile_5_95", "percentile_1_99"]:
            raise ValueError(f"Invalid normalization method: {self.method}")
        if self.scope not in ["global", "per_timestep"]:
            raise ValueError(f"Invalid normalization scope: {self.scope}")

        logging.info(f"RoboticsNormalizer initialized: method={self.method}, scope={self.scope}")

    def _load_statistics(self, statistics_path: str) -> Dict[str, Any]:
        """Load statistics from JSON file."""
        if isinstance(statistics_path, str):
            stats = json_load(statistics_path)
        elif isinstance(statistics_path, list):
            stats = []
            for path in statistics_path:
                stats.append(json_load(path))
        else:
            raise ValueError(f"Invalid statistics path: {statistics_path}")
        logging.info(f"Loaded statistics from {statistics_path}")
        return stats

    def _get_field_config(self, field_name: str) -> FieldNormalizationParams:
        """Get configuration for a specific field."""
        # Check for exact match first
        if field_name in self.field_configs:
            return self.field_configs[field_name]

        # Check for pattern matches (mainly applies the same config to relative fields as the main field)
        for pattern, config in self.field_configs.items():
            if pattern in field_name:
                return config

        # Return default config
        return FieldNormalizationParams(method=self.method, scope=self.scope)

    def _should_normalize_field(self, field_name: str) -> bool:
        """Check if a field should be normalized."""
        if not self.enabled:
            return False

        # Skip excluded fields
        if field_name in self.exclude_fields:
            return False

        # Skip text and mask fields
        if any(keyword in field_name.lower() for keyword in ["text", "language", "instruction", "mask", "valid"]):
            return False

        # Skip if no statistics available
        if field_name not in self.stats:
            logging.warning(f"No statistics available for field: {field_name}")
            return False

        return True

    def _get_normalization_params(self, field_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get normalization parameters (center, scale) for a field.

        Args:
            field_name: Name of the field

        Returns:
            Tuple of (center, scale) tensors
        """
        field_stats = self.stats[field_name]
        field_config = self._get_field_config(field_name)

        method = field_config.method
        scope = field_config.scope
        epsilon = field_config.epsilon

        if not self.enabled:
            return torch.zeros(1), torch.ones(1)

        if scope == "global":
            # Use global statistics
            if method == "std":
                center = torch.tensor(field_stats["mean"], dtype=torch.float32)
                scale = torch.tensor(field_stats["std"], dtype=torch.float32)
            elif method == "percentile_5_95":
                # Use percentile normalization
                center = torch.tensor(field_stats["percentile_5"], dtype=torch.float32)
                scale = torch.tensor(field_stats["percentile_95"], dtype=torch.float32) - torch.tensor(
                    field_stats["percentile_5"], dtype=torch.float32
                )
            elif method == "percentile_1_99":
                center = torch.tensor(field_stats["percentile_1"], dtype=torch.float32)
                scale = torch.tensor(field_stats["percentile_99"], dtype=torch.float32) - torch.tensor(
                    field_stats["percentile_1"], dtype=torch.float32
                )
            else:
                raise ValueError(f"Invalid normalization method: {method}")
        else:
            # Use global statistics
            if method == "std":
                center = torch.tensor(field_stats["mean_per_timestep"], dtype=torch.float32)
                scale = torch.tensor(field_stats["std_per_timestep"], dtype=torch.float32)
            elif method == "percentile_5_95":
                # Use percentile normalization
                center = torch.tensor(field_stats["percentile_5_per_timestep"], dtype=torch.float32)
                scale = torch.tensor(
                    field_stats["percentile_95_per_timestep"],
                    dtype=torch.float32,
                ) - torch.tensor(
                    field_stats["percentile_5_per_timestep"],
                    dtype=torch.float32,
                )
            elif method == "percentile_1_99":
                center = torch.tensor(field_stats["percentile_1_per_timestep"], dtype=torch.float32)
                scale = torch.tensor(
                    field_stats["percentile_99_per_timestep"],
                    dtype=torch.float32,
                ) - torch.tensor(
                    field_stats["percentile_1_per_timestep"],
                    dtype=torch.float32,
                )
            else:
                raise ValueError(f"Invalid normalization method: {method}")

        # Avoid division by zero
        scale = torch.clamp(scale, min=epsilon)

        return center, scale

    def normalize_tensor(self, tensor: torch.Tensor, field_name: str) -> torch.Tensor:
        """
        Normalize a tensor.

        Args:
            tensor: Input tensor of shape [batch_size, timesteps, features] or [batch_size, features]
            field_name: Name of the field being normalized

        Returns:
            Normalized tensor
        """
        if not self._should_normalize_field(field_name):
            return tensor

        field_config = self._get_field_config(field_name)
        scope = field_config.scope

        center, scale = self._get_normalization_params(field_name)
        center = center.to(tensor.device)
        scale = scale.to(tensor.device)
        if scope == "global" or len(tensor.shape) == 2:
            # Global normalization or no time dimension
            # Broadcast to match tensor dimensions
            if len(tensor.shape) == 3:  # [B, T, D]
                center = center.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
                scale = scale.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
            elif len(tensor.shape) == 2:  # [B, D]
                center = center.unsqueeze(0)  # [1, D]
                scale = scale.unsqueeze(0)  # [1, D]

            normalized = (tensor - center) / scale

        elif scope == "per_timestep" and len(tensor.shape) == 3:
            # Per-timestep normalization
            center = center.unsqueeze(0)  # [1, D]
            scale = scale.unsqueeze(0)  # [1, D]
            normalized = (tensor - center) / scale
        else:
            # Unsupported tensor shape
            logging.warning(f"Unsupported tensor shape for normalization: {tensor.shape}")
            normalized = tensor

        return normalized

    def denormalize_tensor(self, normalized_tensor: torch.Tensor, field_name: str) -> torch.Tensor:
        """
        Denormalize a tensor (inverse of normalize_tensor).

        Args:
            normalized_tensor: Normalized tensor
            field_name: Name of the field being denormalized

        Returns:
            Denormalized tensor
        """
        if not self._should_normalize_field(field_name):
            return normalized_tensor

        field_config = self._get_field_config(field_name)
        scope = field_config.scope

        if scope == "global" or len(normalized_tensor.shape) == 2:
            # Global denormalization
            center, scale = self._get_normalization_params(field_name)

            # Broadcast to match tensor dimensions
            if len(normalized_tensor.shape) == 3:  # [B, T, D]
                center = center.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
                scale = scale.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
            elif len(normalized_tensor.shape) == 2:  # [B, D]
                center = center.unsqueeze(0)  # [1, D]
                scale = scale.unsqueeze(0)  # [1, D]

            denormalized = normalized_tensor * scale + center

        elif scope == "per_timestep" and len(normalized_tensor.shape) == 3:
            # Per-timestep denormalization
            batch_size, num_timesteps, feature_dim = normalized_tensor.shape
            denormalized = torch.zeros_like(normalized_tensor)

            for t in range(num_timesteps):
                center, scale = self._get_normalization_params(field_name, timestep=t)
                center = center.unsqueeze(0)  # [1, D]
                scale = scale.unsqueeze(0)  # [1, D]

                denormalized[:, t, :] = normalized_tensor[:, t, :] * scale + center
        else:
            # Unsupported tensor shape
            logging.warning(f"Unsupported tensor shape for denormalization: {normalized_tensor.shape}")
            denormalized = normalized_tensor

        return denormalized


@draccus.encode.register
def encode_field_norm_params(obj: FieldNormalizationParams) -> dict:
    """Custom encoder for FieldNormalizationParams."""
    return {"method": obj.method, "scope": obj.scope, "epsilon": obj.epsilon}


@draccus.encode.register
def encode_norm_params(obj: NormalizationParams) -> dict:
    """Custom encoder for NormalizationParams."""
    result = {
        "enabled": obj.enabled,
        "method": obj.method,
        "scope": obj.scope,
        "epsilon": obj.epsilon,
        "field_configs": {},
    }

    # Convert FieldNormalizationParams objects to dictionaries
    for field_name, field_config in obj.field_configs.items():
        result["field_configs"][field_name] = encode_field_norm_params(field_config)

    return result
