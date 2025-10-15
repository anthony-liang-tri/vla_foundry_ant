"""
Robotics data utilities.

This module provides helper functions for working with robotics data,
including extraction of proprioception and action data based on configuration.
"""

from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from lbm2.params.data_params import RoboticsDataParams


def normalize(x):
    """Normalize a vector or batch of vectors along the last dimension."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def load_action_field_config(config_path: str) -> Dict[str, List[Any]]:
    """Load action field configuration from YAML file."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return {
        "action_key_fields": data.get("action_key_fields", []),
        "action_index_fields": data.get("action_index_fields", []),
    }


def rot_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to rotation matrix using Gram-Schmidt orthogonalization.

    Takes a [N, 6] or [6,] np array and converts to [N, 3, 3] or [3, 3] rotation matrices.
    The input is assumed to be the first 2 rows of a rotation matrix.

    Based on Zhou et al. 2019: "On the Continuity of Rotation Representations in Neural Networks"
    """
    # Handle single vector or batch of vectors
    original_shape = rot_6d.shape
    if rot_6d.ndim == 1:
        rot_6d = rot_6d.reshape(1, 6)

    a1, a2 = rot_6d[..., :3], rot_6d[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    rot_matrices = np.stack((b1, b2, b3), axis=-2)

    # Return original shape
    if len(original_shape) == 1:
        return rot_matrices[0]
    else:
        return rot_matrices.reshape(original_shape[:-1] + (3, 3))


def matrix_to_rot_6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to 6D rotation representation.

    Takes a [N, 3, 3] or [3, 3] np array and converts to [N, 6] or [6,] array.
    Inverse of rot_6d_to_matrix().
    """
    batch_dim = rotation_matrix.shape[:-2]
    rot_6d = rotation_matrix[..., :2, :].copy().reshape(batch_dim + (6,))
    return rot_6d


def xyz_to_relative(xyz_sequence: np.ndarray, reference_index: int) -> np.ndarray:
    """
    Convert a sequence of xyz positions to relative positions with respect to a reference frame.

    Args:
        xyz_sequence: Array of shape (T, 3) where T is the number of timesteps
        reference_index: Index of the reference timestep to compute relative positions from

    Returns:
        Array of shape (T, 3) with relative xyz positions
    """
    reference_position = xyz_sequence[reference_index]
    relative_positions = xyz_sequence - reference_position
    return relative_positions


def rot_6d_to_relative(rot_6d_sequence: np.ndarray, reference_index: int) -> np.ndarray:
    """
    Convert a sequence of 6D rotations to relative rotations with respect to a reference frame.

    Args:
        rot_6d_sequence: Array of shape (T, 6) where T is the number of timesteps
        reference_index: Index of the reference timestep to compute relative rotations from

    Returns:
        Array of shape (T, 6) with relative 6D rotations
    """

    # Convert all rotations to matrices
    rotation_matrices = np.array([rot_6d_to_matrix(rot) for rot in rot_6d_sequence])

    # Get reference rotation matrix and compute its inverse (transpose for rotation matrices)
    reference_rotation = rotation_matrices[reference_index]
    reference_rotation_inv = reference_rotation.T

    # Compute relative rotations: R_relative = R_reference^-1 @ R_current
    relative_rotations = np.array(
        [reference_rotation_inv @ rotation_matrices[i] for i in range(len(rotation_matrices))]
    )

    # Convert back to 6D representation
    relative_rot_6d = np.array([matrix_to_rot_6d(rot) for rot in relative_rotations])

    return relative_rot_6d


def xyz_from_relative(relative_xyz_sequence: np.ndarray, reference_position: np.ndarray) -> np.ndarray:
    """
    Convert a sequence of relative xyz positions back to absolute positions.

    Args:
        relative_xyz_sequence: Array of shape (T, 3) or (B, T, 3) with relative xyz positions
        reference_position: Reference position of shape (3,) to add back

    Returns:
        Array of shape (T, 3) or (B, T, 3) with absolute xyz positions
    """
    return relative_xyz_sequence + reference_position


def rot_6d_from_relative(relative_rot_6d_sequence: np.ndarray, reference_rot_6d: np.ndarray) -> np.ndarray:
    """
    Convert a sequence of relative 6D rotations back to absolute rotations.

    Args:
        relative_rot_6d_sequence: Array of shape (T, 6) or (B, T, 6) with relative 6D rotations
        reference_rot_6d: Reference 6D rotation of shape (6,), (B, 6) or (B, T, 6) to compose with

    Returns:
        Array of shape (T, 6) or (B, T, 6) with absolute 6D rotations
    """
    relative = np.asarray(relative_rot_6d_sequence)
    if relative.shape[-1] != 6:
        raise ValueError("relative_rot_6d_sequence must have last dimension equal to 6")

    original_shape = relative.shape
    relative_flat = relative.reshape(-1, 6)

    reference = np.asarray(reference_rot_6d)
    if reference.shape[-1] != 6:
        raise ValueError("reference_rot_6d must have last dimension equal to 6")

    if reference.ndim == 1:
        reference_flat = np.broadcast_to(reference, (relative_flat.shape[0], 6))
    else:
        target_shape = original_shape[:-1] + (6,)
        expand_dims = len(target_shape) - reference.ndim
        if expand_dims < 0:
            raise ValueError("reference_rot_6d has more dimensions than relative_rot_6d_sequence")
        ref = reference
        for _ in range(expand_dims):
            axis = ref.ndim - 1 if ref.ndim > 1 else 0
            ref = np.expand_dims(ref, axis=axis)
        reference_broadcast = np.broadcast_to(ref, target_shape)
        reference_flat = reference_broadcast.reshape(-1, 6)

    reference_matrices = rot_6d_to_matrix(reference_flat)
    relative_matrices = rot_6d_to_matrix(relative_flat)

    absolute_matrices = np.matmul(reference_matrices, relative_matrices)

    absolute_rot_6d_flat = np.stack([matrix_to_rot_6d(rot) for rot in absolute_matrices], axis=0)

    return absolute_rot_6d_flat.reshape(original_shape)


def extract_proprioception_data(batch, dataset_config: RoboticsDataParams, device=None):
    """
    Extract proprioception data from a batch based on dataset config.

    Args:
        batch: Batch data containing 'lowdim' field
        dataset_config: Dataset configuration with proprioception field definitions
        device: Device to move tensors to (optional)

    Returns:
        Proprioception tensor of shape [batch_size, timesteps, proprioception_dim] or [batch_size, proprioception_dim]
    """
    if "lowdim" not in batch:
        raise ValueError("Batch missing 'lowdim' field")

    proprioception_data = []

    # Extract data for each proprioception field
    for field_name in dataset_config.proprioception_fields:
        if field_name in batch["lowdim"]:
            field_data = batch["lowdim"][field_name]
            if device is not None:
                field_data = field_data.to(device)
            proprioception_data.append(field_data)

    if not proprioception_data:
        raise ValueError("No proprioception fields found in batch")

    # Concatenate all proprioception data along the last dimension
    proprioception = torch.cat(proprioception_data, dim=-1)
    return proprioception


def extract_action_data(batch, dataset_config: RoboticsDataParams, device=None):
    """
    Extract action data from a batch based on dataset config.

    Args:
        batch: Batch data containing 'lowdim' field
        dataset_config: Dataset configuration with action field definitions
        device: Device to move tensors to (optional)

    Returns:
        Action tensor of shape [batch_size, timesteps, action_dim] or [batch_size, action_dim]
    """
    if "lowdim" not in batch:
        raise ValueError("Batch missing 'lowdim' field")

    action_data = []

    # Extract data for each action field
    for field_name in dataset_config.action_fields:
        if field_name in batch["lowdim"]:
            field_data = batch["lowdim"][field_name]
            if device is not None:
                field_data = field_data.to(device)
            action_data.append(field_data)

    if not action_data:
        raise ValueError("No action fields found in batch")

    # Concatenate all action data along the last dimension
    actions = torch.cat(action_data, dim=-1)
    return actions


def extract_and_aggregate_proprioception(
    batch: Dict[str, Any], dataset_config: RoboticsDataParams, device=None, aggregation_method: str = "mean"
) -> torch.Tensor:
    """
    Extract and aggregate proprioception data from batch.

    Args:
        batch: Batch data from robotics dataloader
        dataset_config: Dataset configuration (uses default if None)
        device: Device to move tensors to
        aggregation_method: How to aggregate temporal dimension ("mean", "last", "masked_mean")

    Returns:
        Aggregated proprioception tensor [batch_size, proprioception_dim]
    """
    # Extract proprioception data [B, T, D]
    proprioception = extract_proprioception_data(batch, dataset_config, device)

    # Aggregate temporal dimension
    if aggregation_method == "mean":
        # Simple mean across time
        proprioception = proprioception.mean(dim=1)  # [B, D]
    elif aggregation_method == "last":
        # Take last timestep
        proprioception = proprioception[:, -1, :]  # [B, D]
    elif aggregation_method == "masked_mean":
        # Use past mask if available for masked mean
        if "past_mask" in batch:
            past_mask = batch["past_mask"]
            if device is not None:
                past_mask = past_mask.to(device)

            # Masked mean of past proprioception
            proprioception_masked = proprioception * past_mask.unsqueeze(-1).float()
            proprioception = proprioception_masked.sum(dim=1) / past_mask.sum(dim=1, keepdim=True).float()  # [B, D]
        else:
            # Fallback to regular mean if no mask available
            proprioception = proprioception.mean(dim=1)  # [B, D]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    return proprioception


def extract_and_aggregate_actions(
    batch: Dict[str, Any], dataset_config: RoboticsDataParams, device=None, aggregation_method: str = "none"
) -> torch.Tensor:
    """
    Extract and optionally aggregate action data from batch.

    Args:
        batch: Batch data from robotics dataloader
        dataset_config: Dataset configuration (uses default if None)
        device: Device to move tensors to
        aggregation_method: How to aggregate temporal dimension ("none", "mean", "future_only")

    Returns:
        Action tensor [batch_size, timesteps, action_dim] or [batch_size, action_dim]
    """

    # Extract action data [B, T, D]
    actions = extract_action_data(batch, dataset_config, device)

    # Optionally aggregate temporal dimension
    if aggregation_method == "none":
        # Keep full temporal dimension
        return actions  # [B, T, D]
    elif aggregation_method == "mean":
        # Mean across time
        return actions.mean(dim=1)  # [B, D]
    elif aggregation_method == "future_only":
        # Use future mask if available
        if "future_mask" in batch:
            future_mask = batch["future_mask"]
            if device is not None:
                future_mask = future_mask.to(device)

            # Masked mean of future actions
            actions_masked = actions * future_mask.unsqueeze(-1).float()
            actions = actions_masked.sum(dim=1) / future_mask.sum(dim=1, keepdim=True).float()  # [B, D]
            return actions
        else:
            # Fallback to regular mean if no mask available
            return actions.mean(dim=1)  # [B, D]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")


def extract_robotics_data_for_training(
    batch: Dict[str, Any], dataset_config: RoboticsDataParams, device=None
) -> Dict[str, torch.Tensor]:
    """
    Extract proprioception and action data for training.

    This is a convenience function that replaces hardcoded field extraction
    in training scripts.

    Args:
        batch: Batch data from robotics dataloader
        dataset_config: Dataset configuration (uses default if None)
        device: Device to move tensors to

    Returns:
        Dict with 'proprioception' and 'actions' tensors
    """

    # Extract proprioception using masked mean (considers past context)
    proprioception = extract_and_aggregate_proprioception(
        batch, dataset_config, device, aggregation_method="masked_mean"
    )

    # Extract actions keeping full temporal dimension
    actions = extract_and_aggregate_actions(batch, dataset_config, device, aggregation_method="none")

    return {"proprioception": proprioception, "actions": actions}


# Example usage:
if __name__ == "__main__":
    # Example with xyz positions
    xyz_positions = np.array([[1.0, 2.0, 3.0], [1.5, 2.2, 3.1], [2.0, 2.5, 3.3], [2.2, 2.8, 3.5]])

    reference_idx = 1  # Use second position as reference
    relative_xyz = xyz_to_relative(xyz_positions, reference_idx)
    print("Original positions:")
    print(xyz_positions)
    print(f"\nRelative to index {reference_idx}:")
    print(relative_xyz)

    # Example with 6D rotations (random valid 6D rotations)
    np.random.seed(42)
    # Generate some 6D rotations by creating rotation matrices and taking first 2 columns
    rot_6d_positions = []
    for _ in range(4):
        # Create a random rotation matrix using QR decomposition
        A = np.random.randn(3, 3)
        Q, R = np.linalg.qr(A)
        # Ensure proper rotation (det = 1)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        # Convert to 6D using the conversion function
        rot_6d = matrix_to_rot_6d(Q)
        rot_6d_positions.append(rot_6d)

    rot_6d_positions = np.array(rot_6d_positions)

    relative_rot_6d = rot_6d_to_relative(rot_6d_positions, reference_idx)
    print("\nOriginal 6D rotations:")
    print(rot_6d_positions)
    print(f"\nRelative 6D rotations to index {reference_idx}:")
    print(relative_rot_6d)
