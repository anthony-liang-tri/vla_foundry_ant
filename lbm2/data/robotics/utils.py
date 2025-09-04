"""
Robotics data utilities.

This module provides helper functions for working with robotics data,
including extraction of proprioception and action data based on configuration.
"""

import numpy as np
import torch


def _get_rotation_matrix(x, y, z, perm_config):
    """Compute rotation matrix using specific axis permutation and signs."""
    # Construct rotation matrix based on axis permutation
    if perm_config == (0, 1, 2):  # XYZ
        rot_mat = np.column_stack([x, y, z])
    elif perm_config == (1, 0, 2):  # YXZ
        rot_mat = np.column_stack([y, x, z])
    elif perm_config == (0, 2, 1):  # XZY
        rot_mat = np.column_stack([x, z, y])
    elif perm_config == (1, 2, 0):  # YZX
        rot_mat = np.column_stack([y, z, x])
    else:
        return None
    return rot_mat


def rot_6d_to_matrix(rot_6d: np.ndarray, perm_config: tuple = (0, 1, 2), signs: tuple = (1, 1)) -> np.ndarray:
    """Convert 6D rotation to rotation matrix using comprehensive interpretation configuration."""
    # Handle single vector or batch of vectors
    original_shape = rot_6d.shape
    rot_6d = rot_6d.reshape(-1, 6)

    batch_size = rot_6d.shape[0]
    rot_matrices = []

    for i in range(batch_size):
        single_rot_6d = rot_6d[i]

        # Determine which vectors to use based on configuration
        vec1 = single_rot_6d[:3] * signs[0]
        vec2 = single_rot_6d[3:6] * signs[1]

        # Normalize first vector
        x = vec1 / np.linalg.norm(vec1)

        # Make second vector orthogonal and normalize
        y = vec2 - np.dot(x, vec2) * x
        y = y / np.linalg.norm(y)

        # Compute third vector
        z = np.cross(x, y)

        # Construct rotation matrix based on axis permutation
        rot_mat = _get_rotation_matrix(x, y, z, perm_config)

        det = np.linalg.det(rot_mat)
        if det < 0:
            rot_mat = _get_rotation_matrix(x, y, -z, perm_config)

        rot_matrices.append(rot_mat)

    if isinstance(rot_6d, torch.Tensor):
        rot_matrices = torch.tensor(rot_matrices, device=rot_6d.device, dtype=rot_6d.dtype)
    else:
        rot_matrices = np.array(rot_matrices)

    # Return original shape
    if len(original_shape) == 1:
        return rot_matrices[0]
    else:
        return rot_matrices.reshape(original_shape[:-1] + (3, 3))


def matrix_to_rot_6d(rotation_matrix):
    """Convert 3x3 rotation matrix to 6D rotation representation."""
    # Take first two columns and flatten
    return rotation_matrix[:, :2].flatten()


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
