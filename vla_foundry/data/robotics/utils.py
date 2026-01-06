"""
Robotics data utilities.

This module provides helper functions for working with robotics data,
including extraction of proprioception and action data based on configuration.
"""

from typing import Any, Dict, List, Union

import numpy as np
import yaml
from tdigest_rs import TDigest


def any_to_actual_key(field: str) -> str:
    """Convert any field name to its 'actual' counterpart for field mapping lookup.
    Expects field name to be in format: robot__<desired/actual/action>__...
      - __ are used as separators for the different parts of the field name
      - <desired/actual/action> is the type of the data
      - ... is the rest of the field name separated by __
    """
    parts = field.split("__")
    if len(parts) > 2:
        return "__".join(parts[0:1] + ["actual"] + parts[2:])
    else:
        return None


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


def get_xyz(pose) -> np.ndarray:
    return np.array(pose.translation(), dtype=np.float64)


def get_rot_6d(pose) -> np.ndarray:
    return np.array(pose.rotation().matrix()[:2, :].flatten(), dtype=np.float64)


def matrix_to_rot_6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to 6D rotation representation.

    Takes a [N, 3, 3] or [3, 3] np array and converts to [N, 6] or [6,] array.
    Inverse of rot_6d_to_matrix().
    """
    batch_dim = rotation_matrix.shape[:-2]
    rot_6d = rotation_matrix[..., :2, :].copy().reshape(batch_dim + (6,))
    return rot_6d


def xyz_to_relative(xyz_sequence: np.ndarray, reference_xyz: np.ndarray) -> np.ndarray:
    """
    Convert a sequence of xyz positions to relative positions with respect to a reference frame.

    Args:
        xyz_sequence: Array of shape (T, 3) where T is the number of timesteps
        reference_index: Index of the reference timestep to compute relative positions from

    Returns:
        Array of shape (T, 3) with relative xyz positions
    """
    reference_position = reference_xyz
    relative_positions = xyz_sequence - reference_position
    return relative_positions


def rot_6d_to_relative(rot_6d_sequence: np.ndarray, reference_6d: np.ndarray) -> np.ndarray:
    """
    Convert a sequence of 6D rotations to relative rotations with respect to a reference frame.

    Args:
        rot_6d_sequence: Array of shape (T, 6) where T is the number of timesteps
        reference_data: Reference data of shape (3,) or (T, 3)

    Returns:
        Array of shape (T, 6) with relative 6D rotations
    """

    # Convert all rotations to matrices
    rotation_matrices = np.array([rot_6d_to_matrix(rot) for rot in rot_6d_sequence])

    # Get reference rotation matrix and compute its inverse (transpose for rotation matrices)
    reference_rotation = rot_6d_to_matrix(reference_6d)
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


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert roll, pitch, yaw angles into a 3×3 rotation matrix.

    Args:
        roll: Rotation about the x-axis in radians.
        pitch: Rotation about the y-axis in radians.
        yaw: Rotation about the z-axis in radians.

    Returns:
        A (3, 3) array representing the rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ (Ry @ Rx)


def xyzrpy_to_T(pose: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Convert [x, y, z, roll, pitch, yaw] vector(s) into homogeneous transform(s).

    Args:
        pose: Either a length-6 vector [x, y, z, roll, pitch, yaw] (angles in radians),
            or an array of shape (N, 6) where each row is one pose.

    Returns:
        An array of shape (N, 4, 4) with homogeneous transforms. If a single pose is provided,
        N = 1.
    """
    arr = np.asarray(pose, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 6:
            raise ValueError("Pose must be length 6 (x, y, z, roll, pitch, yaw)")
        arr = arr[None, :]
    elif arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError("Pose must have shape (n, 6)")

    Ts: List[np.ndarray] = []
    for x, y, z, r, p, y_ in arr:
        T = np.eye(4)
        T[:3, :3] = rpy_to_R(r, p, y_)
        T[:3, 3] = [x, y, z]
        Ts.append(T)
    return np.stack(Ts, axis=0)


# Example usage:
if __name__ == "__main__":
    # Example with xyz positions
    xyz_positions = np.array([[1.0, 2.0, 3.0], [1.5, 2.2, 3.1], [2.0, 2.5, 3.3], [2.2, 2.8, 3.5]])

    reference_idx = 1  # Use second position as reference
    relative_xyz = xyz_to_relative(xyz_positions, xyz_positions[reference_idx])
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

    relative_rot_6d = rot_6d_to_relative(rot_6d_positions, rot_6d_positions[reference_idx])
    print("\nOriginal 6D rotations:")
    print(rot_6d_positions)
    print(f"\nRelative 6D rotations to index {reference_idx}:")
    print(relative_rot_6d)


def crop_sequence(
    data,
    anchor_idx,
    past_timesteps,
    future_timesteps,
):
    """
    Crop a sequence to specified past and future timesteps around an anchor point.

    Args:
        data: Array of shape [T, ...] where T is the total number of timesteps
        anchor_idx: Index of the anchor timestep in the original sequence
        past_timesteps: Number of past timesteps to keep (not including anchor)
        future_timesteps: Number of future timesteps to keep (including anchor)

    Returns:
        Cropped array of shape [past_timesteps + 1 +future_timesteps, ...]
    """
    # Calculate the range to extract
    assert anchor_idx >= past_timesteps
    assert data.shape[0] >= anchor_idx + future_timesteps + 1
    start_idx = anchor_idx - past_timesteps
    end_idx = anchor_idx + future_timesteps + 1

    return data[start_idx:end_idx]


def merge_percentiles_from_tdigest(states_list: List[Dict[str, Any]], target_p: float) -> np.ndarray:
    """
    Merge percentiles by merging t-digest states and querying the merged digest.

    T-digest supports native merging of centroids, which provides accurate
    percentile estimates especially for tail quantiles.
    """
    valid_states = [s for s in states_list if s is not None and "digests" in s]
    if not valid_states:
        return None

    # Get shape from first valid state
    example_shape = tuple(valid_states[0].get("shape", []))
    if not example_shape:
        return None

    result = np.zeros(example_shape)

    # For each index position, merge all t-digests and query the percentile
    for idx in np.ndindex(example_shape):
        idx_str = str(idx)
        idx_list = list(idx)
        digests_to_merge = []
        buffer_samples = []

        for state in valid_states:
            # Handle compact serialization format
            if "indices" in state["digests"]:
                indices = state["digests"]["indices"]
                if idx_list in indices:
                    pos = indices.index(idx_list)
                    means = np.array(state["digests"]["means"][pos], dtype=np.float32)
                    weights = np.array(state["digests"]["weights"][pos], dtype=np.uint32)
                    compression = state.get("compression", 100)
                    digests_to_merge.append(TDigest.from_means_weights(means, weights, compression))
                elif "buffers" in state and isinstance(state["buffers"], dict) and "indices" in state["buffers"]:
                    # New sparse buffer format
                    b_indices = state["buffers"]["indices"]
                    if idx_list in b_indices:
                        pos = b_indices.index(idx_list)
                        buffer_samples.extend(state["buffers"]["data"][pos])
            else:
                # Legacy format handling (stringified tuples)
                if state.get("digests") and idx_str in state["digests"]:
                    digest_data = state["digests"][idx_str]
                    means = np.array(digest_data["means"], dtype=np.float32)
                    weights = np.array(digest_data["weights"], dtype=np.uint32)
                    compression = state.get("compression", 100)
                    digests_to_merge.append(TDigest.from_means_weights(means, weights, compression))
                elif state.get("buffer") is not None:
                    # Old monolithic buffer format
                    buffer = np.array(state["buffer"], dtype=np.float32)
                    counts = np.array(state["counts"], dtype=int)
                    cnt = counts[idx]
                    if cnt > 0:
                        selector = (slice(0, cnt),) + idx
                        buffer_samples.extend(buffer[selector].tolist())
                elif state.get("buffers") and idx_str in state["buffers"]:
                    # Intermediate sparse buffer format (Dict[str, List])
                    buffer_samples.extend(state["buffers"][idx_str])

        # Create t-digest from buffer samples if any
        if buffer_samples:
            compression = valid_states[0].get("compression", 100)
            buffer_digest = TDigest.from_array(np.array(buffer_samples, dtype=np.float32), compression)
            digests_to_merge.append(buffer_digest)

        if digests_to_merge:
            # Merge all digests
            merged = digests_to_merge[0]
            for d in digests_to_merge[1:]:
                merged = merged.merge(d)
            result[idx] = merged.quantile(target_p)

    return result


def merge_statistics_single_field(tensor_stats: Dict[str, List[Any]], stat_name: str) -> np.ndarray:
    """
    tensor_stats: {mean: [m1, m2, ... mn], std: [s1, s2, ... sn], ...}
    stat_name: mean, std, min, max, etc.
    """
    # Tensor sizes:
    # mean, min, max, etc. [num_datasets, action_dim]
    # mean_per_timestep, etc. [num_datasets, T, action_dim]
    # count [num_datasets, T]
    if stat_name == "mean":
        mean_per_timestep = merge_statistics_single_field(tensor_stats, "mean_per_timestep")
        counts = np.sum(tensor_stats["count"], axis=0)
        return np.average(mean_per_timestep, axis=0, weights=counts)
    elif stat_name == "mean_per_timestep":
        counts = np.broadcast_to(tensor_stats["count"][..., None], tensor_stats["mean_per_timestep"].shape)
        return np.average(tensor_stats["mean_per_timestep"], axis=0, weights=counts)
    elif stat_name == "std":
        # Use law of total variance: σ²_overall = E[σ²_t] + Var[μ_t]
        std_per_timestep = merge_statistics_single_field(tensor_stats, "std_per_timestep")
        variance_per_timestep = std_per_timestep**2
        mean_per_timestep = merge_statistics_single_field(tensor_stats, "mean_per_timestep")
        # Use weighted mean and variance based on counts per timestep
        counts_per_timestep = np.sum(tensor_stats["count"], axis=0)
        mean_variance = np.average(variance_per_timestep, axis=0, weights=counts_per_timestep)
        weighted_mean = np.average(mean_per_timestep, axis=0, weights=counts_per_timestep)
        variance_of_means = np.average((mean_per_timestep - weighted_mean) ** 2, axis=0, weights=counts_per_timestep)
        overall_variance = mean_variance + variance_of_means
        return np.sqrt(np.maximum(overall_variance, 0.0))

    elif stat_name == "std_per_timestep":
        # Use pooled variance formula to merge per-timestep standard deviations
        # σ²_pooled = [Σ((nᵢ-1)×σᵢ² + nᵢ×(μᵢ - μ_global)²)] / (n_total - 1)
        counts = np.array(tensor_stats["count"])[..., np.newaxis]  # [num_datasets, T, 1]
        total_counts = np.sum(counts, axis=0)  # [T, 1]
        variances = np.array(tensor_stats[stat_name]) ** 2

        pooled_mean_per_timestep = merge_statistics_single_field(tensor_stats, "mean_per_timestep")
        mean_diffs_squared = (tensor_stats["mean_per_timestep"] - pooled_mean_per_timestep[np.newaxis, :, :]) ** 2
        pooled_variance = np.sum((counts - 1) * variances + counts * mean_diffs_squared, axis=0) / np.maximum(
            total_counts - 1, 1
        )
        return np.sqrt(pooled_variance)

    elif stat_name in ["min", "min_per_timestep"]:
        return np.min(tensor_stats[stat_name], axis=0)
    elif stat_name in ["max", "max_per_timestep"]:
        return np.max(tensor_stats[stat_name], axis=0)
    elif stat_name in ["count"]:
        return np.sum(tensor_stats["count"], axis=0)
    elif stat_name in [
        "percentile_1",
        "percentile_2",
        "percentile_5",
        "percentile_95",
        "percentile_98",
        "percentile_99",
        "percentile_1_per_timestep",
        "percentile_2_per_timestep",
        "percentile_5_per_timestep",
        "percentile_95_per_timestep",
        "percentile_98_per_timestep",
        "percentile_99_per_timestep",
    ]:
        p_val_str_raw = stat_name.split("_")[1]
        p_val = float(p_val_str_raw) / 100.0
        is_per_timestep = "per_timestep" in stat_name

        state_key = "tdigest_state_per_timestep" if is_per_timestep else "tdigest_state"

        states = tensor_stats[state_key]
        return merge_percentiles_from_tdigest(states, p_val)

    elif stat_name in ["percentile_sample_count", "tdigest_state", "tdigest_state_per_timestep"]:
        return None  # We don't merge these directly; tdigest states are used for percentiles.
    else:
        raise ValueError(f"Invalid stat name: {stat_name}")


def merge_statistics(statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    `statistics` is a list of dictionaries. Each item on the list represents a different dataset.
    Keys are tensor names. Values are dictionaries with keys mean, std, min, max, etc.

    Merging is done as follows:
    - mean - We can calculate this exactly
    - std - We can calculate this exactly (pooled variance formula)
    - min, max - We can calculate this exactly
    - percentiles - We take a weighted average of the percentiles weighted by the counts
    - count - We sum the counts
    """
    tensor_names, stat_names = set(), set()
    for dataset_statistics in statistics:
        tensor_names.update(dataset_statistics.keys())
        for tensor_name in dataset_statistics:
            stat_names.update(dataset_statistics[tensor_name].keys())

    # Create batched = {
    # robot:left:xyz: {mean: np.array([m1, m2, ... mn]), std: np.array([s1, s2, ... sn]), ...},
    # robot:right:xyz: {mean: np.array([m1, m2, ... mn]), std: np.array([s1, s2, ... sn]), ...},
    # ...
    # }
    batched_stats = {tensor_name: {s: [] for s in stat_names} for tensor_name in tensor_names}
    for tensor_name in tensor_names:
        for stat_name in stat_names:
            for dataset_statistics in statistics:
                val = dataset_statistics[tensor_name].get(stat_name)
                batched_stats[tensor_name][stat_name].append(val)

            if stat_name in ["psquared_state", "psquared_state_per_timestep"]:
                batched_stats[tensor_name][stat_name] = np.array(batched_stats[tensor_name][stat_name], dtype=object)
            else:
                batched_stats[tensor_name][stat_name] = np.array(batched_stats[tensor_name][stat_name])

    merged_stats = {}
    for tensor_name in batched_stats:
        merged_stats[tensor_name] = {}
        for stat_name in batched_stats[tensor_name]:
            merged_stats[tensor_name][stat_name] = merge_statistics_single_field(batched_stats[tensor_name], stat_name)
            if merged_stats[tensor_name][stat_name] is not None:
                merged_stats[tensor_name][stat_name] = merged_stats[tensor_name][stat_name].tolist()

    return merged_stats
