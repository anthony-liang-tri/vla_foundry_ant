import numpy as np
import pytest
import yaml

from vla_foundry.data.robotics.utils import (
    any_to_actual_key,
    crop_sequence,
    load_action_field_config,
    matrix_to_rot_6d,
    normalize,
    rot_6d_from_relative,
    rot_6d_to_matrix,
    rot_6d_to_relative,
    rpy_to_R,
    xyz_from_relative,
    xyz_to_relative,
    xyzrpy_to_T,
)


def test_any_to_actual_key_converts_desired_to_actual():
    field = "robot__desired__end_effector__xyz"
    assert any_to_actual_key(field) == "robot__actual__end_effector__xyz"


def test_any_to_actual_key_returns_none_for_short_field():
    assert any_to_actual_key("robot") is None


def test_normalize_single_vector():
    vector = np.array([3.0, 4.0, 0.0])
    normalized = normalize(vector)
    expected = np.array([0.6, 0.8, 0.0])
    np.testing.assert_allclose(normalized, expected)


def test_normalize_batch_of_vectors():
    batch = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    normalized = normalize(batch)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(normalized, expected)


def test_rpy_to_R_identity():
    # Zero roll, pitch, yaw should yield identity matrix
    R = rpy_to_R(0.0, 0.0, 0.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-7)


def test_rpy_to_R_90deg_rotations():
    # 90 deg roll
    R_roll = rpy_to_R(np.pi / 2, 0, 0)
    expected_roll = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_allclose(R_roll, expected_roll, atol=1e-7)

    # 90 deg pitch
    R_pitch = rpy_to_R(0, np.pi / 2, 0)
    expected_pitch = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    np.testing.assert_allclose(R_pitch, expected_pitch, atol=1e-7)

    # 90 deg yaw
    R_yaw = rpy_to_R(0, 0, np.pi / 2)
    expected_yaw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(R_yaw, expected_yaw, atol=1e-7)


def test_rpy_to_R_is_valid_rotation_matrix():
    # A random set of angles
    r, p, y = 0.3, -0.7, 1.2
    R = rpy_to_R(r, p, y)
    # Orthonormal and right-handed (determinant +1)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-7)


def test_xyzrpy_to_T_single_pose():
    # [x, y, z, roll, pitch, yaw] all zeros
    pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    T = xyzrpy_to_T(pose)
    assert T.shape == (1, 4, 4)
    np.testing.assert_allclose(T[0, :3, :3], np.eye(3), atol=1e-7)
    np.testing.assert_allclose(T[0, :3, 3], np.array([1.0, 2.0, 3.0]), atol=1e-7)
    np.testing.assert_allclose(T[0, 3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7)


def test_xyzrpy_to_T_batch():
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # identity
            [1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2],  # yaw 90
        ]
    )
    T = xyzrpy_to_T(poses)
    assert T.shape == (2, 4, 4)

    # First transform is identity
    np.testing.assert_allclose(T[0], np.eye(4), atol=1e-7)

    # Second transform translation
    np.testing.assert_allclose(T[1, :3, 3], np.array([1.0, 2.0, 3.0]), atol=1e-7)

    # Second transform rotation equals Rz(90°)
    Rz_expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(T[1, :3, :3], Rz_expected, atol=1e-7)


def test_xyzrpy_to_T_invalid_length():
    # Should raise ValueError for wrong length
    with pytest.raises(ValueError):
        xyzrpy_to_T([1, 2, 3, 4, 5])


def test_xyzrpy_to_T_invalid_shape():
    # Should raise ValueError for wrong shape
    arr = np.ones((2, 5))
    with pytest.raises(ValueError):
        xyzrpy_to_T(arr)


def test_load_action_field_config_reads_yaml(tmp_path):
    config_path = tmp_path / "actions.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "action_key_fields": ["robot__desired__poses__right::panda__xyz"],
                "action_index_fields": ["robot__indices"],
            }
        )
    )

    config = load_action_field_config(str(config_path))
    assert config["action_key_fields"] == ["robot__desired__poses__right::panda__xyz"]
    assert config["action_index_fields"] == ["robot__indices"]


def test_rot_6d_to_matrix_identity():
    rot_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    matrix = rot_6d_to_matrix(rot_6d)
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-6)


def test_matrix_rot_6d_roundtrip():
    rotation_matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rot_6d = matrix_to_rot_6d(rotation_matrix)
    reconstructed = rot_6d_to_matrix(rot_6d)
    np.testing.assert_allclose(reconstructed, rotation_matrix, atol=1e-6)


def test_rot_6d_relative_roundtrip():
    identity = np.eye(3)
    rotation_matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    sequence = np.stack([matrix_to_rot_6d(identity), matrix_to_rot_6d(rotation_matrix)])
    relative = rot_6d_to_relative(sequence, sequence[0])

    relative_matrices = np.stack([rot_6d_to_matrix(vec) for vec in relative])
    expected_matrices = np.stack([identity, rotation_matrix])
    np.testing.assert_allclose(relative_matrices, expected_matrices, atol=1e-6)

    recovered = rot_6d_from_relative(relative, sequence[0])
    np.testing.assert_allclose(recovered, sequence, atol=1e-6)


def test_xyz_relative_roundtrip():
    xyz_sequence = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    reference = xyz_sequence[1]
    relative = xyz_to_relative(xyz_sequence, reference)
    np.testing.assert_allclose(relative, np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    restored = xyz_from_relative(relative, reference)
    np.testing.assert_allclose(restored, xyz_sequence)


def test_crop_sequence_extracts_expected_window():
    data = np.arange(10)
    cropped = crop_sequence(data, anchor_idx=5, past_timesteps=2, future_timesteps=3)
    np.testing.assert_array_equal(cropped, np.array([3, 4, 5, 6, 7, 8]))


def test_crop_sequence_invalid_anchor_asserts():
    with pytest.raises(AssertionError):
        crop_sequence(np.arange(5), anchor_idx=1, past_timesteps=2, future_timesteps=1)
