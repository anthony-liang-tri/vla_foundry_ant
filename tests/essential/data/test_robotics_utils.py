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
    xyz_from_relative,
    xyz_to_relative,
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
