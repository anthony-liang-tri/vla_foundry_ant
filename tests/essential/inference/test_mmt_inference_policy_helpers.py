import numpy as np

from vla_foundry.inference.robotics.mmt.action_handlers import (
    MmtActionMapper,
    load_lowdim_index_selection,
    restore_full_values,
)
from vla_foundry.inference.robotics.mmt.field_layouts import MMT_FIELD_LAYOUTS
from vla_foundry.inference.robotics.mmt.mmt_inference_policy import ZzkPolicyInference


def test_apply_index_selection_subsets_correctly():
    full_values = np.arange(7, dtype=np.float32)
    selected = ZzkPolicyInference._apply_index_selection(full_values, [0, 1, 2])
    np.testing.assert_array_equal(selected, np.array([0, 1, 2], dtype=np.float32))


def test_restore_full_values_scatters_into_full_dim():
    layout = MMT_FIELD_LAYOUTS["right_arm_action"]
    # Select only arm (indices 1-6), omitting gripper
    arm_only = np.arange(6, dtype=np.float32) + 10.0
    reconstructed = restore_full_values(arm_only, list(range(1, 7)), layout["full_dim"])

    expected = np.zeros(layout["full_dim"], dtype=np.float32)
    expected[1:7] = arm_only
    np.testing.assert_array_equal(reconstructed, expected)


def test_build_full_eef_pose_concatenates_left_and_right_state():
    state = np.zeros((11, 6), dtype=np.float32)
    state[1, 2] = 0.1
    state[2, 0:6] = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    state[3, 2] = 0.2
    state[4, 0:6] = np.array([7, 8, 9, 10, 11, 12], dtype=np.float32)

    full_pose = ZzkPolicyInference._build_full_eef_pose(
        state,
        left_gripper_row=1,
        left_pose_row=2,
        right_gripper_row=3,
        right_pose_row=4,
    )

    np.testing.assert_array_equal(
        full_pose,
        np.array([0.1, 1, 2, 3, 4, 5, 6, 0.2, 7, 8, 9, 10, 11, 12], dtype=np.float32),
    )


def test_action_mapper_generates_arm_commands():
    mapper = MmtActionMapper(load_lowdim_index_selection({}))
    zzk_action = {}
    debug_parts = []

    right_arm = np.array([0.5, 1, 2, 3, 4, 5, 6], dtype=np.float32)
    mapper.append_right_arm_action_command(right_arm, zzk_action, debug_parts)

    np.testing.assert_array_almost_equal(zzk_action["right_arm"], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_almost_equal(zzk_action["right_gripper"], [0, 0, 0.5, 0, 0, 0])

    left_arm = np.array([0.3, 7, 8, 9, 10, 11, 12], dtype=np.float32)
    mapper.append_left_arm_action_command(left_arm, zzk_action, debug_parts)

    np.testing.assert_array_almost_equal(zzk_action["left_arm"], [7, 8, 9, 10, 11, 12])
    np.testing.assert_array_almost_equal(zzk_action["left_gripper"], [0, 0, 0.3, 0, 0, 0])


def test_action_mapper_generates_gripper_tip_arm_commands():
    mapper = MmtActionMapper(load_lowdim_index_selection({}))
    zzk_action = {}
    debug_parts = []

    right_arm = np.array([0.5, 1, 2, 3, 4, 5, 6], dtype=np.float32)
    mapper.append_right_arm_action_at_gripper_tip_command(right_arm, zzk_action, debug_parts)

    np.testing.assert_array_almost_equal(zzk_action["right_arm_at_gripper_tip"], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_almost_equal(zzk_action["right_gripper"], [0, 0, 0.5, 0, 0, 0])

    left_arm = np.array([0.3, 7, 8, 9, 10, 11, 12], dtype=np.float32)
    mapper.append_left_arm_action_at_gripper_tip_command(left_arm, zzk_action, debug_parts)

    np.testing.assert_array_almost_equal(zzk_action["left_arm_at_gripper_tip"], [7, 8, 9, 10, 11, 12])
    np.testing.assert_array_almost_equal(zzk_action["left_gripper"], [0, 0, 0.3, 0, 0, 0])


def test_action_mapper_generates_base_head_and_lift_commands():
    preprocessing_config = {
        "mmt_lowdim_flatten_indices_selection": {
            "base_action": [0, 1, 5],
            "head_action": [4, 5],
            "lift_action": [2],
        }
    }
    mapper = MmtActionMapper(load_lowdim_index_selection(preprocessing_config))
    zzk_action = {}
    debug_parts = []

    mapper.append_base_action_command(np.array([0.1, -0.2, 0.3], dtype=np.float32), zzk_action, debug_parts)
    mapper.append_head_action_command(np.array([0.4, -0.5], dtype=np.float32), zzk_action, debug_parts)
    mapper.append_lift_action_command(np.array([0.6], dtype=np.float32), zzk_action, debug_parts)

    np.testing.assert_array_equal(zzk_action["chassis"], np.array([0.1, -0.2, 0.0, 0.0, 0.0, 0.3], dtype=np.float32))
    np.testing.assert_array_equal(zzk_action["head"], np.array([0.0, 0.0, 0.0, 0.0, 0.4, -0.5], dtype=np.float32))
    np.testing.assert_array_equal(zzk_action["lift"], np.array([0.0, 0.0, 0.6, 0.0, 0.0, 0.0], dtype=np.float32))


def test_apply_index_selection_for_right_wrench():
    full_wrench = np.arange(12, dtype=np.float32)
    selected = ZzkPolicyInference._apply_index_selection(full_wrench, list(range(6, 12)))
    np.testing.assert_array_equal(selected, np.arange(6, 12, dtype=np.float32))
