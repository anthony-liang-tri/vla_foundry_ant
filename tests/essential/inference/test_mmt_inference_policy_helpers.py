import numpy as np

from vla_foundry.inference.robotics.mmt.action_handlers import (
    MmtActionMapper,
    load_lowdim_field_remap,
    load_lowdim_index_selection,
    restore_full_values,
)
from vla_foundry.inference.robotics.mmt.field_layouts import (
    MMT_FIELD_LAYOUTS,
    build_runtime_layouts,
)
from vla_foundry.inference.robotics.mmt.mmt_inference_policy import ZzkPolicyInference


def test_apply_index_selection_subsets_correctly():
    full_values = np.arange(7, dtype=np.float32)
    selected = ZzkPolicyInference._apply_index_selection(full_values, [0, 1, 2])
    np.testing.assert_array_equal(selected, np.array([0, 1, 2], dtype=np.float32))


def test_restore_full_values_scatters_into_full_dim():
    # right_arm_action has full_dim=7 (gripper + 6 arm joints)
    full_dim = 7
    # Select only arm (indices 1-6), omitting gripper
    arm_only = np.arange(6, dtype=np.float32) + 10.0
    reconstructed = restore_full_values(arm_only, list(range(1, 7)), full_dim)

    expected = np.zeros(full_dim, dtype=np.float32)
    expected[1:7] = arm_only
    np.testing.assert_array_equal(reconstructed, expected)


def test_build_full_eef_pose_concatenates_left_and_right_state():
    state = np.zeros((11, 6), dtype=np.float32)
    state[2, 0:6] = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    state[4, 0:6] = np.array([7, 8, 9, 10, 11, 12], dtype=np.float32)
    status = {
        "left_gripper_position": 0.1,
        "right_gripper_position": 0.2,
    }

    full_pose = ZzkPolicyInference._build_full_eef_pose(
        status,
        state,
        left_pose_row=2,
        right_pose_row=4,
    )

    np.testing.assert_array_equal(
        full_pose,
        np.array([0.1, 1, 2, 3, 4, 5, 6, 0.2, 7, 8, 9, 10, 11, 12], dtype=np.float32),
    )


def test_build_full_eef_pose_returns_none_without_scalar_gripper_fields():
    state = np.zeros((11, 6), dtype=np.float32)
    status = {"left_gripper_position": 0.1}

    assert (
        ZzkPolicyInference._build_full_eef_pose(
            status,
            state,
            left_pose_row=2,
            right_pose_row=4,
        )
        is None
    )


def _make_arm_remap_config():
    """Preprocessing config that remaps arm_action and arm_action_at_gripper_tip."""
    return {
        "lowdim_field_remap": {
            "arm_action": [
                {"to": "left_arm_action", "indices": [0, 1, 2, 3, 4, 5, 6]},
                {"to": "right_arm_action", "indices": [7, 8, 9, 10, 11, 12, 13]},
            ],
            "arm_action_at_gripper_tip": [
                {"to": "left_arm_action_at_gripper_tip", "indices": [0, 1, 2, 3, 4, 5, 6]},
                {"to": "right_arm_action_at_gripper_tip", "indices": [7, 8, 9, 10, 11, 12, 13]},
            ],
        }
    }


def test_action_mapper_generates_arm_commands():
    config = _make_arm_remap_config()
    layouts = build_runtime_layouts(MMT_FIELD_LAYOUTS, config)
    mapper = MmtActionMapper(load_lowdim_index_selection({}), layouts)
    zzk_action = {}
    debug_parts = []

    right_arm = np.array([0.5, 1, 2, 3, 4, 5, 6], dtype=np.float32)
    mapper.append_right_arm_action_command(right_arm, zzk_action, debug_parts)

    np.testing.assert_array_almost_equal(zzk_action["right_arm"], [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(zzk_action["right_gripper"], 0.5)

    left_arm = np.array([0.3, 7, 8, 9, 10, 11, 12], dtype=np.float32)
    mapper.append_left_arm_action_command(left_arm, zzk_action, debug_parts)

    np.testing.assert_array_almost_equal(zzk_action["left_arm"], [7, 8, 9, 10, 11, 12])
    np.testing.assert_allclose(zzk_action["left_gripper"], 0.3)


def test_action_mapper_generates_gripper_tip_arm_commands():
    config = _make_arm_remap_config()
    layouts = build_runtime_layouts(MMT_FIELD_LAYOUTS, config)
    mapper = MmtActionMapper(load_lowdim_index_selection({}), layouts)
    zzk_action = {}
    debug_parts = []

    right_arm = np.array([0.5, 1, 2, 3, 4, 5, 6], dtype=np.float32)
    mapper.append_right_arm_action_at_gripper_tip_command(right_arm, zzk_action, debug_parts)

    np.testing.assert_array_almost_equal(zzk_action["right_arm"], [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(zzk_action["right_gripper"], 0.5)

    left_arm = np.array([0.3, 7, 8, 9, 10, 11, 12], dtype=np.float32)
    mapper.append_left_arm_action_at_gripper_tip_command(left_arm, zzk_action, debug_parts)

    np.testing.assert_array_almost_equal(zzk_action["left_arm"], [7, 8, 9, 10, 11, 12])
    np.testing.assert_allclose(zzk_action["left_gripper"], 0.3)


def test_action_mapper_generates_base_head_and_lift_commands():
    preprocessing_config = {
        "mmt_lowdim_flatten_indices_selection": {
            "base_action": [0, 1, 5],
            "head_action": [4, 5],
            "lift_action": [2],
        }
    }
    layouts = build_runtime_layouts(MMT_FIELD_LAYOUTS, preprocessing_config)
    mapper = MmtActionMapper(load_lowdim_index_selection(preprocessing_config), layouts)
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


# ── load_lowdim_field_remap tests ──


def test_load_lowdim_field_remap_parses_correctly():
    config = {
        "lowdim_field_remap": {
            "chest_T_eef_pose": [
                {"to": "chest_T_left_eef_pose", "indices": [0, 1, 2, 3, 4, 5, 6]},
                {"to": "chest_T_right_eef_pose", "indices": [7, 8, 9, 10, 11, 12, 13]},
            ],
        }
    }
    result = load_lowdim_field_remap(config)
    assert result == {
        "chest_T_left_eef_pose": ("chest_T_eef_pose", [0, 1, 2, 3, 4, 5, 6]),
        "chest_T_right_eef_pose": ("chest_T_eef_pose", [7, 8, 9, 10, 11, 12, 13]),
    }


def test_load_lowdim_field_remap_resolves_range_specs():
    config = {
        "lowdim_field_remap": {
            "arm_action": [
                {"to": "left_arm_action", "indices": [{"start": 0, "end": 7}]},
                {"to": "right_arm_action", "indices": [{"start": 7, "end": 14}]},
            ],
        }
    }
    result = load_lowdim_field_remap(config)
    assert result["left_arm_action"] == ("arm_action", [0, 1, 2, 3, 4, 5, 6])
    assert result["right_arm_action"] == ("arm_action", [7, 8, 9, 10, 11, 12, 13])


def test_load_lowdim_field_remap_empty_when_absent():
    assert load_lowdim_field_remap({}) == {}
    assert load_lowdim_field_remap({"lowdim_field_remap": None}) == {}


# ── build_runtime_layouts tests ──


def test_build_runtime_layouts_preserves_base_entries():
    layouts = build_runtime_layouts(MMT_FIELD_LAYOUTS, {})
    assert "base_action" in layouts
    assert layouts["base_action"]["full_dim"] == 6
    assert "chest_T_eef_pose" in layouts
    assert layouts["chest_T_eef_pose"]["zzk_source"]["type"] == "eef_pose"


def test_build_runtime_layouts_generates_action_remap_entries():
    config = {
        "lowdim_field_remap": {
            "arm_action": [
                {"to": "left_arm_action", "indices": [{"start": 0, "end": 7}]},
                {"to": "right_arm_action", "indices": [{"start": 7, "end": 14}]},
            ],
        }
    }
    layouts = build_runtime_layouts(MMT_FIELD_LAYOUTS, config)

    # Action remaps: full_dim derived, no zzk_source (parent has none)
    assert layouts["left_arm_action"]["full_dim"] == 7
    assert "zzk_source" not in layouts["left_arm_action"]
    assert layouts["right_arm_action"]["full_dim"] == 7


def test_build_runtime_layouts_generates_proprioception_remap_entries():
    config = {
        "lowdim_field_remap": {
            "chest_T_eef_pose": [
                {"to": "chest_T_left_eef_pose", "indices": [{"start": 0, "end": 7}]},
                {"to": "chest_T_right_eef_pose", "indices": [{"start": 7, "end": 14}]},
            ],
            "wrench": [
                {"to": "left_wrench", "indices": [{"start": 0, "end": 6}]},
                {"to": "right_wrench", "indices": [{"start": 6, "end": 12}]},
            ],
        }
    }
    layouts = build_runtime_layouts(MMT_FIELD_LAYOUTS, config)

    # Proprioception remaps: zzk_source with remap_slice type
    left_eef = layouts["chest_T_left_eef_pose"]
    assert left_eef["full_dim"] == 7
    assert left_eef["zzk_source"]["type"] == "remap_slice"
    assert left_eef["zzk_source"]["parent"] == "chest_T_eef_pose"
    assert left_eef["zzk_source"]["indices"] == list(range(7))

    right_wrench = layouts["right_wrench"]
    assert right_wrench["full_dim"] == 6
    assert right_wrench["zzk_source"]["type"] == "remap_slice"
    assert right_wrench["zzk_source"]["parent"] == "wrench"
    assert right_wrench["zzk_source"]["indices"] == [6, 7, 8, 9, 10, 11]


# ── remap-based action decode test ──


def test_action_mapper_decode_with_remap_derived_layout():
    config = _make_arm_remap_config()
    layouts = build_runtime_layouts(MMT_FIELD_LAYOUTS, config)
    selection = {"left_arm_action": [0, 1, 5]}
    mapper = MmtActionMapper(selection, layouts)

    values = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    full = mapper.decode_full_field_action("left_arm_action", values)
    expected = np.zeros(7, dtype=np.float32)
    expected[[0, 1, 5]] = [0.1, -0.2, 0.3]
    np.testing.assert_array_equal(full, expected)


# ── backward compatibility test ──


def test_action_mapper_backward_compat_without_runtime_layouts():
    """MmtActionMapper still works for base fields when no runtime_layouts given."""
    mapper = MmtActionMapper(load_lowdim_index_selection({}), MMT_FIELD_LAYOUTS)
    zzk_action = {}
    debug_parts = []
    mapper.append_base_action_command(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32), zzk_action, debug_parts)
    np.testing.assert_array_almost_equal(zzk_action["chassis"], [1, 2, 3, 4, 5, 6])
