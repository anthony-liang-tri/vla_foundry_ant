"""Canonical field layouts for MMT robotics inference.

Each entry defines:
  - full_dim: the full (pre-selection) dimension of the flattened field
  - groups: named index groups for sub-field access (e.g. left/right arm)
  - zzk_source (optional): how to extract this field from a ZZK status dict
    at inference time.  Omitted for action fields (they come from the model).

zzk_source types:
  - eef_pose: bimanual pose assembled from the state matrix rows
  - state_row: a single row of the state matrix
  - status_field: a top-level key in the ZZK status dict

ZZK state format (from zzk_api_ctypes_client.cc):
  state[0][0:6]  = local_T_chassis (x, y, z, rx, ry, rz)
  state[1][2]    = left_gripper_position
  state[2][0:6]  = chassis_T_left_arm_tip (x, y, z, rx, ry, rz)
  state[3][2]    = right_gripper_position
  state[4][0:6]  = chassis_T_right_arm_tip (x, y, z, rx, ry, rz)
  state[5][0:6]  = chest_T_left_arm_tip (x, y, z, rx, ry, rz)
  state[6][0:6]  = chest_T_right_arm_tip (x, y, z, rx, ry, rz)
  state[7][0:6]  = chassis_T_left_gripper_tip (x, y, z, rx, ry, rz)
  state[8][0:6]  = chassis_T_right_gripper_tip (x, y, z, rx, ry, rz)
  state[9][0:6]  = chest_T_left_gripper_tip (x, y, z, rx, ry, rz)
  state[10][0:6] = chest_T_right_gripper_tip (x, y, z, rx, ry, rz)
  chest_T_head    = chest_T_head pose [x, y, z, rx, ry, rz]
  chassis_T_chest = chassis_T_chest pose [x, y, z, rx, ry, rz]
  wrench          = [left_fx, left_fy, left_fz, left_tx, left_ty, left_tz,
                     right_fx, right_fy, right_fz, right_tx, right_ty, right_tz]
"""

ZZK_STATE_ROWS = 11
ZZK_STATE_COLS = 6

# arm_action split boundary: raw NPZ arm_action is 14-dim [left(7) | right(7)]
ARM_ACTION_SPLIT = 7

MMT_FIELD_LAYOUTS = {
    # ── action fields (no zzk_source — produced by the model) ──
    "left_arm_action": {
        "full_dim": 7,
        "groups": {
            "gripper": (0,),
            "arm": tuple(range(1, 7)),
        },
    },
    "right_arm_action": {
        "full_dim": 7,
        "groups": {
            "gripper": (0,),
            "arm": tuple(range(1, 7)),
        },
    },
    "base_action": {
        "full_dim": 6,
        "groups": {
            "chassis": tuple(range(6)),
        },
    },
    "head_action": {
        "full_dim": 6,
        "groups": {
            "head": tuple(range(6)),
        },
    },
    "left_arm_action_at_gripper_tip": {
        "full_dim": 7,
        "groups": {
            "gripper": (0,),
            "arm": tuple(range(1, 7)),
        },
    },
    "right_arm_action_at_gripper_tip": {
        "full_dim": 7,
        "groups": {
            "gripper": (0,),
            "arm": tuple(range(1, 7)),
        },
    },
    "lift_action": {
        "full_dim": 6,
        "groups": {
            "lift": tuple(range(6)),
        },
    },
    # ── proprioception fields (zzk_source defines how to extract from ZZK status) ──
    "chassis_T_eef_pose": {
        "full_dim": 14,
        "groups": {
            "left_gripper": (0,),
            "left_pose": tuple(range(1, 7)),
            "right_gripper": (7,),
            "right_pose": tuple(range(8, 14)),
        },
        "zzk_source": {
            "type": "eef_pose",
            "left_gripper_row": 1,
            "left_pose_row": 2,
            "right_gripper_row": 3,
            "right_pose_row": 4,
        },
    },
    "chest_T_eef_pose": {
        "full_dim": 14,
        "groups": {
            "left_gripper": (0,),
            "left_pose": tuple(range(1, 7)),
            "right_gripper": (7,),
            "right_pose": tuple(range(8, 14)),
        },
        "zzk_source": {
            "type": "eef_pose",
            "left_gripper_row": 1,
            "left_pose_row": 5,
            "right_gripper_row": 3,
            "right_pose_row": 6,
        },
    },
    "chassis_T_gripper_tip_pose": {
        "full_dim": 14,
        "groups": {
            "left_gripper": (0,),
            "left_pose": tuple(range(1, 7)),
            "right_gripper": (7,),
            "right_pose": tuple(range(8, 14)),
        },
        "zzk_source": {
            "type": "eef_pose",
            "left_gripper_row": 1,
            "left_pose_row": 7,
            "right_gripper_row": 3,
            "right_pose_row": 8,
        },
    },
    "chest_T_gripper_tip_pose": {
        "full_dim": 14,
        "groups": {
            "left_gripper": (0,),
            "left_pose": tuple(range(1, 7)),
            "right_gripper": (7,),
            "right_pose": tuple(range(8, 14)),
        },
        "zzk_source": {
            "type": "eef_pose",
            "left_gripper_row": 1,
            "left_pose_row": 9,
            "right_gripper_row": 3,
            "right_pose_row": 10,
        },
    },
    "base_pose": {
        "full_dim": 6,
        "groups": {
            "base": tuple(range(6)),
        },
        "zzk_source": {"type": "state_row", "row": 0},
    },
    "chest_T_head_pose": {
        "full_dim": 6,
        "groups": {
            "head_pose": tuple(range(6)),
        },
        "zzk_source": {"type": "status_field", "key": "chest_T_head_pose", "expected_size": 6},
    },
    "chassis_T_chest_pose": {
        "full_dim": 6,
        "groups": {
            "chest_pose": tuple(range(6)),
        },
        "zzk_source": {"type": "status_field", "key": "chassis_T_chest_pose", "expected_size": 6},
    },
    "wrench": {
        "full_dim": 12,
        "groups": {
            "left_wrench": tuple(range(6)),
            "right_wrench": tuple(range(6, 12)),
        },
        "zzk_source": {"type": "status_field", "key": "wrench", "expected_size": 12},
    },
}


def get_group_indices(field_name: str, group_name: str) -> tuple[int, ...]:
    if field_name not in MMT_FIELD_LAYOUTS:
        raise KeyError(f"Unknown MMT field layout '{field_name}'")
    groups = MMT_FIELD_LAYOUTS[field_name]["groups"]
    if group_name not in groups:
        raise KeyError(f"Unknown group '{group_name}' for MMT field layout '{field_name}'")
    return groups[group_name]
