"""Helper utilities for MMT inference lowdim decoding and ZZK action mapping."""

import numpy as np

from vla_foundry.inference.robotics.mmt.field_layouts import MMT_FIELD_LAYOUTS


def _to_index_list(selection) -> list[int] | None:
    """Normalize a preprocessing index selection into a simple list form."""
    if selection is None:
        return None
    if isinstance(selection, int):
        return [selection]
    return [int(idx) for idx in selection]


def load_lowdim_index_selection(preprocessing_config: dict) -> dict[str, list[int] | None]:
    """Extract per-field index selections from a preprocessing config.

    Converts the ``mmt_lowdim_flatten_indices_selection`` block into a dict
    that maps each field name to a list of selected indices (or None).

    Example input (from preprocessing_configs.yaml)::

        {"mmt_lowdim_flatten_indices_selection": {
            "chest_T_eef_pose": [7, 8, 9, 10, 11, 12, 13],  # right arm only
            "base_action": [0, 1, 5],
            "lift_action": 2,                                 # single int
        }}

    Example output::

        {"chest_T_eef_pose": [7, 8, 9, 10, 11, 12, 13],
         "base_action": [0, 1, 5],
         "lift_action": [2]}
    """
    index_selection_config = preprocessing_config.get("mmt_lowdim_flatten_indices_selection") or {}
    return {name: _to_index_list(sel) for name, sel in index_selection_config.items()}


def restore_full_values(values: np.ndarray, selection: list[int] | None, full_dim: int) -> np.ndarray:
    """Scatter selected values back into a zero-filled array of the original full dimension.

    Inverse of index selection: places each element of ``values`` at the
    corresponding position in ``selection``, leaving other positions as zero.

    Example::

        >>> restore_full_values(np.array([0.1, -0.2, 0.3]), [0, 1, 5], full_dim=6)
        array([0.1, -0.2, 0.0, 0.0, 0.0, 0.3], dtype=float32)
    """
    if selection is None:
        if len(values) == full_dim:
            return values.astype(np.float32, copy=False)
        raise ValueError(f"Cannot scatter values of dim {len(values)} into full dim {full_dim} without a selection.")

    full_values = np.zeros(full_dim, dtype=np.float32)
    if len(values) != len(selection):
        raise ValueError(f"Selection length {len(selection)} does not match values dim {len(values)}.")
    full_values[np.asarray(selection, dtype=np.int64)] = values
    return full_values


class MmtActionMapper:
    """Maps model action fields into ZZK command dictionaries."""

    def __init__(self, lowdim_index_selection: dict[str, list[int] | None]):
        self.lowdim_index_selection = lowdim_index_selection
        self.action_field_handlers = {
            "left_arm_action": self.append_left_arm_action_command,
            "right_arm_action": self.append_right_arm_action_command,
            "left_arm_action_at_gripper_tip": self.append_left_arm_action_at_gripper_tip_command,
            "right_arm_action_at_gripper_tip": self.append_right_arm_action_at_gripper_tip_command,
            "base_action": self.append_base_action_command,
            "head_action": self.append_head_action_command,
            "lift_action": self.append_lift_action_command,
        }

    def decode_full_field_action(self, field_name: str, action_values: np.ndarray) -> np.ndarray:
        if field_name not in MMT_FIELD_LAYOUTS:
            raise KeyError(f"Unknown MMT field layout '{field_name}'")
        full_dim = MMT_FIELD_LAYOUTS[field_name]["full_dim"]
        selection = self.lowdim_index_selection.get(field_name)
        return restore_full_values(action_values, selection, full_dim)

    def _append_single_arm_command(
        self,
        field_name: str,
        side_name: str,
        zzk_arm_key: str,
        action_values: np.ndarray,
        zzk_action: dict,
        debug_parts: list[str],
    ) -> None:
        full_action = self.decode_full_field_action(field_name, action_values)
        gripper = float(full_action[0])
        arm = full_action[1:]
        zzk_action[zzk_arm_key] = arm.tolist()
        zzk_action[f"{side_name}_gripper"] = [0, 0, gripper, 0, 0, 0]
        debug_parts.append(f"{side_name}[{zzk_arm_key}](vx={arm[0]:.3f}, vy={arm[1]:.3f}, gripper={gripper:.3f})")

    def append_left_arm_action_command(
        self,
        action_values: np.ndarray,
        zzk_action: dict,
        debug_parts: list[str],
    ) -> None:
        self._append_single_arm_command(
            "left_arm_action",
            "left",
            "left_arm",
            action_values,
            zzk_action,
            debug_parts,
        )

    def append_right_arm_action_command(
        self,
        action_values: np.ndarray,
        zzk_action: dict,
        debug_parts: list[str],
    ) -> None:
        self._append_single_arm_command(
            "right_arm_action",
            "right",
            "right_arm",
            action_values,
            zzk_action,
            debug_parts,
        )

    def append_left_arm_action_at_gripper_tip_command(
        self,
        action_values: np.ndarray,
        zzk_action: dict,
        debug_parts: list[str],
    ) -> None:
        self._append_single_arm_command(
            "left_arm_action_at_gripper_tip",
            "left",
            "left_arm_at_gripper_tip",
            action_values,
            zzk_action,
            debug_parts,
        )

    def append_right_arm_action_at_gripper_tip_command(
        self,
        action_values: np.ndarray,
        zzk_action: dict,
        debug_parts: list[str],
    ) -> None:
        self._append_single_arm_command(
            "right_arm_action_at_gripper_tip",
            "right",
            "right_arm_at_gripper_tip",
            action_values,
            zzk_action,
            debug_parts,
        )

    def append_base_action_command(self, action_values: np.ndarray, zzk_action: dict, debug_parts: list[str]) -> None:
        full_base_action = self.decode_full_field_action("base_action", action_values)
        zzk_action["chassis"] = full_base_action.tolist()
        debug_parts.append(
            f"chassis(vx={full_base_action[0]:.3f}, vy={full_base_action[1]:.3f}, yaw={full_base_action[5]:.3f})"
        )

    def append_head_action_command(self, action_values: np.ndarray, zzk_action: dict, debug_parts: list[str]) -> None:
        full_head_action = self.decode_full_field_action("head_action", action_values)
        zzk_action["head"] = full_head_action.tolist()
        debug_parts.append(
            f"head(roll={full_head_action[3]:.3f}, pitch={full_head_action[4]:.3f}, yaw={full_head_action[5]:.3f})"
        )

    def append_lift_action_command(self, action_values: np.ndarray, zzk_action: dict, debug_parts: list[str]) -> None:
        full_lift_action = self.decode_full_field_action("lift_action", action_values)
        zzk_action["lift"] = full_lift_action.tolist()
        debug_parts.append(f"lift(z={full_lift_action[2]:.3f})")
