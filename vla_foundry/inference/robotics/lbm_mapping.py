from typing import Any, Dict, List, Sequence

import fsspec
import numpy as np
import torch
import yaml
from pydrake.math import RigidTransform, RotationMatrix
from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers

from vla_foundry.data.robotics.utils import (
    rot_6d_from_relative,
    rot_6d_to_matrix,
    xyz_from_relative,
)
from vla_foundry.inference.robotics.utils import (
    any_to_actual_map,
    relative_to_absolute_map,
)


class ObservationMapping:
    """
    Utility wrapper to map observation fields of the robot in gym format to the policy format.

    Args:
        mapping_path: The path to the YAML field mapping definition.
        image_names: The names of the images in the observation (in the format of camera_name_t<timestep>).
    """

    def __init__(self, mapping_path: str, image_names: List[str], num_past_timesteps: int = None):
        with fsspec.open(mapping_path, "r") as handle:
            mapping = yaml.safe_load(handle)
        self._field_paths = mapping["field_paths"]
        self._laterality = ("left", "right")
        self._components = ("xyz", "rot_6d", "gripper", "joint_position")
        self._types = ("actual", "action", "desired")
        self._type_map = self._build_type_map()
        self._camera_names = self._get_camera_names(image_names)
        self.num_past_timesteps = num_past_timesteps

    def _build_type_map(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        type_map: Dict[str, Dict[str, Dict[str, str]]] = {
            type_name: {side: {} for side in self._laterality} for type_name in self._types
        }

        for field in self._field_paths:
            type_name = None
            for _type in self._types:
                if _type in field:
                    type_name = _type
                    break
            side_name = None
            for side in self._laterality:
                if side in field:
                    side_name = side
                    break
            component_name = None
            for component in self._components:
                if component in field:
                    component_name = component
                    break
            if type_name is None or side_name is None or component_name is None:
                raise ValueError(
                    f"Field '{field}' could not be mapped: "
                    f"type_name={type_name}, side_name={side_name}, component_name={component_name}"
                )
            type_map[type_name][side_name][component_name] = field

        return type_map

    def _get_camera_names(self, image_names: List[str]) -> List[str]:
        camera_names: List[str] = []
        for img_name in image_names:
            if "_t" in img_name:
                camera_name, _timestep_str = img_name.rsplit("_t", 1)
            else:
                camera_name = img_name
            if camera_name not in camera_names:
                camera_names.append(camera_name)
        return camera_names

    def _path(self, field: str) -> Sequence[str]:
        absolute = relative_to_absolute_map(field)
        return self._field_paths[absolute]

    def get_type(self, observation: MultiarmObservation, type_name: str) -> Dict[str, Dict[str, str]]:
        if "actual" in type_name:
            return observation.robot.actual
        elif "desired" in type_name:
            return observation.robot.desired
        else:
            raise ValueError(f"Invalid type name: {type_name}")

    def get_component(self, observation: MultiarmObservation, type: str, laterality: str, component: str) -> np.ndarray:
        """
        Get the component of the pose or gripper of the robot.
        Args:
            observation: The observation of the robot.
            type: The type of the component in (actual, action, desired).
            laterality: The laterality of the component in (left, right).
            component: The component of the pose or gripper in (xyz, rot_6d, gripper).
        Returns:
            The component of the pose or gripper of the robot.
        """

        path = self._path(self._type_map[type][laterality][component])
        robot_data = self.get_type(observation, type)
        if component == "gripper":
            robot_data = robot_data.grippers
            return robot_data[path[0]]
        elif component == "joint_position":
            robot_data = robot_data.joint_position
            return robot_data[path[0]]
        else:
            robot_data = robot_data.poses
            if "xyz" in component:
                return robot_data[path[0]].translation()
            elif "rot_6d" in component:
                return robot_data[path[0]].rotation().matrix()[:, :2].flatten("F")
            else:
                raise ValueError(f"Invalid component: {component} for pose")

    def get_type_name(self, field: str) -> str:
        for type_name in self._types:
            if type_name in field:
                return type_name
        raise ValueError(f"Invalid field: {field}")

    def get_component_name(self, field: str) -> str:
        for component in self._components:
            if component in field:
                return component
        raise ValueError(f"Invalid field: {field}")

    def get_laterality_name(self, field: str) -> str:
        for laterality in self._laterality:
            if laterality in field:
                return laterality
        raise ValueError(f"Invalid field: {field}")

    def get_field(self, observation: MultiarmObservation, field: str) -> np.ndarray:
        """
        Get the field of the robot.
        Args:
            observation: The observation of the robot.
            field: The field of the robot.
        Returns:
            The field of the robot.
        """
        type_name, laterality, component = (
            self.get_type_name(field),
            self.get_laterality_name(field),
            self.get_component_name(field),
        )
        return self.get_component(observation, type_name, laterality, component)

    def get_all_images(self, observation: MultiarmObservation) -> Dict[str, Any]:
        """
        Get all the images of the observation in the order listed in the image_names.
        Args:
            observation: The observation of the robot.
        Returns:
            The all the images of the robot.
        """
        images: Dict[str, Any] = {}
        for camera_name in self._camera_names:
            if camera_name in observation.visuo:
                images[camera_name] = observation.visuo[camera_name].rgb.array.copy()
        return images


class ActionMapping:
    """
    Utility wrapper to map action fields of the robot in gym format to the policy format and vice versa.
    """

    def __init__(
        self,
        mapping_path: str,
        action_fields: List[str],
        robotics_processor,
        clamp_std: float = 1.0,
        num_past_timesteps: int = None,
    ):
        with fsspec.open(mapping_path, "r") as handle:
            mapping = yaml.safe_load(handle)
        self._field_paths = mapping["field_paths"]
        self.action_fields = action_fields
        self.relative_action_fields = [field for field in self.action_fields if field.endswith("_relative")]
        self.field_dims = {
            field: robotics_processor.normalizer.get_field_dimension(field) for field in self.action_fields
        }
        self.action_dim = sum(self.field_dims.values())
        self.normalizer = robotics_processor.normalizer
        self.clamp_std = clamp_std
        self.num_past_timesteps = num_past_timesteps

    def get_field_std(self, field: str, scope: str = "per_timestep") -> np.ndarray:
        """Get the standard deviation for a field from the normalizer statistics."""
        absolute_field = relative_to_absolute_map(field)
        if absolute_field not in self.normalizer.stats:
            raise ValueError(f"Field {field} (absolute: {absolute_field}) not found in normalizer statistics")

        field_stats = self.normalizer.stats[absolute_field]

        if scope == "global":
            return np.array(field_stats["std"], dtype=np.float64)
        else:
            return np.array(field_stats["std_per_timestep"], dtype=np.float64)

    def from_action_model(
        self,
        action_from_model: torch.Tensor,
        normalizer,
        reference: Dict[str, np.ndarray],
    ) -> dict:
        """
        Convert the action from the action model output format to the buffer action format.

        Uses the action buffer as reference by default, but clips it to be within self.clamp_std std
        of the actual observation reference to prevent drift.

        Args:
            action_from_model: The action from the action model output format. (B, T, D)
            normalizer: The normalizer to denormalize the action.
            reference: The reference to convert the relative action fields to absolute action fields.
        Returns:
            The action in the buffer action format.
        """
        action = {}
        start_idx = 0
        # Denormalize all action fields
        for field in self.action_fields:
            dim = self.field_dims[field]
            action[field] = (
                normalizer.denormalize_tensor(
                    action_from_model[..., start_idx : start_idx + dim], field, anchor_timestep=self.num_past_timesteps
                )
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            start_idx += dim

        # Convert relative to current position action fields to absolute action fields
        for field in self.relative_action_fields:
            absolute_field = relative_to_absolute_map(field)
            absolute_actual_field = any_to_actual_map(absolute_field)

            # Get the action buffer reference and actual observation reference
            actual_reference = reference[absolute_actual_field]

            # Use the clipped reference for relative-to-absolute conversion
            if "xyz" in field or "gripper" in field:
                action[absolute_field] = xyz_from_relative(action[field], actual_reference)
            elif "rot_6d" in field:
                action[absolute_field] = rot_6d_from_relative(action[field], actual_reference)
            else:
                raise ValueError(f"Invalid field: {field}")

        # Create a list of action dictionaries for each timestep instead of a dictionary of sequences
        action_list = []
        absolute_fields = [relative_to_absolute_map(field) for field in self.action_fields]
        for t in range(action_from_model.shape[1]):
            action_dict = {absolute_field: action[absolute_field][0, t, :] for absolute_field in absolute_fields}
            action_list.append(action_dict)
        return action_list

    def create_pose_and_gripper(
        self,
        action: dict,
    ) -> PosesAndGrippers:
        grippers = {}
        poses = {}
        for field in self.action_fields:
            absolute_field = relative_to_absolute_map(field)
            mapping_fields = self._field_paths[absolute_field]
            if len(mapping_fields) == 1:
                grippers[mapping_fields[0]] = action[absolute_field]
            elif len(mapping_fields) == 2:
                action_field = action[absolute_field]
                if action_field.shape[0] == 3:
                    field_name = "p"
                else:
                    assert action_field.shape[0] == 6
                    rot_matrix = rot_6d_to_matrix(action_field)
                    action_field = RotationMatrix(rot_matrix)
                    field_name = "R"
                if mapping_fields[0] in poses:
                    poses[mapping_fields[0]][field_name] = action_field
                    poses[mapping_fields[0]] = RigidTransform(
                        R=RotationMatrix(poses[mapping_fields[0]]["R"]), p=poses[mapping_fields[0]]["p"]
                    )
                else:
                    poses[mapping_fields[0]] = {field_name: action_field}
            else:
                raise ValueError(f"Unknown action field: {field}")

        return PosesAndGrippers(poses=poses, grippers=grippers)

    def from_sim(self, action_from_sim: PosesAndGrippers) -> dict:
        action = {}
        for field in self.action_fields:
            absolute_field = relative_to_absolute_map(field)
            mapping_fields = self._field_paths[absolute_field]
            # TODO: find a better way than to check the length of the mapping fields
            if len(mapping_fields) == 1:
                action[absolute_field] = action_from_sim.grippers[mapping_fields[0]]
            elif len(mapping_fields) == 2:
                if "xyz" in field:
                    action[absolute_field] = action_from_sim.poses[mapping_fields[0]].translation()
                elif "rot" in field:
                    action[absolute_field] = (
                        action_from_sim.poses[mapping_fields[0]].rotation().matrix()[:, :2].flatten("F")
                    )
                else:
                    raise ValueError(f"Unknown field: {field}")
            else:
                raise ValueError(f"Unknown action field: {field}")
        return action
