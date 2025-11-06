"""
Rerun Backend Implementation

This file contains the implementation of the RerunBackend class, which provides
logging functionality to rerun.io.
"""

import subprocess
from typing import Any, Dict, List, Mapping

import numpy as np
import rerun as rr
from pydrake.math import RigidTransform
from rerun import Transform3D
from rerun.datatypes import Quaternion
from robot_gym.multiarm_spaces import PosesAndGrippers  # Import from robot_gym.multiarm_spaces


class RerunBackend:
    name = "rerun"

    def __init__(self) -> None:
        """
        Initialize the RerunBackend instance.
        """
        self._initialized = False

    def init(self, run_name: str, add_rank_to_run: bool = False, **kwargs) -> None:
        """
        Initialize the Rerun visualizer.

        Parameters:
        - run_name: Name of the run.
        - add_rank_to_run: Whether to add rank information to the run name.
        - **kwargs: Additional arguments (ignored).
        """
        if not self._initialized:
            rr.init(run_name, spawn=True)
            self._initialized = True

    def log_image(self, path: str, image: np.ndarray, **kwargs) -> None:
        """
        Log an image to the Rerun backend.

        Parameters:
        - path: The hierarchical path for the image.
        - image: The image data as a NumPy array.
        """
        rr.log(path, rr.Image(image))

    def log_images(self, path: str, images: Dict[str, np.ndarray], **kwargs) -> None:
        """
        Log multiple images to the Rerun backend.

        Parameters:
        - images: A dictionary where keys are image paths and values are NumPy arrays representing the images.
        """
        for path, image in images.items():
            self.log_image(path, image, **kwargs)

    def log_scalar(self, path: str, value: float, **kwargs) -> None:
        """
        Log a scalar value to the Rerun backend.

        Parameters:
        - path: The hierarchical path for the scalar.
        - value: The scalar value.
        """
        rr.log(path, rr.Scalars(value))

    def log_points3d(self, path: str, points: np.ndarray, **kwargs) -> None:
        """
        Log 3D points to the Rerun backend.

        Parameters:
        - path: The hierarchical path for the 3D points.
        - points: The 3D points as a NumPy array of shape (N, 3).
        """
        rr.log(path, rr.Points3D(points))

    def log_trajectory(self, path: str, trajectory: np.ndarray, **kwargs) -> None:
        """
        Log a trajectory to the Rerun backend.

        Parameters:
        - path: The hierarchical path for the trajectory.
        - trajectory: The trajectory as a NumPy array of shape (N, 3).
        """
        rr.log(path, rr.LineStrips3D([trajectory]))
        rr.log(f"{path}/waypoints", rr.Points3D(trajectory))

    def log_rigid_transform(self, path: str, transform: RigidTransform, axis_length: float = 1.0, **kwargs) -> None:
        """
        Log a rigid transform to the Rerun backend.

        Parameters:
        - path: The hierarchical path for the rigid transform.
        - transform: The RigidTransform object.
        - axis_length: The length of the axes for visualization.
        """
        translation = transform.translation()
        rotation = transform.rotation().ToQuaternion()
        quaternion = np.roll([rotation.w(), rotation.x(), rotation.y(), rotation.z()], -1)  # Drake to Rerun order
        rr.log(
            path,
            Transform3D(
                translation=translation,
                quaternion=quaternion,
                axis_length=axis_length,
            ),
        )

    def log_arm_poses(self, arm_poses: Dict[str, PosesAndGrippers], **kwargs: Any) -> None:
        """
        Log arm poses to the Rerun backend.

        Parameters:
        - arm_poses: A dictionary containing arm pose data.
        """
        for client_id, poses_and_grippers in arm_poses.items():
            if not poses_and_grippers or not hasattr(poses_and_grippers, "poses"):
                continue

            for model_name, transform in poses_and_grippers.poses.items():
                rotation = transform.rotation().ToQuaternion()
                rr.log(
                    f"clients/{client_id}/models/{model_name}/pose",
                    Transform3D(
                        translation=transform.translation(),
                        rotation=Quaternion(xyzw=[rotation.x(), rotation.y(), rotation.z(), rotation.w()]),
                        axis_length=0.25,
                    ),
                )

            if hasattr(poses_and_grippers, "grippers") and poses_and_grippers.grippers:
                for gripper_name, value in poses_and_grippers.grippers.items():
                    rr.log(f"clients/{client_id}/grippers/{gripper_name}/grip", rr.Scalar(value))

    def log_action_predictions(self, predictions: List[PosesAndGrippers], **kwargs: Any) -> None:
        """
        Log action predictions to the Rerun backend.

        Parameters:
        - predictions: A list of PosesAndGrippers objects containing action prediction data.
        """
        traj = {}
        for step in predictions:
            poses = getattr(step, "poses", {})
            for model_name, transform in poses.items():
                if model_name not in traj:
                    traj[model_name] = []
                traj[model_name].append(transform.translation())

        for model_name, points in traj.items():
            if len(points) >= 2:
                rr.log(f"predictions/{model_name}/trajectory", rr.LineStrips3D([np.array(points)]))
                rr.log(f"predictions/{model_name}/waypoints", rr.Points3D(np.array(points)))
            elif points:
                rr.log(f"predictions/{model_name}/waypoints", rr.Points3D(np.array(points)))

    def log_line_strips3d(self, path: str, line_strips: np.ndarray, **kwargs) -> None:
        """
        Log 3D line strips to the Rerun backend.

        Parameters:
        - path: The hierarchical path for the line strips.
        - line_strips: The 3D line strips as a NumPy array of shape (N, 3).
        """
        rr.log(path, rr.LineStrips3D([line_strips]))

    def flush(self) -> None:
        # rerun flush is implicit; no-op here
        return

    def shutdown(self) -> None:
        # rerun doesn't strictly need it; keep for parity
        return

    def log_dict(
        self,
        data: Mapping[str, Any],
        *,
        base_path: str = "",
        recurse: bool = True,
        sanitize_keys: bool = True,
        max_depth: int = 8,
        _depth: int = 0,
        **kwargs,
    ) -> None:
        if not self._initialized or data is None:
            return
        if _depth > max_depth:
            print("[rerun_backend] log_dict: max_depth exceeded; truncating.")
            return

        def _join(a: str, b: str) -> str:
            if not a:
                return b
            return f"{a.rstrip('/')}/{b.lstrip('/')}"

        def _clean(k: str) -> str:
            if not sanitize_keys:
                return k
            return "".join(ch if ch.isalnum() or ch in "-_./" else "_" for ch in str(k))

        for k, v in data.items():
            key = _clean(k)
            p = _join(base_path, key)

            if recurse and isinstance(v, Mapping):
                self.log_dict(
                    v,
                    base_path=p,
                    recurse=recurse,
                    sanitize_keys=sanitize_keys,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    **kwargs,
                )
                continue

            self.log(p, v, **kwargs)

    def _disable_rerun_analytics(self) -> None:
        try:
            subprocess.run(
                ["rerun", "analytics", "disable"],
                check=True,
                capture_output=True,
                text=True,
            )
            print("[rerun_backend] Rerun analytics disabled.")
        except FileNotFoundError:
            print("[rerun_backend] rerun CLI not found; analytics may be enabled.")
        except subprocess.CalledProcessError as e:
            print(f"[rerun_backend] Failed to disable analytics: {e.stderr or e}")
