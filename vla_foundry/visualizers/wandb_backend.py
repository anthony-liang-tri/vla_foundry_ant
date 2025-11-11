"""
WandB Backend Implementation

This file contains the implementation for the WandbBackend class, which provides
logging functionality to Weights & Biases (wandb).
"""

from typing import Any, Dict, List

import numpy as np
from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers

import wandb


class WandbBackend:
    name = "wandb"

    def __init__(self) -> None:
        """
        Initialize the WandbBackend instance.
        """
        self._initialized = False
        self._run = None

    def init(self, run_name: str, **kwargs) -> None:
        """
        Initialize the WandB visualizer.

        Parameters:
        - run_name: Name of the run.
        - **kwargs: Additional arguments passed to wandb.init().
        """
        if not self._initialized:
            self._run = wandb.init(project="vla_foundry", name=run_name)
            self._initialized = True

    def flush(self) -> None:
        """
        Flush any pending logs to the WandB backend.
        """
        if self._run:
            self._run.log({})  # WandB automatically flushes logs; this is a no-op.

    def shutdown(self) -> None:
        """
        Shutdown the WandB backend and clean up resources.
        """
        if self._run:
            self._run.finish()
            self._run = None
            self._initialized = False

    def log_image(self, path: str, image: np.ndarray, **kwargs) -> None:
        """
        Log an image to the WandB backend.

        Parameters:
        - path: The hierarchical path for the image.
        - image: The image data as a NumPy array.
        """
        wandb.log({path: wandb.Image(image)})

    def log_scalar(self, path: str, value: float, **kwargs) -> None:
        """
        Log a scalar value to the WandB backend.

        Parameters:
        - path: The hierarchical path for the scalar.
        - value: The scalar value.
        """
        wandb.log({path: value})

    def log_points3d(self, path: str, points: np.ndarray, **kwargs) -> None:
        """
        Log 3D points to the WandB backend.

        Parameters:
        - path: The hierarchical path for the 3D points.
        - points: The 3D points as a NumPy array of shape (N, 3).
        """
        wandb.log({path: wandb.Object3D(points)})

    def log_line_strips3d(self, path: str, line_strips: np.ndarray, **kwargs) -> None:
        """
        Log 3D line strips to the WandB backend.

        Parameters:
        - path: The hierarchical path for the line strips.
        - line_strips: The 3D line strips as a NumPy array of shape (N, 3).
        """
        wandb.log({path: wandb.Object3D(line_strips)})

    def log_trajectory(self, path: str, trajectory: np.ndarray, **kwargs) -> None:
        """
        Log a trajectory to the WandB backend.

        Parameters:
        - path: The hierarchical path for the trajectory.
        - trajectory: The trajectory as a NumPy array of shape (N, 3).
        """
        self.log_line_strips3d(f"{path}/trajectory", trajectory, **kwargs)
        self.log_points3d(f"{path}/waypoints", trajectory, **kwargs)

    def log_rigid_transform(self, path: str, transform: Any, **kwargs) -> None:
        """
        Log a rigid transform to the WandB backend.

        Parameters:
        - path: The hierarchical path for the transform.
        - transform: The rigid transform object.
        """
        translation = transform.translation()
        rotation = transform.rotation().ToQuaternion()
        wandb.log(
            {
                f"{path}/translation": translation.tolist(),
                f"{path}/rotation": [rotation.w(), rotation.x(), rotation.y(), rotation.z()],
            }
        )

    def log_arm_poses(self, arm_poses: Dict[str, PosesAndGrippers], **kwargs) -> None:
        """
        Log arm poses to the WandB backend.

        Parameters:
        - arm_poses: A dictionary containing arm pose data.
        """
        for client_id, poses_and_grippers in arm_poses.items():
            if not poses_and_grippers or not hasattr(poses_and_grippers, "poses"):
                continue

            for model_name, transform in poses_and_grippers.poses.items():
                self.log_rigid_transform(f"clients/{client_id}/models/{model_name}/pose", transform)

            if hasattr(poses_and_grippers, "grippers") and poses_and_grippers.grippers:
                for gripper_name, value in poses_and_grippers.grippers.items():
                    self.log_scalar(f"clients/{client_id}/grippers/{gripper_name}/grip", value)

    def log_action_predictions(self, predictions: List[PosesAndGrippers], **kwargs) -> None:
        """
        Log action predictions to the WandB backend.

        Parameters:
        - predictions: A list of objects containing action prediction data.
        """
        traj = {}
        for step in predictions:
            poses = getattr(step, "poses", {})
            for model_name, transform in poses.items():
                if model_name not in traj:
                    traj[model_name] = []
                traj[model_name].append(transform.translation().tolist())

        for model_name, points in traj.items():
            if len(points) >= 2:
                self.log_line_strips3d(f"predictions/{model_name}/trajectory", np.array(points))
            if points:
                self.log_points3d(f"predictions/{model_name}/waypoints", np.array(points))

    def log_multiarm_observation(self, path: str, observation: MultiarmObservation, **kwargs) -> None:
        """
        Log a MultiarmObservation to the WandB backend.

        Parameters:
        - path: The hierarchical path for the observation.
        - observation: The MultiarmObservation object to log.
        """
        self.log_arm_poses({f"{path}/robot": observation.robot.actual}, **kwargs)

        for camera_id, image_set in observation.visuo.items():
            if image_set.rgb:
                self.log_image(f"{path}/cameras/{camera_id}/rgb", image_set.rgb.array, **kwargs)
            if image_set.depth:
                self.log_image(f"{path}/cameras/{camera_id}/depth", image_set.depth.array, **kwargs)
            if image_set.label:
                self.log_image(f"{path}/cameras/{camera_id}/label", image_set.label.array, **kwargs)

        if observation.language_instruction:
            self.log_text(f"{path}/language_instruction", observation.language_instruction, **kwargs)

    def log_text(self, path: str, text: str, **kwargs) -> None:
        """
        Log a text value to the WandB backend.

        Parameters:
        - path: The hierarchical path for the text.
        - text: The text value to log.
        """
        wandb.log({path: text})
