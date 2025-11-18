"""
WandB Backend Implementation

This file contains the implementation for the WandbBackend class, which provides
logging functionality to Weights & Biases (wandb).
"""

from typing import Any

import numpy as np

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

    def log_images(self, path: str, images: Any, **kwargs) -> None:
        """
        Log images to the WandB backend. Supports both single images and dictionaries of images.

        Parameters
        ----------
        path : str
            Base path in the visualization hierarchy.
        images : Any
            Either a single NumPy array representing an image or a dictionary of images.
        """
        if isinstance(images, np.ndarray):
            wandb.log({path: wandb.Image(images)})
        elif isinstance(images, dict):
            for sub_path, image in images.items():
                wandb.log({f"{path}/{sub_path}": wandb.Image(image)})
        else:
            raise TypeError("Unsupported type for 'images'. Must be a NumPy array or a dictionary of NumPy arrays.")

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

    def log_pose(
        self, path: str, translation: np.ndarray, rotation: np.ndarray, axis_length: float = 1.0, **kwargs
    ) -> None:
        """
        Log a generic pose to the WandB backend.

        Parameters
        ----------
        path : str
            Path in the visualization hierarchy.
        translation : np.ndarray
            Translation vector of shape (3,).
        rotation : np.ndarray
            Quaternion [x, y, z, w] of shape (4,).
        axis_length : float, optional
            Length of the axes for visualization, by default 1.0.
        """
        # WandB does not have a direct pose primitive, so log translation and quaternion as separate fields
        wandb.log(
            {
                f"{path}/translation": translation.tolist(),
                f"{path}/quaternion": rotation.tolist(),
                f"{path}/axis_length": axis_length,
            }
        )

    def log_text(self, path: str, text: str, **kwargs) -> None:
        """
        Log a text value to the WandB backend.

        Parameters
        ----------
        path : str
            Path in the visualization hierarchy.
        text : str
            Text value to log.
        """
        wandb.log({path: text})
