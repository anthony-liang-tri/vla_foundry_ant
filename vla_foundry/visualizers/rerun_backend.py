"""
Rerun Backend Implementation

This file contains the implementation of the RerunBackend class, which provides
logging functionality to rerun.io.
"""

import subprocess
from typing import Dict

import numpy as np
import rerun as rr
from rerun import Transform3D
from rerun.datatypes import Quaternion


class RerunBackend:
    name = "rerun"

    def __init__(self) -> None:
        """
        Initialize the RerunBackend instance.
        """
        self._initialized = False
        self._disable_rerun_analytics()  # Ensure analytics are disabled during initialization

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

    def flush(self) -> None:
        # rerun flush is implicit; no-op here
        return

    def shutdown(self) -> None:
        # rerun doesn't strictly need it; keep for parity
        return

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
            print("[rerun_backend] rerun CLI not found; analytics may be enabled. Ensure the rerun CLI is installed.")
        except subprocess.CalledProcessError as e:
            print(f"[rerun_backend] Failed to disable analytics: {e.stderr or e}")

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

    def log_pose(
        self, path: str, translation: np.ndarray, rotation: np.ndarray, axis_length: float = 1.0, **kwargs
    ) -> None:
        """
        Log a generic pose to the Rerun backend.

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
        rr.log(
            path,
            Transform3D(
                translation=translation,
                rotation=Quaternion(xyzw=rotation),
                axis_length=axis_length,
            ),
        )

    def log_line_strips3d(self, path: str, line_strips: np.ndarray, **kwargs) -> None:
        """
        Log 3D line strips to the Rerun backend.

        Parameters
        ----------
        path : str
            The hierarchical path for the line strips.
        line_strips : np.ndarray
            The 3D line strips as a NumPy array of shape (N, 3).
        """
        rr.log(path, rr.LineStrips3D([line_strips]))

    def log_text(self, path: str, text: str, **kwargs) -> None:
        """
        Log a text value to the Rerun backend.

        Parameters
        ----------
        path : str
            Path in the visualization hierarchy.
        text : str
            Text value to log.
        """
        rr.log(path, rr.TextLog(text))
