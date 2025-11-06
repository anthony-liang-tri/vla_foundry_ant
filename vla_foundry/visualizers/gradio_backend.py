from typing import Any, Dict, List

import gradio as gr
import numpy as np
from pydrake.math import RigidTransform
from robot_gym.multiarm_spaces import PosesAndGrippers  # Import from robot_gym.multiarm_spaces

# State to store logged data
state = {
    "images": [],
    "scalars": [],
    "points3d": [],
    "trajectories": [],
    "rigid_transforms": [],
    "arm_poses": [],
    "action_predictions": [],
}


class GradioBackend:
    name = "gradio"

    def __init__(self) -> None:
        """
        Initialize the GradioBackend instance.
        """
        self._app_launched = False  # Track if the app has been launched

    def init(self, run_name: str, add_rank_to_run: bool = False, **kwargs) -> None:
        """
        Initialize the Gradio visualizer.

        Parameters:
        - run_name: Name of the run.
        - add_rank_to_run: Whether to add rank information to the run name.
        - **kwargs: Additional arguments (ignored).
        """
        print(f"Gradio visualizer initialized with run_name={run_name}, add_rank_to_run={add_rank_to_run}")

    def log_image(self, path: str, image: np.ndarray, **kwargs) -> None:
        """
        Log an image to the Gradio backend.

        Parameters:
        - path: The hierarchical path for the image.
        - image: The image data as a NumPy array.
        """
        state["images"].append((path, image))

    def log_scalar(self, path: str, value: float, **kwargs) -> None:
        """
        Log a scalar value to the Gradio backend.

        Parameters:
        - path: The hierarchical path for the scalar.
        - value: The scalar value.
        """
        state["scalars"].append((path, value))

    def log_points3d(self, path: str, points: np.ndarray, **kwargs) -> None:
        """
        Log 3D points to the Gradio backend.

        Parameters:
        - path: The hierarchical path for the 3D points.
        - points: The 3D points as a NumPy array of shape (N, 3).
        """
        state["points3d"].append((path, points))

    def log_trajectory(self, path: str, trajectory: np.ndarray, **kwargs) -> None:
        """
        Log a trajectory to the Gradio backend.

        Parameters:
        - path: The hierarchical path for the trajectory.
        - trajectory: The trajectory as a NumPy array of shape (N, 3).
        """
        state["trajectories"].append((path, trajectory))

    def log_rigid_transform(self, path: str, transform: RigidTransform, axis_length: float = 1.0, **kwargs) -> None:
        """
        Log a rigid transform to the Gradio backend.

        Parameters:
        - path: The hierarchical path for the rigid transform.
        - transform: The RigidTransform object.
        - axis_length: The length of the axes for visualization.
        """
        state["rigid_transforms"].append((path, transform, axis_length))

    def log_arm_poses(self, arm_poses: Dict[str, PosesAndGrippers], **kwargs: Any) -> None:
        """
        Log arm poses to the Gradio backend.

        Parameters:
        - arm_poses: A dictionary containing arm pose data.
        """
        state["arm_poses"].append(arm_poses)

    def log_action_predictions(self, predictions: List[PosesAndGrippers], **kwargs: Any) -> None:
        """
        Log action predictions to the Gradio backend.

        Parameters:
        - predictions: A list of action prediction data.
        """
        state["action_predictions"].append(predictions)

    def log_line_strips3d(self, path: str, line_strips: np.ndarray, **kwargs) -> None:
        """
        Log 3D line strips to the Gradio backend.

        Parameters:
        - path: The hierarchical path for the line strips.
        - line_strips: The 3D line strips as a NumPy array of shape (N, 3).
        """
        state["trajectories"].append((path, line_strips))

    def flush(self) -> None:
        pass  # No-op for Gradio

    def shutdown(self) -> None:
        """
        Handle cleanup during shutdown. Ensure the app lifecycle is not interrupted.
        """
        if self._app_launched:
            print("[GradioBackend] App is running. Shutdown will not interrupt the app.")
        else:
            print("[GradioBackend] Shutting down. Cleanup complete.")

    def launch(self) -> None:
        """
        Launch the Gradio app. This should be called explicitly during execution.
        """
        if self._app_launched:
            print("[GradioBackend] App already launched.")
            return

        def display_images():
            # Prepare images for rendering in the gallery with valid captions
            return [(image, f"Path: {tag}") for tag, image in state["images"]]

        def display_state():
            # Preprocess state for rendering
            return {
                "Scalars": [{"tag": tag, "value": value} for tag, value in state["scalars"]],
                "3D Points": [
                    {"tag": tag, "points": points.tolist()}  # Convert NumPy array to list
                    for tag, points in state["points3d"]
                ],
                "Trajectories": [
                    {"tag": tag, "trajectory": trajectory.tolist()}  # Convert NumPy array to list
                    for tag, trajectory in state["trajectories"]
                ],
                "Rigid Transforms": [
                    {
                        "tag": tag,
                        "translation": transform.translation().tolist(),
                        "rotation": transform.rotation().ToQuaternion().wxyz().tolist(),
                        "axis_length": axis_length,
                    }
                    for tag, transform, axis_length in state["rigid_transforms"]
                ],
                "Arm Poses": [
                    {"client": client, "poses": str(poses)} for client, poses in enumerate(state["arm_poses"])
                ],
                "Action Predictions": [
                    {"step": idx, "poses": str(pred)} for idx, pred in enumerate(state["action_predictions"])
                ],
            }

        with gr.Blocks() as demo:
            gr.Markdown("# Gradio Visualizer")

            # Display images dynamically
            with gr.Row():
                image_gallery = gr.Gallery(label="Images")

            # Display other data as JSON
            with gr.Row():
                state_display = gr.JSON(label="Logged Data")

            # Refresh button to update the data
            gr.Button("Refresh").click(
                fn=lambda: (display_images(), display_state()),
                inputs=[],
                outputs=[image_gallery, state_display],
            )

        self._app_launched = True
        print("[GradioBackend] Launching Gradio app. Press Ctrl+C to stop.")
        demo.launch()
