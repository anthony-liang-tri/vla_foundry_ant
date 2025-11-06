from typing import Any, Dict, List

import gradio as gr
import numpy as np
from plotly.graph_objects import Figure, Scatter3d
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
        if points.size == 0:
            print(f"[Warning] Empty points data for path: {path}")
            return
        state["points3d"].append((path, points))

    def _create_3d_trajectory_plot(self, trajectory: np.ndarray) -> Figure:
        """
        Create a 3D plot for a trajectory using Plotly.

        Parameters:
        - trajectory: The trajectory as a NumPy array of shape (N, 3).

        Returns:
        - A Plotly Figure object for the 3D trajectory.
        """
        print(f"Creating 3D trajectory plot for trajectory: {trajectory}")  # Debug print
        fig = Figure(
            data=[
                Scatter3d(
                    x=trajectory[:, 0],
                    y=trajectory[:, 1],
                    z=trajectory[:, 2],
                    mode="lines+markers",
                    marker=dict(size=4),
                    line=dict(width=2),
                )
            ]
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
            title="3D Trajectory",
        )
        return fig

    def _create_3d_pose_plot(self, poses: Dict[str, RigidTransform]) -> Figure:
        """
        Create a 3D plot for poses using Plotly.

        Parameters:
        - poses: A dictionary of RigidTransform objects representing poses.

        Returns:
        - A Plotly Figure object for the 3D poses.
        """
        print(f"Creating 3D pose plot for poses: {poses}")  # Debug print
        data = []
        for name, pose in poses.items():
            translation = pose.translation()
            data.append(
                Scatter3d(
                    x=[translation[0]],
                    y=[translation[1]],
                    z=[translation[2]],
                    mode="markers",
                    marker=dict(size=6),
                    name=name,
                )
            )
        fig = Figure(data=data)
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
            title="3D Poses",
        )
        return fig

    def log_trajectory(self, path: str, trajectory: np.ndarray, **kwargs) -> None:
        """
        Log a trajectory to the Gradio backend.

        Parameters:
        - path: The hierarchical path for the trajectory.
        - trajectory: The trajectory as a NumPy array of shape (N, 3).
        """
        if trajectory.size == 0:
            print(f"[Warning] Empty trajectory data for path: {path}")
            return
        state["trajectories"].append((path, trajectory))
        # No need to append the Plotly figure to images; it will be rendered directly in the Gradio app.

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
        for client, poses_and_grippers in arm_poses.items():
            pose_plot = self._create_3d_pose_plot(poses_and_grippers.poses)
            state["images"].append((f"arm_poses_client_{client}_3d_plot", pose_plot.to_image(format="png")))

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

        def display_3d_trajectory():
            # Render the first trajectory in the state as a 3D plot
            if state["trajectories"]:
                _, trajectory = state["trajectories"][0]
                return self._create_3d_trajectory_plot(trajectory)
            return Figure()  # Return an empty figure if no trajectory is logged

        with gr.Blocks() as demo:
            gr.Markdown("# Gradio Visualizer")

            # Display images dynamically
            with gr.Row():
                image_gallery = gr.Gallery(label="Images")

            # Display other data as JSON
            with gr.Row():
                state_display = gr.JSON(label="Logged Data")

            # Display 3D trajectory
            with gr.Row():
                trajectory_plot = gr.Plot(label="3D Trajectory")

            # Refresh button to update the data
            gr.Button("Refresh").click(
                fn=lambda: (display_images(), display_state(), display_3d_trajectory()),
                inputs=[],
                outputs=[image_gallery, state_display, trajectory_plot],
            )

        self._app_launched = True
        print("[GradioBackend] Launching Gradio app. Press Ctrl+C to stop.")
        demo.launch()
