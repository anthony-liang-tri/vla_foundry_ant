"""
Tutorial: How to Use the Visualizer Module

This script demonstrates how to use the `visualizer` module for logging images, scalars, 3D points, trajectories,
and rigid transforms. It also shows how to use the methods for logging robot arm poses and model action predictions.

Make sure the `VISUALIZER` environment variable is set to `rerun` or `disabled` before running this script.
"""

import os

import numpy as np
import visualizer as vz
from pydrake.math import RigidTransform, RollPitchYaw
from robot_gym.multiarm_spaces import PosesAndGrippers  # Import PosesAndGrippers from robot_gym.multiarm_spaces

# Initialize the visualizer
backend = os.environ.get("VISUALIZER", "rerun").lower()
vz.init(run_name="tutorial_logging", backend=backend, add_rank_to_run=True)

# 1. Log an image
print("Logging an image...")
image = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image
vz.log_image("images/camera_view", image)

# 2. Log a scalar value
print("Logging a scalar value...")
vz.log_scalar("metrics/loss", 0.123)

# 3. Log 3D points
print("Logging 3D points...")
points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
vz.log_points3d("points/scene", points)

# 4. Log a 3D trajectory
print("Logging a 3D trajectory...")
trajectory = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
vz.log_trajectory("trajectory/robot_path", trajectory)

# 5. Log a rigid transform
print("Logging a rigid transform...")
pose = RigidTransform(
    RollPitchYaw(np.pi / 4, np.pi / 6, np.pi / 3),  # Rotation
    np.array([1.0, 2.0, 3.0]),  # Translation
)
vz.log_rigid_transform("robot/pose", pose, axis_length=0.5)

# 6. Log robot arm poses
print("Logging robot arm poses...")
arm_poses = {
    "client_1": PosesAndGrippers(
        poses={
            "arm_joint_1": RigidTransform(RollPitchYaw(0, 0, 0), np.array([0, 0, 0])),
            "arm_joint_2": RigidTransform(RollPitchYaw(0, 0, np.pi / 2), np.array([1, 0, 0])),
        },
        grippers={"gripper_1": 0.5},
    )
}
vz.log_arm_poses(arm_poses)

# 7. Log model action predictions
print("Logging model action predictions...")
action_predictions = [
    PosesAndGrippers(
        poses={
            "arm_joint_1": RigidTransform(RollPitchYaw(0, 0, 0), np.array([0, 0, 0])),
            "arm_joint_2": RigidTransform(RollPitchYaw(0, 0, np.pi / 2), np.array([1, 0, 0])),
        },
        grippers={"gripper_1": 0.5},
    ),
    PosesAndGrippers(
        poses={
            "arm_joint_1": RigidTransform(RollPitchYaw(0, 0, 0), np.array([0.5, 0.5, 0])),
            "arm_joint_2": RigidTransform(RollPitchYaw(0, 0, np.pi / 2), np.array([1.5, 0.5, 0])),
        },
        grippers={"gripper_1": 0.5},
    ),
]
vz.log_action_predictions(action_predictions)

# Launch the Gradio app if the backend is Gradio
# TODO: move this into the backend...
if backend == "gradio":
    print("Launching Gradio app...")
    vz._STATE.backend.launch()  # Explicitly call the launch method for Gradio

print("Tutorial complete!")
