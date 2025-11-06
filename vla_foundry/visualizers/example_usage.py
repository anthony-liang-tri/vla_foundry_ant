"""
Tutorial: How to Use the Visualizer Module

This script demonstrates how to use the `visualizer` module for logging images, scalars, 3D points, trajectories,
and rigid transforms. It also shows how to use the methods for logging robot arm poses and model action predictions.

Make sure the `VISUALIZER` environment variable is set to `rerun` or `disabled` before running this script.
"""

import os
import random
import time

import numpy as np
import visualizer as vz
from pydrake.math import RigidTransform, RollPitchYaw
from robot_gym.multiarm_spaces import PosesAndGrippers  # Import PosesAndGrippers from robot_gym.multiarm_spaces

# Initialize the visualizer
backend = os.environ.get("VISUALIZER", "rerun").lower()
vz.init(run_name="tutorial_logging", backend=backend, add_rank_to_run=True)

# Loop to log dynamic data
for step in range(10):
    print(f"Step {step + 1}/10")

    # 1. Log a random image
    print("Logging a random image...")
    image = np.random.randint(0, 256, (255, 255, 3), dtype=np.uint8)  # Random image
    vz.log_image("images/step", image)

    # 2. Log a scalar value that changes over time
    print("Logging a scalar value...")
    vz.log_scalar("metrics/loss", random.uniform(0.1, 1.0))

    # 3. Log 3D points with random positions
    print("Logging random 3D points...")
    points = np.random.rand(5, 3) * 10  # Random points in a 10x10x10 cube
    vz.log_points3d("points/step", points)

    # 4. Log a trajectory that changes shape
    print("Logging a dynamic trajectory...")
    trajectory = np.cumsum(np.random.randn(10, 3), axis=0)  # Random walk in 3D
    vz.log_trajectory("trajectory/step", trajectory)

    # 5. Log a rigid transform with random translation and rotation
    print("Logging a random rigid transform...")
    pose = RigidTransform(
        RollPitchYaw(*np.random.uniform(0, np.pi, 3)),  # Random rotation
        np.random.uniform(-5, 5, 3),  # Random translation
    )
    vz.log_rigid_transform("robot/pose_step", pose, axis_length=0.5)

    # 6. Log robot arm poses with random configurations
    print("Logging random robot arm poses...")
    arm_poses = {
        "client": PosesAndGrippers(
            poses={
                "arm_joint_1": RigidTransform(RollPitchYaw(0, 0, 0), np.random.uniform(-1, 1, 3)),
                "arm_joint_2": RigidTransform(RollPitchYaw(0, 0, np.pi / 2), np.random.uniform(-1, 1, 3)),
            },
            grippers={"gripper_1": random.uniform(0, 1)},
        )
    }
    vz.log_arm_poses(arm_poses)

    # 7. Log model action predictions with random data
    print("Logging random model action predictions...")
    action_predictions = [
        PosesAndGrippers(
            poses={
                "arm_joint_1": RigidTransform(RollPitchYaw(0, 0, 0), np.random.uniform(-1, 1, 3)),
                "arm_joint_2": RigidTransform(RollPitchYaw(0, 0, np.pi / 2), np.random.uniform(-1, 1, 3)),
            },
            grippers={"gripper_1": random.uniform(0, 1)},
        )
        for _ in range(2)
    ]
    vz.log_action_predictions(action_predictions)

    # Simulate time delay between steps
    time.sleep(1)

# Launch the Gradio app if the backend is Gradio
# TODO: move this into the backend...
if backend == "gradio":
    print("Launching Gradio app...")
    vz._STATE.backend.launch()  # Explicitly call the launch method for Gradio

print("Tutorial complete!")
