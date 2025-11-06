"""
This file provides convenience decorators and functions to log information to rerun.io

The intention is to eventually refactor these functionalities into a workflow
that enables visualization in wandb as well as rerun.io. It is added here as-is to
provide immediate value for debugging and visualization.

"""

import functools
import subprocess
from collections.abc import Iterable
from typing import Callable, Dict

import numpy as np
import rerun as rr
from pydrake.math import RigidTransform
from rerun import Transform3D
from rerun.datatypes import Quaternion


def disable_rerun_analytics():
    try:
        subprocess.run(["rerun", "analytics", "disable"], check=True, capture_output=True, text=True)
        print("Rerun analytics disabled.")
    except FileNotFoundError:
        print("Warning: rerun CLI not found. Make sure rerun-sdk is installed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to disable rerun analytics: {e.stderr or e}")


# Utility function to initialize the rerun server
def initialize_rerun_server():
    """
    Initializes the rerun server if it is not already enabled.
    """
    if not rr.is_enabled():
        disable_rerun_analytics()
        rr.init("vla_foundry_logging", spawn=True)


def log_rigid_transform(entity: str, X_AB: RigidTransform) -> None:
    """
    Visualise a Drake pose in Rerun.

    Parameters
    ----------
    entity : str
        Path of the entity in the Rerun hierarchy (e.g. "world/robot_base").
    X_AB : RigidTransform
        Pose of frame B expressed in frame A (Drake notation X_AB).

    Generated with ChatGPT
    """
    # Translation vector (m) expressed in parent frame A
    t = X_AB.translation()  # shape (3,)

    # Quaternion conversion: Drake (w,x,y,z) → Rerun (x,y,z,w)
    q_wxyz = X_AB.rotation().ToQuaternion().wxyz()  # ndarray (4,)
    q_xyzw = np.roll(q_wxyz, -1)  # shift order

    # Send to Rerun
    rr.log(
        entity,
        rr.Transform3D(
            translation=t,
            quaternion=q_xyzw,
            axis_length=0.25,  # 25 cm axes so the pose is visible
            clear=False,  # keep axes while you stream updates
        ),
    )


def log_images(images: Dict[str, np.ndarray]) -> None:
    """
    Log multiple images to rerun.io.

    Parameters:
    - images: A dictionary where keys are image paths and values are NumPy arrays representing the images.
    """
    initialize_rerun_server()
    for path, image in images.items():
        rr.log(f"images/{path}", rr.Image(image))


def rerun_log_images(func):
    """
    A decorator to log images and their associated metadata using rerun.io.
    Use this decorator on functions that return a dictionary of images.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize the rerun server
        initialize_rerun_server()

        # Extract the class instance (self) and call the original function
        result = func(*args, **kwargs)

        # Log each image and its metadata
        for image_name, image_data in result.items():
            rr.log(f"images/{image_name}", rr.Image(image_data))

        return result

    return wrapper


def rerun_log_arm_poses(func: Callable):
    """
    A decorator to visualize robot arm poses.
    Use this decorator on functions that return action dictionaries with poses
    of type PosesAndGrippers.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize the rerun server
        initialize_rerun_server()

        # Call the original step_batch
        actions_dict = func(*args, **kwargs)

        # Log each client's poses
        for client_id, poses_and_grippers in actions_dict.items():
            if not poses_and_grippers or not hasattr(poses_and_grippers, "poses"):
                continue

            timestamp = getattr(poses_and_grippers, "timestamp_data", None)
            if timestamp is not None:
                rr.set_time_seconds("time", timestamp)

            for model_name, transform in poses_and_grippers.poses.items():
                # Inside the decorator loop
                rotation_quat = transform.rotation().ToQuaternion()
                print("logging ", transform.translation())
                rr.log(
                    f"clients/{client_id}/models/{model_name}/pose",
                    Transform3D(
                        translation=transform.translation(),
                        rotation=Quaternion(
                            xyzw=[rotation_quat.x(), rotation_quat.y(), rotation_quat.z(), rotation_quat.w()]
                        ),
                        axis_length=0.25,
                    ),
                )

            if hasattr(poses_and_grippers, "grippers") and poses_and_grippers.grippers:
                for gripper_name, value in poses_and_grippers.grippers.items():
                    rr.log(f"clients/{client_id}/grippers/{gripper_name}/grip", rr.Scalars(value))

        return actions_dict

    return wrapper


def log_model_action_predictions(func):
    """
    A decorator to visualize trajectories.
    Use this decorator on functions that return lists of PosesAndGrippers.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        initialize_rerun_server()

        results = func(*args, **kwargs)
        if not results:
            return results
        if not isinstance(results, Iterable):
            results = [results]

        # 1) Decide which model keys to track from the first valid step
        first_poses = None
        for r in results:
            p = getattr(r, "poses", None)
            if p:
                first_poses = p
                break
        if not first_poses:
            return results

        # If you only want grippers, filter here:
        model_keys = [k for k in first_poses if "gripper" in k.lower()] or list(first_poses.keys())

        # 2) Accumulate per-model trajectories in timestep order
        traj = {k: [] for k in model_keys}
        for r in results:
            poses = getattr(r, "poses", None)
            if not poses:
                continue
            for k in model_keys:
                if k in poses:
                    xyz = np.asarray(poses[k].translation(), dtype=float).reshape(3)
                    traj[k].append(xyz)

        # 3) Log one strip per model (no cross-linking between models)
        for k, pts in traj.items():
            if len(pts) >= 2:
                arr = np.vstack(pts)
                rr.log(f"predictions/{k}/trajectory", rr.LineStrips3D([arr]))
                rr.log(f"predictions/{k}/waypoints", rr.Points3D(arr))
            elif pts:
                rr.log(f"predictions/{k}/waypoints", rr.Points3D(np.vstack(pts)))

        return results

    return wrapper
