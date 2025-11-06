# visualizer.py
"""
Lightweight visualization facade with a pluggable backend.

Goals:
- One-time init(), then visualizer.log(...)
- Backend chosen by env (VISUALIZER=rerun|disabled), default = rerun if available
- Safe to import anywhere; does nothing if disabled or unavailable
- Works across multi-node/multi-process jobs (rank-aware namespacing)
- Designed to later support wandb, gradio, etc. via the backend registry

Current support: rerun.io only (optional dependency)
"""

from __future__ import annotations

import atexit
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

# Optional imports (gate behind backend)
try:
    import rerun as rr  # type: ignore
    from rerun import Transform3D  # type: ignore
    from rerun.datatypes import Quaternion  # type: ignore

    _HAS_RERUN = True
except Exception:
    _HAS_RERUN = False

# Optional Drake import for RigidTransform convenience
try:
    from pydrake.math import RigidTransform  # type: ignore

    _HAS_DRAKE = True
except Exception:
    _HAS_DRAKE = False

from vla_foundry.visualizers.gradio_backend import GradioBackend  # Import the GradioBackend class
from vla_foundry.visualizers.rerun_backend import RerunBackend  # Import the RerunBackend class

# ---------------------------
# Backend interface + registry
# ---------------------------


class Backend:
    """Abstract backend API. Keep this minimal."""

    name: str = "abstract"

    def init(self, run_name: str, **kwargs) -> None:  # noqa: D401
        """Initialize the session/run."""
        raise NotImplementedError

    def log(self, path: str, value: Any, **kwargs) -> None:
        """Log a value at a hierarchical path."""
        raise NotImplementedError

    def flush(self) -> None:
        pass  # Optional

    def shutdown(self) -> None:
        pass  # Optional


_BACKENDS: Dict[str, Backend] = {}


def register_backend(backend: Backend) -> None:
    _BACKENDS[backend.name] = backend


# Register rerun if importable
if _HAS_RERUN:
    register_backend(RerunBackend())

# Register gradio if importable
try:
    register_backend(GradioBackend())
except Exception as e:
    print(f"[visualizer] Failed to register Gradio backend: {e}")

# ---------------------------
# Global state / facade
# ---------------------------


@dataclass
class _State:
    backend_name: str = "auto"  # "rerun" | "disabled" | ...
    backend: Optional[Backend] = None
    run_name: str = ""
    enabled: bool = False
    initialized: bool = False
    rank_prefix: str = ""  # used to namespace multi-process logs


_STATE = _State()


def _detect_rank_prefix() -> str:
    # Common envs: torchrun/SLURM/MPI. Keep it simple.
    rank = (
        os.environ.get("RANK")
        or os.environ.get("SLURM_PROCID")
        or os.environ.get("LOCAL_RANK")
        or os.environ.get("OMPI_COMM_WORLD_RANK")
    )
    if rank is None:
        return ""
    return f"r{rank}"


def _choose_backend_from_env() -> str:
    # VISUALIZER values:
    #   disabled|off|0 -> disabled
    #   rerun|gradio (default to rerun if installed)
    val = (os.environ.get("VISUALIZER") or "").strip().lower()
    if val in {"disabled", "off", "0", "none"}:
        return "disabled"
    if val in {"rerun", "gradio"}:
        return val
    # Auto
    return "rerun" if "rerun" in _BACKENDS else "gradio" if "gradio" in _BACKENDS else "disabled"


def _get_backend(name: str) -> Optional[Backend]:
    if name == "disabled":
        return None
    b = _BACKENDS.get(name)
    if b is None:
        print(f"[visualizer] Backend '{name}' not registered; using disabled.")
    return b


def init(
    run_name: Optional[str] = None,
    *,
    backend: Optional[str] = None,
    spawn: bool = True,
    allow_disabled: bool = True,
    add_rank_to_run: bool = False,
) -> None:
    """
    Initialize the visualizer once per process.

    - backend: override backend selection (e.g., "rerun"). If None, use VISUALIZER env.
    - VISUALIZER=disabled disables everything.
    - run_name: default from VISUALIZER_RUN_NAME or basename of CWD.
    - add_rank_to_run: append "-r{rank}" to run name.
    """
    if _STATE.initialized:
        return

    chosen = backend or _choose_backend_from_env()
    be = _get_backend(chosen)

    # Derive run name
    default_run = os.environ.get("VISUALIZER_RUN_NAME")
    if not default_run:
        default_run = os.path.basename(os.getcwd())
    rn = run_name or default_run or "run"

    rp = _detect_rank_prefix()
    if add_rank_to_run and rp:
        rn = f"{rn}-{rp}"

    _STATE.backend_name = chosen
    _STATE.backend = be
    _STATE.run_name = rn
    _STATE.rank_prefix = rp

    if be is None:
        _STATE.enabled = allow_disabled is False  # typically False
        _STATE.initialized = True
        print("[visualizer] disabled (no backend).")
        return

    # Backend init
    be.init(rn, spawn=spawn)
    _STATE.enabled = True
    _STATE.initialized = True

    # Ensure we shut down cleanly
    atexit.register(shutdown)


def enabled() -> bool:
    return _STATE.enabled and _STATE.backend is not None


def _prefix(path: str) -> str:
    # Namespace logs by rank to support multi-node/multi-process debugging
    if _STATE.rank_prefix:
        return f"{_STATE.rank_prefix}/{path}"
    return path


def log_image(path: str, image: np.ndarray, **kwargs) -> None:
    """
    Log an image to the active backend.

    Parameters
    ----------
    path : str
        Path in the visualization hierarchy (e.g., "images/cam0").
    image : np.ndarray
        Image data as a NumPy array.
    """
    if not _STATE.initialized:
        init()
    if not enabled():
        return
    assert _STATE.backend is not None
    _STATE.backend.log_image(_prefix(path), image, **kwargs)


def log_images(path: str, images: Dict[str, np.ndarray], **kwargs) -> None:
    """
    Log multiple images to the active backend.

    Parameters
    ----------
    images : Dict[str, np.ndarray]
        A dictionary where keys are image paths and values are NumPy arrays representing the images.
    """
    for path, image in images.items():
        log_image(path, image, **kwargs)


def log_points3d(path: str, points: np.ndarray, **kwargs) -> None:
    """
    Log 3D points to the active backend.

    Parameters
    ----------
    path : str
        Path in the visualization hierarchy (e.g., "points/scene").
    points : np.ndarray
        3D points as a NumPy array of shape (N, 3).
    """
    if not _STATE.initialized:
        init()
    if not enabled():
        return
    assert _STATE.backend is not None
    _STATE.backend.log_points3d(_prefix(path), points, **kwargs)


def log_line_strips3d(path: str, line_strips: np.ndarray, **kwargs) -> None:
    """
    Log 3D line strips to the active backend.

    Parameters
    ----------
    path : str
        Path in the rerun hierarchy (e.g., "lines/trajectory").
    line_strips : np.ndarray
        Line strips as a NumPy array of shape (N, 3).
    """
    if not _STATE.initialized:
        init()
    if not enabled():
        return
    assert _STATE.backend is not None
    _STATE.backend.log_line_strips3d(_prefix(path), line_strips, **kwargs)


def log_scalar(path: str, value: float, **kwargs) -> None:
    """
    Log a scalar value to the active backend.

    Parameters
    ----------
    path : str
        Path in the rerun hierarchy (e.g., "metrics/loss").
    value : float
        Scalar value to log.
    """
    if not _STATE.initialized:
        init()
    if not enabled():
        return
    assert _STATE.backend is not None
    _STATE.backend.log_scalar(_prefix(path), value, **kwargs)


def log_trajectory(path: str, trajectory_points: np.ndarray) -> None:
    """
    Log a trajectory as waypoints and a path in rerun.

    Parameters
    ----------
    path : str
        Base path in the rerun hierarchy (e.g., "robot/trajectory").
    trajectory_points : np.ndarray
        Array of shape (N, 3) representing the trajectory points.
    """
    if trajectory_points.ndim != 2 or trajectory_points.shape[1] != 3:
        raise ValueError("trajectory_points must be a (N, 3) array")

    # Log waypoints
    log_points3d(f"{path}/waypoints", trajectory_points)

    # Log path
    log_line_strips3d(f"{path}/path", trajectory_points)


def flush() -> None:
    if not enabled():
        return
    _STATE.backend.flush()  # type: ignore[union-attr]


def shutdown() -> None:
    """
    Perform cleanup during shutdown. Ensure the backend is active before shutting down.
    """
    if not enabled():
        return
    try:
        if _STATE.backend is not None:
            print(f"[visualizer] Shutting down backend: {_STATE.backend_name}")
            _STATE.backend.shutdown()  # type: ignore[union-attr]
    finally:
        _STATE.enabled = False


# ---------------------------
# Convenience helpers (optional)
# ---------------------------


def log_rigid_transform(path: str, X_AB: "RigidTransform", *, axis_length: float = 0.25) -> None:
    """
    Log a Drake pose as a Transform3D in rerun.

    Parameters
    ----------
    path : str
        Path in the rerun hierarchy (e.g., "world/robot_base").
    X_AB : RigidTransform
        Pose of frame B expressed in frame A (Drake notation X_AB).
    axis_length : float, optional
        Length of the axes to visualize, by default 0.25.
    """
    if not _HAS_DRAKE:
        raise RuntimeError("Drake not available; cannot log RigidTransform")

    # Translation vector (m) expressed in parent frame A
    t = np.asarray(X_AB.translation(), dtype=float).reshape(3)

    # Quaternion conversion: Drake (w,x,y,z) → Rerun (x,y,z,w)
    q_wxyz = X_AB.rotation().ToQuaternion().wxyz()  # (w, x, y, z)
    q_xyzw = np.roll(q_wxyz, -1)  # → (x, y, z, w)

    # Log the transform
    rr.log(
        path,
        Transform3D(
            translation=t,
            rotation=Quaternion(xyzw=q_xyzw),
            axis_length=axis_length,
        ),
    )


def log_arm_poses(actions_dict: Dict[str, Any]) -> None:
    """
    Visualize robot arm poses.

    Parameters
    ----------
    actions_dict : Dict[str, Any]
        A dictionary where keys are client IDs and values are objects containing
        `poses` (RigidTransform mappings), `grippers` (optional), and `timestamp_data` (optional).
    """
    if not _STATE.initialized:
        init()

    for client_id, poses_and_grippers in actions_dict.items():
        if not poses_and_grippers or not hasattr(poses_and_grippers, "poses"):
            continue

        # Set time to wallclock time
        timestamp = getattr(poses_and_grippers, "timestamp_data", None)
        if timestamp is not None:
            rr.set_time_seconds("time", timestamp)

        for model_name, transform in poses_and_grippers.poses.items():
            # Ensure data compatibility with rerun
            translation = np.asarray(transform.translation(), dtype=float).reshape(3)
            rotation_quat = transform.rotation().ToQuaternion()
            quaternion = np.roll(
                np.array([rotation_quat.w(), rotation_quat.x(), rotation_quat.y(), rotation_quat.z()], dtype=float),
                -1,  # Convert (w, x, y, z) → (x, y, z, w)
            )

            rr.log(
                f"clients/{client_id}/models/{model_name}/pose",
                Transform3D(
                    translation=translation,
                    rotation=Quaternion(xyzw=quaternion),
                    axis_length=0.25,
                ),
            )

        if hasattr(poses_and_grippers, "grippers") and poses_and_grippers.grippers:
            for gripper_name, value in poses_and_grippers.grippers.items():
                if isinstance(value, (int, float)):
                    rr.log(f"clients/{client_id}/grippers/{gripper_name}/grip", rr.Scalars([value]))


def log_action_predictions(results: Iterable[Any]) -> None:
    """
    Visualize trajectories from model action predictions.

    Parameters
    ----------
    results : Iterable[Any]
        A list of objects containing `poses` (RigidTransform mappings).
    """
    if not _STATE.initialized:
        init()

    if not results:
        return

    # 1) Decide which model keys to track from the first valid step
    first_poses = None
    for r in results:
        p = getattr(r, "poses", None)
        if p:
            first_poses = p
            break
    if not first_poses:
        return

    # If you only want grippers, filter here:
    model_keys = [k for k in first_poses if "gripper" in k.lower()] or list(first_poses.keys())
    traj = {k: [] for k in model_keys}

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
            log_line_strips3d(f"predictions/{k}/trajectory", arr)
            log_points3d(f"predictions/{k}/waypoints", arr)
        elif pts:
            log_points3d(f"predictions/{k}/waypoints", np.vstack(pts))

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
            log_line_strips3d(f"predictions/{k}/trajectory", arr)
            log_points3d(f"predictions/{k}/waypoints", arr)
        elif pts:
            log_points3d(f"predictions/{k}/waypoints", np.vstack(pts))
