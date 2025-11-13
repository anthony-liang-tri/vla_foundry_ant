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
import importlib.util  # Add this import
import os
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Optional

import numpy as np
from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers

# Optional imports (gate behind backend)
_HAS_RERUN = importlib.util.find_spec("rerun") is not None

# Optional Drake import for RigidTransform convenience
try:
    from pydrake.math import RigidTransform  # type: ignore

    _HAS_DRAKE = True
except Exception:
    _HAS_DRAKE = False

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
    #   rerun (default to disabled if no env variable is set)
    val = (os.environ.get("VISUALIZER") or "").strip().lower()
    if not val:
        return "disabled"  # Default to disabled if no VISUALIZER is set
    if val in {"disabled", "off", "0", "none"}:
        return "disabled"
    if val in {"rerun"}:
        return val
    # Auto
    return "disabled"


def _get_backend(name: str) -> Optional[Backend]:
    if name == "disabled":
        return None
    if name == "rerun":
        if not _HAS_RERUN:
            print("[visualizer] Rerun package not available; using disabled.")
            return None
        try:
            from vla_foundry.visualizers.rerun_backend import RerunBackend  # Import only when needed

            register_backend(RerunBackend())
        except ImportError as e:
            print(f"[visualizer] Rerun backend import failed: {e}")
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

    - backend: If None, automatically detect the backend using the VISUALIZER environment variable.
    - VISUALIZER=disabled disables everything.
    - run_name: Default from VISUALIZER_RUN_NAME or basename of CWD.
    - add_rank_to_run: Append "-r{rank}" to run name.
    """
    if _STATE.initialized:
        return

    # Automatically detect the backend if not explicitly provided
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


def ensure_initialized_and_enabled(func):
    """Decorator to ensure the visualizer is initialized and enabled before logging."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _STATE.initialized:
            init()
        if not enabled():
            return  # Bypass if disabled
        return func(*args, **kwargs)

    return wrapper


class Visualizer:
    """Base visualizer facade with general-purpose logging methods."""

    @ensure_initialized_and_enabled
    def log_image(self, path: str, image: np.ndarray, **kwargs) -> None:
        """
        Log an image to the active backend.

        Parameters
        ----------
        path : str
            Path in the visualization hierarchy (e.g., "images/cam0").
        image : np.ndarray
            Image data as a NumPy array.
        """
        _STATE.backend.log_image(_prefix(path), image, **kwargs)  # type: ignore[union-attr]

    @ensure_initialized_and_enabled
    def log_images(self, path: str, images: Dict[str, np.ndarray], **kwargs) -> None:
        """
        Log multiple images to the active backend.

        Parameters
        ----------
        path : str
            Base path in the visualization hierarchy.
        images : Dict[str, np.ndarray]
            A dictionary where keys are image paths and values are NumPy arrays representing the images.
        """
        for path, image in images.items():
            self.log_image(path, image, **kwargs)

    @ensure_initialized_and_enabled
    def log_scalar(self, path: str, value: float, **kwargs) -> None:
        """
        Log a scalar value to the active backend.

        Parameters
        ----------
        path : str
            Path in the visualization hierarchy (e.g., "metrics/loss").
        value : float
            Scalar value to log.
        """
        _STATE.backend.log_scalar(_prefix(path), value, **kwargs)  # type: ignore[union-attr]

    @ensure_initialized_and_enabled
    def log_points3d(self, path: str, points: np.ndarray, **kwargs) -> None:
        """
        Log 3D points to the active backend.

        Parameters
        ----------
        path : str
            Path in the visualization hierarchy (e.g., "points/scene").
        points : np.ndarray
            3D points as a NumPy array of shape (N, 3).
        """
        _STATE.backend.log_points3d(_prefix(path), points, **kwargs)  # type: ignore[union-attr]

    @ensure_initialized_and_enabled
    def log_trajectory(self, path: str, trajectory_points: np.ndarray) -> None:
        """
        Log a trajectory as waypoints and a path in the visualization hierarchy.

        Parameters
        ----------
        path : str
            Base path in the visualization hierarchy (e.g., "robot/trajectory").
        trajectory_points : np.ndarray
            Array of shape (N, 3) representing the trajectory points.
        """
        if trajectory_points.ndim != 2 or trajectory_points.shape[1] != 3:
            raise ValueError("trajectory_points must be a (N, 3) array")
        self.log_points3d(f"{path}/waypoints", trajectory_points)
        self.log_line_strips3d(f"{path}/path", trajectory_points)

    @ensure_initialized_and_enabled
    def log_line_strips3d(self, path: str, line_strips: np.ndarray, **kwargs) -> None:
        """
        Log 3D line strips to the active backend.

        Parameters
        ----------
        path : str
            Path in the visualization hierarchy (e.g., "lines/trajectory").
        line_strips : np.ndarray
            Line strips as a NumPy array of shape (N, 3).
        """
        _STATE.backend.log_line_strips3d(_prefix(path), line_strips, **kwargs)  # type: ignore[union-attr]

    @ensure_initialized_and_enabled
    def log_text(self, path: str, text: str, **kwargs) -> None:
        """
        Log a text value to the active backend.

        Parameters
        ----------
        path : str
            Path in the visualization hierarchy (e.g., "language/instruction").
        text : str
            Text value to log.
        """
        _STATE.backend.log_text(_prefix(path), text, **kwargs)  # type: ignore[union-attr]

    def flush(self) -> None:
        if not enabled():
            return
        _STATE.backend.flush()  # type: ignore[union-attr]

    def shutdown(self) -> None:
        """
        Perform cleanup during shutdown. Ensure the backend is active before shutting down.
        """
        if not enabled():
            return
        try:
            if _STATE.backend is not None:
                # Only print if a backend was initialized
                if _STATE.backend_name != "disabled":
                    print(f"[visualizer] Shutting down backend: {_STATE.backend_name}")
                _STATE.backend.shutdown()  # type: ignore[union-attr]
        finally:
            _STATE.enabled = False


# Create a new DrakeVisualizer class for robot_gym and drake-specific methods
class DrakeVisualizer(Visualizer):
    """Extended visualizer with robot_gym and drake-specific logging methods."""

    @ensure_initialized_and_enabled
    def log_rigid_transform(self, path: str, transform: RigidTransform, **kwargs) -> None:
        """
        Log a rigid transform to the active backend.

        Parameters
        ----------
        path : str
            Path in the visualization hierarchy.
        transform : RigidTransform
            Rigid transform object.
        """
        _STATE.backend.log_rigid_transform(_prefix(path), transform, **kwargs)  # type: ignore[union-attr]

    @ensure_initialized_and_enabled
    def log_robot_gym_poses_and_grippers(self, path: str, poses_and_grippers: PosesAndGrippers, **kwargs) -> None:
        """
        Log poses and grippers (robot-gym-specific) to the active backend.

        Parameters
        ----------
        path : str
            Base path in the visualization hierarchy.
        poses_and_grippers : PosesAndGrippers
            Object containing poses and gripper data.
        """
        if not poses_and_grippers or not hasattr(poses_and_grippers, "poses"):
            return

        for model_name, transform in poses_and_grippers.poses.items():
            self.log_rigid_transform(
                f"{path}/models/{model_name}/pose",
                transform,
                axis_length=0.25,
                **kwargs,
            )

        if hasattr(poses_and_grippers, "grippers") and poses_and_grippers.grippers:
            for gripper_name, value in poses_and_grippers.grippers.items():
                self.log_scalar(f"{path}/grippers/{gripper_name}/grip", value, **kwargs)

    @ensure_initialized_and_enabled
    def log_robot_gym_action_predictions(self, path: str, predictions: List[PosesAndGrippers], **kwargs) -> None:
        """
        Log action predictions (robot-gym-specific) to the active backend.

        Parameters
        ----------
        path : str
            Base path in the visualization hierarchy.
        predictions : List[PosesAndGrippers]
            A list of objects containing action prediction data.
        """
        traj = {}
        for step in predictions:
            poses = getattr(step, "poses", {})
            for model_name, transform in poses.items():
                if model_name not in traj:
                    traj[model_name] = []
                traj[model_name].append(transform.translation())

        for model_name, points in traj.items():
            points_array = np.array(points)
            if len(points) >= 2:
                self.log_trajectory(f"{path}/{model_name}", points_array, **kwargs)
            elif points:
                self.log_points3d(f"{path}/{model_name}/waypoints", points_array, **kwargs)

    @ensure_initialized_and_enabled
    def log_robot_gym_multiarm_observation(self, path: str, observation: MultiarmObservation, **kwargs) -> None:
        """
        Log a MultiarmObservation (robot-gym-specific) to the active backend.

        Parameters
        ----------
        path : str
            Base path in the visualization hierarchy.
        observation : MultiarmObservation
            The MultiarmObservation object to log.
        """
        self.log_robot_gym_poses_and_grippers(f"{path}/robot", observation.robot.actual, **kwargs)
        for camera_id, image_set in observation.visuo.items():
            if image_set.rgb:
                self.log_image(f"{path}/cameras/{camera_id}/rgb", image_set.rgb.array, **kwargs)
            if image_set.depth:
                self.log_image(f"{path}/cameras/{camera_id}/depth", image_set.depth.array, **kwargs)
            if image_set.label:
                self.log_image(f"{path}/cameras/{camera_id}/label", image_set.label.array, **kwargs)
        if observation.language_instruction:
            self.log_text(f"{path}/language_instruction", observation.language_instruction, **kwargs)


# Default instances for facade-like usage
_default_visualizer = Visualizer()

# Lazy initialization for DrakeVisualizer
_drake_visualizer_instance = None


def _get_drake_visualizer() -> DrakeVisualizer:
    """
    Lazily initialize and return the DrakeVisualizer instance.
    """
    global _drake_visualizer_instance
    if _drake_visualizer_instance is None:
        _drake_visualizer_instance = DrakeVisualizer()
    return _drake_visualizer_instance


# Expose module-level functions for the default visualizer
log_image = _default_visualizer.log_image
log_images = _default_visualizer.log_images
log_scalar = _default_visualizer.log_scalar
log_points3d = _default_visualizer.log_points3d
log_trajectory = _default_visualizer.log_trajectory
log_line_strips3d = _default_visualizer.log_line_strips3d
log_text = _default_visualizer.log_text
flush = _default_visualizer.flush
shutdown = _default_visualizer.shutdown


# Expose Drake-specific functions with lazy initialization
def log_rigid_transform(path: str, transform: RigidTransform, **kwargs) -> None:
    _get_drake_visualizer().log_rigid_transform(path, transform, **kwargs)


def log_robot_gym_poses_and_grippers(path: str, poses_and_grippers: PosesAndGrippers, **kwargs) -> None:
    _get_drake_visualizer().log_robot_gym_poses_and_grippers(path, poses_and_grippers, **kwargs)


def log_robot_gym_action_predictions(path: str, predictions: List[PosesAndGrippers], **kwargs) -> None:
    _get_drake_visualizer().log_robot_gym_action_predictions(path, predictions, **kwargs)


def log_robot_gym_multiarm_observation(path: str, observation: MultiarmObservation, **kwargs) -> None:
    _get_drake_visualizer().log_robot_gym_multiarm_observation(path, observation, **kwargs)
