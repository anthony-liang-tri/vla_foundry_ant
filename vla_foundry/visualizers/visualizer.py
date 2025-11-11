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
_HAS_WANDB = importlib.util.find_spec("wandb") is not None  # Add this line

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
    #   rerun | wandb (default to disabled if no env variable is set)
    val = (os.environ.get("VISUALIZER") or "").strip().lower()
    if not val:
        return "disabled"  # Default to disabled if no VISUALIZER is set
    if val in {"disabled", "off", "0", "none"}:
        return "disabled"
    if val in {"rerun", "wandb"}:
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
    if name == "wandb":  # Add this block
        if not _HAS_WANDB:
            print("[visualizer] WandB package not available; using disabled.")
            return None
        try:
            from vla_foundry.visualizers.wandb_backend import WandbBackend  # Import only when needed

            register_backend(WandbBackend())
        except ImportError as e:
            print(f"[visualizer] WandB backend import failed: {e}")
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


@ensure_initialized_and_enabled
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
    _STATE.backend.log_image(_prefix(path), image, **kwargs)  # type: ignore[union-attr]


@ensure_initialized_and_enabled
def log_images(path: str, images: Dict[str, np.ndarray], **kwargs) -> None:
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
        log_image(path, image, **kwargs)


@ensure_initialized_and_enabled
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
    _STATE.backend.log_points3d(_prefix(path), points, **kwargs)  # type: ignore[union-attr]


@ensure_initialized_and_enabled
def log_line_strips3d(path: str, line_strips: np.ndarray, **kwargs) -> None:
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
def log_scalar(path: str, value: float, **kwargs) -> None:
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
def log_trajectory(path: str, trajectory_points: np.ndarray) -> None:
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
    log_points3d(f"{path}/waypoints", trajectory_points)
    log_line_strips3d(f"{path}/path", trajectory_points)


@ensure_initialized_and_enabled
def log_rigid_transform(path: str, transform: RigidTransform, **kwargs) -> None:
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
def log_arm_poses(arm_poses: Dict[str, PosesAndGrippers], **kwargs) -> None:
    """
    Log arm poses to the active backend.

    Parameters
    ----------
    arm_poses : Dict[str, PosesAndGrippers]
        A dictionary containing arm pose data.
    """
    _STATE.backend.log_arm_poses(arm_poses, **kwargs)  # type: ignore[union-attr]


@ensure_initialized_and_enabled
def log_action_predictions(predictions: List[PosesAndGrippers], **kwargs) -> None:
    """
    Log action predictions to the active backend.

    Parameters
    ----------
    predictions : List[PosesAndGrippers]
        A list of objects containing action prediction data.
    """
    _STATE.backend.log_action_predictions(predictions, **kwargs)  # type: ignore[union-attr]


@ensure_initialized_and_enabled
def log_multiarm_observation(path: str, observation: MultiarmObservation, **kwargs) -> None:
    """
    Log a MultiarmObservation to the active backend.

    Parameters
    ----------
    path : str
        Base path in the visualization hierarchy.
    observation : MultiarmObservation
        The MultiarmObservation object to log.
    """
    log_arm_poses({f"{path}/robot": observation.robot.actual}, **kwargs)
    for camera_id, image_set in observation.visuo.items():
        if image_set.rgb:
            log_image(f"{path}/cameras/{camera_id}/rgb", image_set.rgb.array, **kwargs)
        if image_set.depth:
            log_image(f"{path}/cameras/{camera_id}/depth", image_set.depth.array, **kwargs)
        if image_set.label:
            log_image(f"{path}/cameras/{camera_id}/label", image_set.label.array, **kwargs)
    if observation.language_instruction:
        log_text(f"{path}/language_instruction", observation.language_instruction, **kwargs)


@ensure_initialized_and_enabled
def log_text(path: str, text: str, **kwargs) -> None:
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
            # Only print if a backend was initialized
            if _STATE.backend_name != "disabled":
                print(f"[visualizer] Shutting down backend: {_STATE.backend_name}")
            _STATE.backend.shutdown()  # type: ignore[union-attr]
    finally:
        _STATE.enabled = False
