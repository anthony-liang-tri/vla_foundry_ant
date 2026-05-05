#!/usr/bin/env python3
"""Standalone Rerun visualizer for Unitree G1 Dex3 datasets stored on S3.

Reads episode tar files directly from S3 using webdataset, decodes images and
lowdim proprioception/action data, and streams everything to a Rerun viewer.

Visualises:
  - Camera images (stereo_head_left/right, left/right wrist at t-1 and t0)
  - 3-D trajectories for actual and commanded EE poses (left & right)
  - Low-dim proprioception & action scalars over time

Usage:
    # Visualize 3 episodes (default)
    uv run --group visualization python examples/visualization/visualize_g1_data.py

    # Custom episode count and dataset path
    uv run --group visualization python examples/visualization/visualize_g1_data.py \
        --dataset_path s3://robotics-cam-data/platform/unitree_g1_dex3/tarfiles/v1/move_block_on_plate/real/teleop \
        --num_episodes 5

    # Use shuffled shards instead of ordered episodes
    uv run --group visualization python examples/visualization/visualize_g1_data.py --use_shards --num_episodes 2
"""

import argparse
import contextlib
import io
import json
import signal
import socket
import tarfile
from collections.abc import Iterator

import numpy as np
import rerun as rr
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation

from vla_foundry.aws.s3_path import S3Path
from vla_foundry.aws.s3_utils import create_s3_client
from vla_foundry.data.robotics.utils import rot_6d_to_matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fields we visualise as 3-D trajectories.  Each entry maps the xyz field
# to a (Rerun path, corresponding rot_6d field) pair.
EE_POSE_FIELDS = {
    "actual_ee_pose_left__xyz": ("proprio/left_ee", "actual_ee_pose_left__rot_6d"),
    "actual_ee_pose_right__xyz": ("proprio/right_ee", "actual_ee_pose_right__rot_6d"),
    "action_ee_pose_left__xyz": ("action/left_ee", "action_ee_pose_left__rot_6d"),
    "action_ee_pose_right__xyz": ("action/right_ee", "action_ee_pose_right__rot_6d"),
}

# Length of the axis arrows drawn at the current EE pose (metres).
_EE_AXIS_LENGTH = 0.05

# Actual EE trajectories use saturated colors; commanded action trajectories
# use lighter, semi-transparent variants so they can be compared at a glance.
TRAJECTORY_COLORS = {
    "proprio/left_ee": [0, 120, 255, 255],  # actual left EE: blue
    "proprio/right_ee": [255, 80, 0, 255],  # actual right EE: orange
    "action/left_ee": [100, 200, 255, 180],  # commanded left EE: light blue
    "action/right_ee": [255, 180, 100, 180],  # commanded right EE: light orange
}
_DEFAULT_TRAJECTORY_COLOR = [200, 200, 200, 255]

# Approximate G1 stereo head camera parameters.
# The G1 stands ~1.27m tall. The stereo head cameras are at the top of the
# head, looking slightly downward.  No calibration is stored in this dataset
# so we use reasonable approximations.
_HEAD_HEIGHT = 1.30  # metres above ground (robot base frame origin)
_HEAD_FORWARD = 0.08  # how far in front of torso centre
_STEREO_BASELINE = 0.06  # half-baseline offset in Y
_HEAD_PITCH_DEG = 25.0  # downward pitch angle
_WORLD_R_OPENCV_CAMERA_BASE = np.array(
    [
        [0, 0, 1],  # world-X from cam-Z
        [-1, 0, 0],  # world-Y from cam-X (negated -> right)
        [0, -1, 0],  # world-Z from cam-Y (negated -> down)
    ],
    dtype=np.float64,
)

# Approximate pinhole intrinsics for the resized 224×224 crops.
# Original stereo head images are 672×376; a ~70° HFOV gives
# fx ≈ 672 / (2·tan 35°) ≈ 480.  After RESIZE_FIT to 224×224 the
# scale factor is 224/672 ≈ 0.333, so fx_resized ≈ 160.
_DEFAULT_IMAGE_SIZE = [224, 224]
_DEFAULT_FOCAL_LENGTH = 160.0

# How far in front of the camera origin to render the image plane in Rerun
_IMAGE_PLANE_DISTANCE = 0.35

# Trajectory waypoints fade from dim/transparent in the past to bright/opaque
# near the end of the logged window. The anchor/current timestep is enlarged.
_WAYPOINT_MIN_BRIGHTNESS = 0.3
_WAYPOINT_MIN_ALPHA = 80
_WAYPOINT_ALPHA_RANGE = 175
_WAYPOINT_RADIUS = 0.003
_ANCHOR_WAYPOINT_RADIUS = 0.008
_CURRENT_POSITION_RADIUS = 0.012

# Orientation arrows use the standard RGB axis convention: X red, Y green, Z blue.
_ORIENTATION_AXIS_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
_ORIENTATION_AXIS_RADIUS = 0.002

_WARNED_MISSING_IMAGE_KEYS: set[str] = set()

# ---------------------------------------------------------------------------
# Tar reading helpers
# ---------------------------------------------------------------------------


def _iter_samples_from_tar(tar_bytes: bytes) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Yield dicts of {extension: bytes} grouped by sample id from a tar archive.

    Samples are yielded in tar insertion order (which matches temporal order
    for episode tars written sequentially during preprocessing).

    Args:
        tar_bytes: Raw bytes of a tar archive containing sample files named
            ``<sample_id>.<extension>``.

    Yields:
        Tuples of ``(sample_id, files)`` where *files* is a dict mapping
        file extension to raw bytes (e.g. ``{"jpg": b"...", "lowdim.npz": b"..."}``).
    """
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
        sample_order: list[str] = []
        samples: dict[str, dict[str, bytes]] = {}
        for member in tf:
            if member.isfile():
                name = member.name
                dot = name.find(".")
                if dot < 0:
                    continue
                sample_id = name[:dot]
                ext = name[dot + 1 :]
                f = tf.extractfile(member)
                if f is None:
                    continue
                if sample_id not in samples:
                    sample_order.append(sample_id)
                samples.setdefault(sample_id, {})[ext] = f.read()

        for sample_id in sample_order:
            yield sample_id, samples[sample_id]


def _decode_image(jpg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes into an HWC uint8 RGB array.

    Args:
        jpg_bytes: Raw JPEG-encoded image bytes.

    Returns:
        Numpy array of shape ``(H, W, 3)`` with dtype ``uint8`` in RGB order.
    """
    img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    return np.asarray(img)


def _decode_lowdim(npz_bytes: bytes) -> dict[str, np.ndarray]:
    """Decode a .lowdim.npz blob.

    Args:
        npz_bytes: Raw bytes of an ``.npz`` archive.

    Returns:
        Dict mapping field names to numpy arrays.
    """
    return dict(np.load(io.BytesIO(npz_bytes)))


def _load_visualization_config(
    dataset_path: str,
    s3_client,
) -> tuple[list[str], list[int], list[int]]:
    """Load visualization camera/image settings from preprocessing config.

    Args:
        dataset_path: S3 dataset root containing the ``shards/`` directory.
        s3_client: boto3 S3 client reused for config download.

    Returns:
        Tuple of ``(camera_names, image_indices, image_size)`` where
        ``image_size`` is ``[width, height]``.
    """
    uri = f"{dataset_path}/shards/preprocessing_config.yaml"
    config_bytes = S3Path(s3_path=uri, s3_client=s3_client).download_to_buffer().getvalue()
    preprocessing_config = yaml.safe_load(config_bytes.decode()) or {}
    if not isinstance(preprocessing_config, dict):
        raise ValueError(f"Expected {uri} to contain a YAML mapping")
    print(f"Loaded preprocessing config: {uri}")

    camera_names = preprocessing_config.get("camera_names")
    if not (isinstance(camera_names, list) and all(isinstance(name, str) for name in camera_names)):
        raise ValueError("Expected preprocessing_config.yaml to contain a string list field: camera_names")

    image_indices = preprocessing_config.get("image_indices")
    if not (isinstance(image_indices, list) and all(isinstance(idx, int) for idx in image_indices)):
        raise ValueError("Expected preprocessing_config.yaml to contain an integer list field: image_indices")

    image_size = preprocessing_config.get("resize_images_size")
    if not (
        isinstance(image_size, list)
        and len(image_size) == 2
        and all(isinstance(dim, int | float) for dim in image_size)
    ):
        image_size = _DEFAULT_IMAGE_SIZE

    return camera_names, image_indices, [int(image_size[0]), int(image_size[1])]


def _camera_intrinsics_for_image_size(image_size: list[int]) -> tuple[float, float, float, int, int]:
    """Scale the approximate 224px focal length for the configured image size."""
    img_w, img_h = image_size
    focal_length = _DEFAULT_FOCAL_LENGTH * (img_w / _DEFAULT_IMAGE_SIZE[0])
    return focal_length, img_w / 2.0, img_h / 2.0, img_w, img_h


# ---------------------------------------------------------------------------
# Rerun logging helpers
# ---------------------------------------------------------------------------


def _head_cam_world_from_cam(y_offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the world-frame pose for a stereo head camera.

    The camera convention is X-right, Y-down, Z-forward (OpenCV).

    Args:
        y_offset: Lateral offset in the world Y axis (positive-left).
            Use ``> 0`` for the left camera and ``< 0`` for the right.

    Returns:
        Tuple of ``(translation, quaternion_xyzw)`` where *translation* is a
        length-3 array and *quaternion_xyzw* is a length-4 array in
        ``[x, y, z, w]`` order.
    """
    # World: RIGHT_HAND_Z_UP → X-forward, Y-left, Z-up.
    # OpenCV camera: X-right, Y-down, Z-forward (optical axis).
    #
    # world_R_cam columns = where each camera axis lands in world:
    #   cam-X (right)   → world -Y  (right in a Y-left frame)
    #   cam-Y (down)    → world -Z  (down in a Z-up frame)
    #   cam-Z (forward) → world +X  (forward)
    translation = np.array([_HEAD_FORWARD, y_offset, _HEAD_HEIGHT])

    # Pitch down: rotate around cam-X (the "right" axis).  A *negative*
    # angle in the OpenCV convention tilts the optical axis from +Z toward
    # +Y, i.e. from world-X toward world-(-Z) = downward.
    pitch = Rotation.from_euler("x", -_HEAD_PITCH_DEG, degrees=True)
    world_R_cam = Rotation.from_matrix(_WORLD_R_OPENCV_CAMERA_BASE) * pitch
    quat_xyzw = world_R_cam.as_quat()  # [x, y, z, w]

    return translation, quat_xyzw


def _head_camera_y_offsets(head_camera_names: list[str]) -> list[tuple[str, float]]:
    """Assign approximate lateral offsets to configured head cameras."""
    if len(head_camera_names) == 1:
        return [(head_camera_names[0], 0.0)]
    offsets = np.linspace(_STEREO_BASELINE, -_STEREO_BASELINE, len(head_camera_names))
    return list(zip(head_camera_names, offsets, strict=True))


def _setup_rerun_blueprint(head_camera_names: list[str], image_size: list[int]) -> None:
    """Configure static Rerun entities: coordinate frame & head camera rigs."""
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    focal_length, cx, cy, img_w, img_h = _camera_intrinsics_for_image_size(image_size)

    # --- Static stereo head camera frustums in the 3-D world ---------------
    for cam_name, y_off in _head_camera_y_offsets(head_camera_names):
        entity = f"world/cameras/{cam_name}"
        t, q = _head_cam_world_from_cam(y_off)
        rr.log(
            entity,
            rr.Transform3D(
                translation=t,
                rotation=rr.Quaternion(xyzw=q),
            ),
            static=True,
        )
        rr.log(
            entity,
            rr.Pinhole(
                focal_length=[focal_length, focal_length],
                principal_point=[cx, cy],
                resolution=[img_w, img_h],
                image_plane_distance=_IMAGE_PLANE_DISTANCE,
            ),
            static=True,
        )


def _build_valid_mask(lowdim: dict[str, np.ndarray]) -> np.ndarray:
    """Build a boolean mask selecting non-padded timesteps from lowdim data.

    Each sample has a fixed-size temporal window (e.g. 49 steps).  Near the
    end of an episode the future slots are zero-padded.  ``past_mask`` covers
    index 0 (the t-1 observation) and ``future_mask`` covers index 1+ (anchor
    and future horizon); OR-ing them gives all non-padded timesteps.

    Args:
        lowdim: Dict of lowdim arrays, may contain ``past_mask`` and/or
            ``future_mask``.

    Returns:
        Boolean array of shape ``(T,)`` where True indicates a valid timestep.
    """
    sample_len = next(iter(lowdim.values())).shape[0]
    masks = [lowdim[k].astype(bool).ravel() for k in ("past_mask", "future_mask") if k in lowdim]
    if not masks:
        return np.ones(sample_len, dtype=bool)
    return np.logical_or.reduce(masks)


def _log_images(
    sample_id: str,
    files: dict[str, bytes],
    camera_names: list[str],
    head_camera_names: list[str],
    image_indices: list[int],
) -> None:
    """Log camera images for the current timestep.

    Args:
        sample_id: Unique identifier for this sample within the episode.
        files: Dict mapping file extensions to raw bytes.
        camera_names: Camera names to log.
        head_camera_names: Subset of camera names to also log in the 3-D world.
        image_indices: Temporal image offsets configured during preprocessing.
    """
    for cam in camera_names:
        for image_index in image_indices:
            suffix = f"t{image_index}"
            key = f"{cam}_{suffix}.jpg"
            if key in files:
                img = _decode_image(files[key])
                rr.log(f"images/{cam}/{suffix}", rr.Image(img))
                if cam in head_camera_names and suffix == "t0":
                    rr.log(f"world/cameras/{cam}", rr.Image(img))
            elif key not in _WARNED_MISSING_IMAGE_KEYS:
                # Avoid printing the same missing-key warning on every sample.
                _WARNED_MISSING_IMAGE_KEYS.add(key)
                print(
                    f"Image key {key!r} is missing in sample {sample_id!r}; "
                    "check the dataset preprocessing image_indices/camera_names."
                )


def _log_ee_trajectory(
    rr_path: str,
    xyz: np.ndarray,
    color: list[int],
    anchor_idx: int,
    lowdim: dict[str, np.ndarray],
    rot_field: str,
) -> None:
    """Log 3-D EE trajectory, waypoints, current position, and orientation.

    Args:
        rr_path: Rerun entity path prefix (e.g. ``"proprio/left_ee"``).
        xyz: Valid (non-padded) positions of shape ``(T_valid, 3)``.
        color: RGBA colour for this trajectory.
        anchor_idx: Index of the current timestep within ``xyz``.
        lowdim: Full lowdim dict (unmasked), used to read rotation data.
        rot_field: Key into ``lowdim`` for the 6D rotation array.
    """
    rr.log(
        f"world/{rr_path}/trajectory",
        rr.LineStrips3D([xyz], colors=[color]),
    )

    n_pts = xyz.shape[0]
    t_frac = np.linspace(0.0, 1.0, n_pts)
    pt_colors = np.zeros((n_pts, 4), dtype=np.uint8)
    base = np.array(color[:3], dtype=np.float32)
    for i in range(n_pts):
        brightness = _WAYPOINT_MIN_BRIGHTNESS + (1.0 - _WAYPOINT_MIN_BRIGHTNESS) * t_frac[i]
        pt_colors[i, :3] = (base * brightness).astype(np.uint8)
        pt_colors[i, 3] = int(_WAYPOINT_MIN_ALPHA + _WAYPOINT_ALPHA_RANGE * t_frac[i])
    radii = _WAYPOINT_RADIUS * np.ones(n_pts)
    radii[anchor_idx] = _ANCHOR_WAYPOINT_RADIUS
    rr.log(
        f"world/{rr_path}/waypoints",
        rr.Points3D(xyz, colors=pt_colors, radii=radii),
    )

    rr.log(
        f"world/{rr_path}/current",
        rr.Points3D(
            xyz[anchor_idx : anchor_idx + 1],
            colors=[color],
            radii=[_CURRENT_POSITION_RADIUS],
        ),
    )

    if rot_field in lowdim and anchor_idx < lowdim[rot_field].shape[0]:
        rot_mat = rot_6d_to_matrix(lowdim[rot_field][anchor_idx])
        origin = xyz[anchor_idx]
        rr.log(
            f"world/{rr_path}/orientation",
            rr.Arrows3D(
                origins=np.tile(origin, (3, 1)),
                vectors=rot_mat * _EE_AXIS_LENGTH,
                colors=_ORIENTATION_AXIS_COLORS,
                radii=_ORIENTATION_AXIS_RADIUS,
            ),
        )


def _log_scalars(lowdim: dict[str, np.ndarray], anchor_idx: int) -> None:
    """Log per-field scalar values at the anchor timestep.

    Args:
        lowdim: Dict of lowdim arrays.
        anchor_idx: Index of the current timestep.
    """
    for field_name, arr in sorted(lowdim.items()):
        if field_name in ("past_mask", "future_mask"):
            continue
        val_at_anchor = np.atleast_1d(arr[anchor_idx])
        if val_at_anchor.shape[0] == 1:
            rr.log(f"scalars/{field_name}", rr.Scalars(float(val_at_anchor[0])))
        else:
            for j, v in enumerate(val_at_anchor):
                rr.log(f"scalars/{field_name}/{j}", rr.Scalars(float(v)))


def _log_sample(
    sample_id: str,
    files: dict[str, bytes],
    time_idx: int,
    camera_names: list[str],
    head_camera_names: list[str],
    image_indices: list[int],
) -> None:
    """Log one sample (one timestep of one episode) to Rerun.

    Decodes images, lowdim proprioception/action data, and metadata from
    the sample files.  3-D EE trajectories are masked using ``past_mask``
    and ``future_mask`` to exclude zero-padded timesteps.

    Args:
        sample_id: Unique identifier for this sample within the episode.
        files: Dict mapping file extensions to raw bytes, as yielded by
            :func:`_iter_samples_from_tar`.
        time_idx: Global sequence index used as the Rerun time axis.
        camera_names: Camera names to log.
        head_camera_names: Subset of camera names to also log in the 3-D world.
        image_indices: Temporal image offsets configured during preprocessing.
    """
    rr.set_time("sample", sequence=time_idx)
    _log_images(sample_id, files, camera_names, head_camera_names, image_indices)

    npz_key = "lowdim.npz"
    if npz_key not in files:
        return
    lowdim = _decode_lowdim(files[npz_key])

    meta_key = "metadata.json"
    metadata = json.loads(files[meta_key]) if meta_key in files else {}
    anchor_idx = metadata.get("anchor_relative_idx", 1)

    lang_key = "language_instructions.json"
    if lang_key in files:
        lang = json.loads(files[lang_key])
        instructions = lang.get("original", [])
        if instructions:
            rr.log("language", rr.TextLog(instructions[0]))

    valid_mask = _build_valid_mask(lowdim)

    for xyz_field, (rr_path, rot_field) in EE_POSE_FIELDS.items():
        if xyz_field not in lowdim:
            continue
        xyz = lowdim[xyz_field][valid_mask]
        if xyz.shape[0] == 0:
            continue
        color = TRAJECTORY_COLORS.get(rr_path, _DEFAULT_TRAJECTORY_COLOR)
        _log_ee_trajectory(rr_path, xyz, color, anchor_idx, lowdim, rot_field)

    _log_scalars(lowdim, anchor_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize Unitree G1 data with Rerun")
    p.add_argument(
        "--dataset_path",
        default="s3://robotics-cam-data/platform/unitree_g1_dex3/tarfiles/v1/move_block_on_plate/real/teleop",
        help="S3 path to the dataset root (contains episodes/ and shards/)",
    )
    p.add_argument(
        "--num_episodes",
        type=int,
        default=3,
        help="Number of episodes to visualise",
    )
    p.add_argument(
        "--use_shards",
        action="store_true",
        help="Read from shuffled shards/ instead of ordered episodes/",
    )
    p.add_argument(
        "--web_port",
        type=int,
        default=9090,
        help="Port for the Rerun web viewer",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.rstrip("/")
    s3_client = create_s3_client()
    camera_names, image_indices, image_size = _load_visualization_config(dataset_path, s3_client)
    head_camera_names = [name for name in camera_names if "head" in name]

    # Initialise Rerun
    rr.init("g1_visualizer")
    server_uri = rr.serve_grpc()
    rr.serve_web_viewer(open_browser=False, web_port=args.web_port, connect_to=server_uri)

    # URL-encode the '+' in the rerun+http:// scheme so browsers parse it correctly.
    encoded_uri = server_uri.replace("+", "%2B")
    # Extract the gRPC port from the server URI (e.g. "rerun+http://127.0.0.1:9876/proxy")
    grpc_port = server_uri.split(":")[-1].split("/")[0]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            host_ip = s.getsockname()[0]
    except OSError:
        host_ip = "<REMOTE_HOST>"
    print(f"Rerun viewer: http://localhost:{args.web_port}/?url={encoded_uri}&renderer=webgl")
    print()
    print("For remote access, run this on your laptop first:")
    print(f"  ssh -L {args.web_port}:localhost:{args.web_port} -L {grpc_port}:localhost:{grpc_port} {host_ip}")

    _setup_rerun_blueprint(head_camera_names, image_size)

    # Decide whether to read episodes or shards
    if args.use_shards:
        manifest_uri = f"{dataset_path}/shards/manifest.jsonl"
        data_prefix = f"{dataset_path}/shards"
    else:
        data_prefix = f"{dataset_path}/episodes"

    print(f"Reading data from {data_prefix} ...")

    if args.use_shards:
        manifest_bytes = S3Path(s3_path=manifest_uri, s3_client=s3_client).download_to_buffer().getvalue()
        shard_entries = [json.loads(line) for line in manifest_bytes.decode().strip().split("\n")]
        tar_names = [f"{e['shard']}.tar" for e in shard_entries[: args.num_episodes]]
    else:
        keys = S3Path(s3_path=data_prefix, s3_client=s3_client).list_objects()
        tar_keys = sorted([k for k in keys if k.endswith(".tar")])
        tar_names = [k.split("/")[-1] for k in tar_keys[: args.num_episodes]]

    print(f"Will visualise {len(tar_names)} {'shards' if args.use_shards else 'episodes'}")

    global_sample_idx = 0
    for tar_idx, tar_name in enumerate(tar_names):
        tar_uri = f"{data_prefix}/{tar_name}"
        label = "shard" if args.use_shards else "episode"
        print(f"  [{tar_idx + 1}/{len(tar_names)}] Downloading {label}: {tar_name} ...")

        tar_bytes = S3Path(s3_path=tar_uri, s3_client=s3_client).download_to_buffer().getvalue()
        sample_count = 0

        for sample_id, files in _iter_samples_from_tar(tar_bytes):
            _log_sample(
                sample_id,
                files,
                time_idx=global_sample_idx,
                camera_names=camera_names,
                head_camera_names=head_camera_names,
                image_indices=image_indices,
            )
            global_sample_idx += 1
            sample_count += 1

        print(f"    Logged {sample_count} samples (total so far: {global_sample_idx})")

    print(f"\nDone — {global_sample_idx} samples sent to Rerun.")
    print("Press Ctrl+C to stop the viewer.")

    with contextlib.suppress(KeyboardInterrupt):
        signal.pause()


if __name__ == "__main__":
    main()
