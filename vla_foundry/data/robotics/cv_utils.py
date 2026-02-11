"""
CV utilities.

This module provides helper functions for working with computer vision data,
including rescaling of camera intrinsics to match image preprocessing steps.
"""

from typing import Any, Dict, List, Tuple, Union

import matplotlib.cm as cm
import numpy as np


def intrinsics_3x3_to_4(K: np.ndarray) -> np.ndarray:
    """
    Convert camera intrinsics from shape (3,3) or (N,3,3)
    to (fx, fy, cx, cy) with shape (4,) or (N,4).
    """
    K = np.asarray(K)

    if K.shape[-2:] != (3, 3):
        raise ValueError(f"Expected shape (..., 3, 3), got {K.shape}")

    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    return np.stack([fx, fy, cx, cy], axis=-1).astype(float)


def scale_intrinsics_for_resize_and_crop(
    original_intrinsics: np.ndarray,
    original_image_size: Tuple[int, int],
    processed_image_size: Tuple[int, int],
) -> np.ndarray:
    """Scales camera intrinsics to account for resizing and square cropping.

    Parameters
    ----------
    original_intrinsics : np.ndarray
        Camera intrinsics as (fx, fy, cx, cy) or (N, 4).
    original_image_size : Tuple[int, int]
        Original image size as (width, height).
    processed_image_size : Tuple[int, int]
        Processed image size as (width, height).
    """
    intrinsics = np.asarray(original_intrinsics, dtype=float)

    if intrinsics.shape[-1] != 4:
        raise ValueError(f"Expected shape (..., 4), got {intrinsics.shape}")

    fx, fy, cx, cy = intrinsics[..., 0], intrinsics[..., 1], intrinsics[..., 2], intrinsics[..., 3]

    W0, H0 = original_image_size
    W, H = processed_image_size

    scale = max(W / W0, H / H0)
    sx = sy = scale

    cx_offset = (W0 * scale - W) / 2
    cy_offset = (H0 * scale - H) / 2

    fx1 = fx * sx
    fy1 = fy * sy
    cx1 = cx * sx - cx_offset
    cy1 = cy * sy - cy_offset

    return np.stack([fx1, fy1, cx1, cy1], axis=-1)


def transform_points_to_camera_frame(camera_T_base: np.ndarray, base_t_pts: np.ndarray) -> np.ndarray:
    """Transform 3D points from base frame to camera frame.

    Parameters
    ----------
    camera_T_base : np.ndarray
        Homogeneous transformation matrix from base frame to camera frame of shape (4, 4).
    base_t_pts : np.ndarray
        Points in base frame of shape (N, 3).

    Returns
    -------
    Points in camera frame of shape (N, 3).
    """

    base_t_pts_homogeneous = np.concatenate(
        [base_t_pts, np.ones((base_t_pts.shape[0], 1), dtype=base_t_pts.dtype)], axis=1
    )
    camera_t_pts_homogeneous = (camera_T_base @ base_t_pts_homogeneous.T).T
    camera_t_pts = camera_t_pts_homogeneous[:, :-1]
    return camera_t_pts


def draw_circle(img: np.ndarray, center: Tuple[int, int], radius: int, color: Union[int, Tuple[int, int, int]]):
    """
    Draw a circle onto an image.

    Parameters
    ----------
    img : (H, W, 3) or (H, W)
        Numpy array for RGB or depth images respectively.
    center : (x, y)
        Coordinates of the circle's center.
    radius : int
    color: color: Union[int, Tuple[int, int, int]]
    """
    out = np.array(img, copy=True)
    x0, y0 = center
    h, w = out.shape[:2]

    y, x = np.ogrid[:h, :w]
    mask = (x - x0) ** 2 + (y - y0) ** 2 <= radius**2

    out[mask] = color
    return out


def draw_projected_trajectory_if_rgb(
    image: np.ndarray,
    intrinsics: np.ndarray,
    trace_pts_list: List[np.ndarray],
):
    """
    Draw a 3D trajectory as a set of circles projected onto the image, if the image is RGB.

    Parameters
    ----------
    img : (H, W, 3) or (H, W)
        Numpy array for RGB or depth images respectively.
    intrinsics : np.ndarray
        Camera intrinsics containing (fx, fy, cx, cy). This function expects to unpack them as
        `fx, fy, cx, cy = intrinsics`.
    trace_pts_list : List[np.ndarray]
        List of 3D trajectories (N, 3) in the camera frame that will be
        projected onto the 2D image. Each trajectory is plotted independently from each
        other, and can come from differents sources (e.g. right arm and left arm).
    """
    # Only handle RGB for now, and not depth.
    if image.ndim != 3 or image.shape[2] < 3:
        return image
    img_out = image

    # Get camera intrinsics.
    fx, fy, cx, cy = intrinsics

    for trace_pts in trace_pts_list:
        X, Y, Z = trace_pts[:, 0], trace_pts[:, 1], trace_pts[:, 2]
        valid = Z > 0

        Xv, Yv, Zv = X[valid], Y[valid], Z[valid]

        u = fx * (Xv / Zv) + cx
        v = fy * (Yv / Zv) + cy

        # Radius scaling so that a 3D trace_pt farther away appears smaller when projected.
        R_m = 0.01  # Radius of the "real" sphere in meters.
        r_min, r_max = 3, 7  # Min and max radii in the 2D image.
        f_eff = 0.5 * (fx + fy)
        r_px = f_eff * (R_m / Zv)
        r_px = np.clip(r_px, r_min, r_max)

        circle_coords = np.stack([u, v], axis=1)
        circle_coords = np.round(circle_coords).astype(np.int32)

        cmap = cm.get_cmap("viridis")
        num_trace_pts = len(circle_coords)

        # Sort all trace points by depth so that closer points are drawn last.
        order = np.argsort(Zv)[::-1]  # Indices for decreasing Zv.
        circle_coords_sorted = circle_coords[order]
        r_px_sorted = r_px[order]

        for circle_coord, r_p, orig_idx in zip(circle_coords_sorted, r_px_sorted, order, strict=True):
            trace_idx = orig_idx / max(num_trace_pts - 1, 1)
            rgba = cmap(trace_idx)
            rgb = tuple((np.array(rgba[:3]) * 255).astype(np.uint8))
            img_out = draw_circle(img_out, circle_coord, r_p, rgb)

    return img_out


def create_images_with_projected_trace(
    images: Any,
    intrinsics: np.ndarray,
    trace_pts_list_or_dict: Union[List[np.ndarray], Dict[str, List[np.ndarray]]],
    **kwargs,
) -> None:
    """
    Create images with projected traces.

    Parameters
    ----------
    images : Any
        Either a single NumPy array representing an image or a dictionary of images.
    intrinsics : np.ndarray
        Either a single NumPy array of amera intrinsics containing (fx, fy, cx, cy) = intrinsics,
        or a dictionary of intrinsics.
    trace_pts_list_or_dict : List[np.ndarray] or Dict[str, List[np.ndarray]]
        List (or dictionary of lists) of 3D trajectories (N, 3) in the camera frame that will be
        projected onto the 2D image. Each trajectory is plotted independently from each
        other, and can come from differents sources (e.g. right arm and left arm).
    """

    # Project 3D points onto the image. Do it only for RGB images for now.
    if isinstance(images, np.ndarray):
        assert isinstance(trace_pts_list_or_dict, list), "trace_pts must be a list if images is a single image"
        images = draw_projected_trajectory_if_rgb(images, intrinsics, trace_pts_list_or_dict)

    elif isinstance(images, dict):
        for sub_path, image in images.items():
            trace_pts_list = trace_pts_list_or_dict
            if isinstance(trace_pts_list_or_dict, dict) and sub_path in trace_pts_list_or_dict:
                trace_pts_list = trace_pts_list_or_dict[sub_path]
            else:
                assert isinstance(trace_pts_list_or_dict, list), (
                    "trace_pts must be a list if images is a dict and sub_path not in trace_pts"
                )
            intrinsics_for_image = intrinsics
            if isinstance(intrinsics, dict) and sub_path in intrinsics:
                intrinsics_for_image = intrinsics[sub_path]
            images[sub_path] = draw_projected_trajectory_if_rgb(image, intrinsics_for_image, trace_pts_list)
    return images
