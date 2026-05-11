"""
voxelizer.py
============
Voxelization utilities for comparing ground truth meshes with Poisson SDF fields.

The voxel cube is 20×20×20 cm at 128³ resolution.
Pitch = 0.20 / 128 ≈ 1.5625 mm per voxel.
"""
import numpy as np
import trimesh
from typing import Tuple, Optional

# ── Constants ────────────────────────────────────────────────────────────────
VOXEL_CUBE_SIZE_M = 0.20       # 20 cm
VOXEL_RESOLUTION = 128          # 128 × 128 × 128
VOXEL_PITCH = VOXEL_CUBE_SIZE_M / VOXEL_RESOLUTION  # ~1.5625 mm


def voxelize_ground_truth(
    mesh: trimesh.Trimesh,
    cube_size: float = VOXEL_CUBE_SIZE_M,
    resolution: int = VOXEL_RESOLUTION,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Voxelize a ground truth mesh into a fixed 20×20×20 cm cube at 128³.

    The mesh is assumed to be centered at origin. We create a uniform grid
    centered at origin and check occupancy.

    Args:
        mesh: trimesh object (centered at origin)
        cube_size: side length of the voxel cube in meters
        resolution: number of voxels per axis

    Returns:
        occupancy: (128, 128, 128) bool array
        grid_centers: (128, 128, 128, 3) array of voxel center coords
        occupied_points: (M, 3) array of centers of occupied voxels
    """
    half = cube_size / 2.0
    pitch = cube_size / resolution

    # Create grid of voxel centers
    lin = np.linspace(-half + pitch / 2, half - pitch / 2, resolution)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    grid_centers = np.stack([gx, gy, gz], axis=-1)  # (128,128,128,3)

    # Flatten to query points
    query_points = grid_centers.reshape(-1, 3)

    # Check which points are inside the mesh
    try:
        inside = mesh.contains(query_points)
    except Exception:
        # Fallback: use proximity-based check
        closest, _, _ = trimesh.proximity.closest_point(mesh, query_points)
        dists = np.linalg.norm(query_points - closest, axis=1)
        inside = dists < pitch

    occupancy = inside.reshape(resolution, resolution, resolution)
    occupied_points = query_points[inside]

    return occupancy, grid_centers, occupied_points


def voxelize_sdf_field(
    implicit_field: np.ndarray,
    grid_info: dict,
    cube_size: float = VOXEL_CUBE_SIZE_M,
    resolution: int = VOXEL_RESOLUTION,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample an implicit field (from Poisson) onto the fixed 20×20×20 cm cube at 128³
    and threshold it to create a binary voxel grid.

    Args:
        implicit_field: 3D scalar field from Poisson reconstruction
        grid_info: dict with bb_min, bb_max, isovalue
        cube_size: side length of output voxel cube (meters)
        resolution: output voxel resolution per axis
        threshold: if None, uses isovalue from grid_info

    Returns:
        occupancy: (128, 128, 128) bool array (inside = True)
        sdf_resampled: (128, 128, 128) resampled SDF values
        occupied_points: (M, 3) centers of occupied voxels
    """
    from scipy.interpolate import RegularGridInterpolator

    bb_min = grid_info["bb_min"]
    bb_max = grid_info["bb_max"]
    isovalue = grid_info.get("isovalue", 0.0)

    # Source grid coordinates
    src_res = implicit_field.shape[0]
    src_x = np.linspace(bb_min[0], bb_max[0], src_res)
    src_y = np.linspace(bb_min[1], bb_max[1], src_res)
    src_z = np.linspace(bb_min[2], bb_max[2], src_res)

    interpolator = RegularGridInterpolator(
        (src_x, src_y, src_z), implicit_field,
        bounds_error=False, fill_value=isovalue - 1.0  # outside → below isovalue
    )

    # Target grid: 20×20×20 cm centered at origin
    half = cube_size / 2.0
    pitch = cube_size / resolution
    lin = np.linspace(-half + pitch / 2, half - pitch / 2, resolution)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    query_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # Interpolate SDF
    sdf_values = interpolator(query_points)
    sdf_resampled = sdf_values.reshape(resolution, resolution, resolution)

    # Threshold: inside = above isovalue
    actual_threshold = isovalue + threshold
    occupancy = sdf_resampled >= actual_threshold

    grid_centers = np.stack([gx, gy, gz], axis=-1)
    occupied_points = query_points[occupancy.ravel()]

    return occupancy, sdf_resampled, occupied_points


def compute_voxel_metrics(
    gt_occupancy: np.ndarray,
    pred_occupancy: np.ndarray,
) -> dict:
    """
    Compute comparison metrics between two binary voxel grids.

    Returns:
        dict with IoU, precision, recall, dice, n_gt, n_pred
    """
    gt = gt_occupancy.astype(bool)
    pred = pred_occupancy.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    n_gt = gt.sum()
    n_pred = pred.sum()

    iou = float(intersection / max(union, 1))
    precision = float(intersection / max(n_pred, 1))
    recall = float(intersection / max(n_gt, 1))
    dice = float(2 * intersection / max(n_gt + n_pred, 1))

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "n_gt_voxels": int(n_gt),
        "n_pred_voxels": int(n_pred),
        "n_intersection": int(intersection),
        "n_union": int(union),
    }
