"""
poisson.py
==========
Screened Poisson Surface Reconstruction (FFT-based approximation).

Algorithm:
1. Spread oriented normals onto a 3D voxel grid using Gaussian splatting
2. Compute divergence of the resulting vector field
3. Solve Poisson equation via FFT: Δχ = div(V)
4. Extract isosurface with Marching Cubes
"""
import numpy as np
import trimesh
from scipy.fft import fftn, ifftn
from skimage import measure
from typing import Optional, Tuple


def poisson_reconstruct(
    positions: np.ndarray,
    normals: np.ndarray,
    resolution: int = 64,
    padding: float = 0.02,
    sigma_factor: float = 4.0,
) -> Tuple[Optional[trimesh.Trimesh], Optional[np.ndarray], Optional[dict]]:
    """
    FFT-based Screened Poisson reconstruction.

    Args:
        positions: (N, 3) contact positions
        normals: (N, 3) contact normals
        resolution: grid resolution (cubic)
        padding: bounding box padding in meters
        sigma_factor: Gaussian kernel width = extent / (resolution / sigma_factor)

    Returns:
        mesh: reconstructed trimesh (or None)
        implicit_field: the scalar field (resolution³)
        grid_info: dict with bb_min, bb_max, extent for coordinate mapping
    """
    if len(positions) < 3:
        return None, None, None

    # Bounding box with padding
    bb_min = positions.min(axis=0) - padding
    bb_max = positions.max(axis=0) + padding
    extent = bb_max - bb_min

    # 3D grid
    xs = np.linspace(bb_min[0], bb_max[0], resolution)
    ys = np.linspace(bb_min[1], bb_max[1], resolution)
    zs = np.linspace(bb_min[2], bb_max[2], resolution)
    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]

    # Gaussian splatting of normals onto the grid
    Vx = np.zeros((resolution, resolution, resolution))
    Vy = np.zeros_like(Vx)
    Vz = np.zeros_like(Vx)

    sigma = max(extent) / (resolution / sigma_factor)

    for pt, nrm in zip(positions, normals):
        # Grid index of this point
        ix = (pt[0] - bb_min[0]) / (bb_max[0] - bb_min[0]) * (resolution - 1)
        iy = (pt[1] - bb_min[1]) / (bb_max[1] - bb_min[1]) * (resolution - 1)
        iz = (pt[2] - bb_min[2]) / (bb_max[2] - bb_min[2]) * (resolution - 1)

        i0 = int(np.clip(ix, 0, resolution - 2))
        j0 = int(np.clip(iy, 0, resolution - 2))
        k0 = int(np.clip(iz, 0, resolution - 2))

        # Splat into neighborhood
        radius = 3
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                for dk in range(-radius, radius + 1):
                    ni, nj, nk = i0 + di, j0 + dj, k0 + dk
                    if 0 <= ni < resolution and 0 <= nj < resolution and 0 <= nk < resolution:
                        gx = xs[ni] - pt[0]
                        gy = ys[nj] - pt[1]
                        gz = zs[nk] - pt[2]
                        w = np.exp(-(gx**2 + gy**2 + gz**2) / (2 * sigma**2))
                        Vx[ni, nj, nk] += w * nrm[0]
                        Vy[ni, nj, nk] += w * nrm[1]
                        Vz[ni, nj, nk] += w * nrm[2]

    # Divergence of the vector field
    divV = (np.gradient(Vx, dx, axis=0) +
            np.gradient(Vy, dy, axis=1) +
            np.gradient(Vz, dz, axis=2))

    # Solve Poisson equation via FFT
    F = fftn(divV)
    kx = np.fft.fftfreq(resolution, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(resolution, d=dy) * 2 * np.pi
    kz = np.fft.fftfreq(resolution, d=dz) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    denom = KX**2 + KY**2 + KZ**2
    denom[0, 0, 0] = 1.0
    chi = F / denom
    chi[0, 0, 0] = 0.0
    implicit = np.real(ifftn(chi))

    # Compute isovalue: average field value at contact points
    isovals = []
    for pt in positions:
        ix = int(np.clip(
            (pt[0] - bb_min[0]) / (bb_max[0] - bb_min[0]) * (resolution - 1),
            0, resolution - 1
        ))
        iy = int(np.clip(
            (pt[1] - bb_min[1]) / (bb_max[1] - bb_min[1]) * (resolution - 1),
            0, resolution - 1
        ))
        iz = int(np.clip(
            (pt[2] - bb_min[2]) / (bb_max[2] - bb_min[2]) * (resolution - 1),
            0, resolution - 1
        ))
        isovals.append(implicit[ix, iy, iz])
    isovalue = float(np.mean(isovals))

    # Marching Cubes
    try:
        verts_idx, faces, _, _ = measure.marching_cubes(implicit, level=isovalue)
    except Exception:
        return None, implicit, {"bb_min": bb_min, "bb_max": bb_max, "extent": extent}

    # Convert voxel indices to world coordinates
    verts_world = np.column_stack([
        bb_min[0] + verts_idx[:, 0] / (resolution - 1) * extent[0],
        bb_min[1] + verts_idx[:, 1] / (resolution - 1) * extent[1],
        bb_min[2] + verts_idx[:, 2] / (resolution - 1) * extent[2],
    ])

    try:
        recon_mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=True)
    except Exception:
        return None, implicit, {"bb_min": bb_min, "bb_max": bb_max, "extent": extent}

    grid_info = {
        "bb_min": bb_min,
        "bb_max": bb_max,
        "extent": extent,
        "isovalue": isovalue,
    }

    return recon_mesh, implicit, grid_info
