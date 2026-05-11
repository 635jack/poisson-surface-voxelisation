"""
visualization.py
================
Plotly-based 3D visualization helpers for Streamlit.
"""
import numpy as np
import plotly.graph_objects as go
from typing import Optional

# ── Color palette ────────────────────────────────────────────────────────────
FINGER_COLORS = {
    "thumb":  "#8B4513",
    "index":  "#FFD700",
    "middle": "#FFA500",
    "ring":   "#E61E1E",
    "pinky":  "#00BE00",
    "palm":   "#646464",
}

DARK_LAYOUT = dict(
    paper_bgcolor="#0a0a0f",
    font_color="#e0e0e0",
    legend=dict(
        bgcolor="rgba(20, 20, 30, 0.85)",
        bordercolor="#333",
        font=dict(color="#ccc"),
    ),
)

DARK_SCENE = dict(
    bgcolor="#0a0a0f",
    xaxis=dict(backgroundcolor="#0a0a0f", gridcolor="#1a1a2e",
               title="X (m)", color="#888"),
    yaxis=dict(backgroundcolor="#0a0a0f", gridcolor="#1a1a2e",
               title="Y (m)", color="#888"),
    zaxis=dict(backgroundcolor="#0a0a0f", gridcolor="#1a1a2e",
               title="Z (m)", color="#888"),
    aspectmode="data",
)


def make_mesh_trace(mesh, name: str, color: str, opacity: float = 0.5,
                    showlegend: bool = True) -> go.Mesh3d:
    """Create a Plotly Mesh3d trace from a trimesh."""
    v, f = mesh.vertices, mesh.faces
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color, opacity=opacity,
        name=name, showlegend=showlegend,
        flatshading=False,
        lighting=dict(ambient=0.4, diffuse=0.6, specular=0.3, roughness=0.5),
        lightposition=dict(x=100, y=200, z=300),
    )


def make_contact_traces(positions: np.ndarray, normals: np.ndarray,
                        fingers: list, normal_scale: float = 0.018) -> list:
    """Create contact point and normal traces."""
    traces = []

    for finger, color in FINGER_COLORS.items():
        mask = [i for i, f in enumerate(fingers) if f == finger]
        if not mask:
            continue
        pos = positions[mask]
        nrm = normals[mask]

        # Contact points
        traces.append(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode="markers",
            marker=dict(size=7, color=color, symbol="circle",
                        line=dict(width=1, color="white")),
            name=finger,
            legendgroup=finger,
        ))

        # Normal vectors
        for p, n in zip(pos, nrm):
            n_norm = n / (np.linalg.norm(n) + 1e-9)
            e = p + n_norm * normal_scale
            traces.append(go.Scatter3d(
                x=[p[0], e[0], None],
                y=[p[1], e[1], None],
                z=[p[2], e[2], None],
                mode="lines",
                line=dict(color="white", width=2),
                showlegend=False,
            ))

    return traces


def make_voxel_trace(points: np.ndarray, name: str, color: str,
                     opacity: float = 0.3, size: int = 3) -> go.Scatter3d:
    """Create a voxel visualization trace (scatter of cube centers)."""
    if len(points) == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode="markers", name=name)

    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(
            size=size,
            color=color,
            opacity=opacity,
            symbol="square",
        ),
        name=name,
    )


def make_voxel_colored_trace(
    points: np.ndarray,
    values: np.ndarray,
    name: str,
    colorscale: str = "RdBu_r",
    opacity: float = 0.4,
    size: int = 3,
) -> go.Scatter3d:
    """Create a voxel trace colored by scalar values (e.g. SDF)."""
    if len(points) == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode="markers", name=name)

    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(
            size=size,
            color=values,
            colorscale=colorscale,
            opacity=opacity,
            symbol="square",
            colorbar=dict(title="SDF", len=0.5, thickness=15),
        ),
        name=name,
    )


def build_comparison_figure(
    gt_mesh=None,
    recon_mesh=None,
    positions=None,
    normals=None,
    fingers=None,
    gt_opacity: float = 0.3,
    recon_opacity: float = 0.55,
    show_contacts: bool = True,
    title: str = "GT vs Poisson Reconstruction",
) -> go.Figure:
    """Build the main comparison figure with GT mesh, reconstruction, and contacts."""
    traces = []

    if gt_mesh is not None:
        traces.append(make_mesh_trace(
            gt_mesh, "Ground Truth", "#4a90d9", gt_opacity
        ))

    if recon_mesh is not None:
        traces.append(make_mesh_trace(
            recon_mesh, "Poisson Recon", "#ff8c42", recon_opacity
        ))

    if show_contacts and positions is not None and len(positions) > 0:
        traces.extend(make_contact_traces(positions, normals, fingers))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e0e0e0")),
        scene=DARK_SCENE,
        **DARK_LAYOUT,
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_voxel_comparison_figure(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    gt_opacity: float = 0.25,
    pred_opacity: float = 0.4,
    title: str = "Voxelized Comparison (128³, 20cm cube)",
) -> go.Figure:
    """Build a figure comparing voxelized GT and Poisson SDF."""
    traces = []

    if len(gt_points) > 0:
        traces.append(make_voxel_trace(
            gt_points, "GT Voxels", "#4a90d9", gt_opacity, size=2
        ))

    if len(pred_points) > 0:
        traces.append(make_voxel_trace(
            pred_points, "Poisson SDF Voxels", "#ff8c42", pred_opacity, size=2
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e0e0e0")),
        scene=DARK_SCENE,
        **DARK_LAYOUT,
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_sdf_slice_figure(
    sdf_field: np.ndarray,
    axis: int = 2,
    slice_idx: Optional[int] = None,
    title: str = "SDF Cross-Section",
) -> go.Figure:
    """Show a 2D heatmap slice of the SDF field."""
    if slice_idx is None:
        slice_idx = sdf_field.shape[axis] // 2

    if axis == 0:
        slice_2d = sdf_field[slice_idx, :, :]
        x_label, y_label = "Y", "Z"
    elif axis == 1:
        slice_2d = sdf_field[:, slice_idx, :]
        x_label, y_label = "X", "Z"
    else:
        slice_2d = sdf_field[:, :, slice_idx]
        x_label, y_label = "X", "Y"

    fig = go.Figure(data=go.Heatmap(
        z=slice_2d.T,
        colorscale="RdBu_r",
        colorbar=dict(title="SDF Value", len=0.8),
    ))
    fig.update_layout(
        title=dict(text=f"{title} (slice {slice_idx})", font=dict(size=14, color="#e0e0e0")),
        xaxis_title=x_label,
        yaxis_title=y_label,
        **DARK_LAYOUT,
        height=450,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return fig
