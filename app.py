#!/usr/bin/env python3
"""
app.py — Mesh Comparison Viewer
================================
Interactive Streamlit application for comparing ground truth 3D meshes
with Screened Poisson Surface Reconstructions.

Features:
  - 3D mesh comparison: GT vs Poisson reconstruction
  - Voxelized comparison: GT voxels vs SDF voxels (128³, 20×20×20 cm)
  - SDF cross-section viewer
  - Quantitative metrics (IoU, Dice, Precision, Recall)

Usage:
    streamlit run app.py -- --data-dir ../grasp-dataset-gen
"""
import os
import sys
import argparse
import streamlit as st
import numpy as np

# ── Parse CLI args (before Streamlit eats them) ──────────────────────────────
def get_data_dir():
    """Extract --data-dir from sys.argv, handling Streamlit's '--' separator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../grasp-dataset-gen",
                        help="Path to grasp-dataset-gen directory")
    # Filter out Streamlit's own args
    args_after_sep = []
    found_sep = False
    for arg in sys.argv[1:]:
        if arg == "--":
            found_sep = True
            continue
        if found_sep:
            args_after_sep.append(arg)
    if args_after_sep:
        parsed, _ = parser.parse_known_args(args_after_sep)
    else:
        # Try the parent directory
        default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "grasp-dataset-gen")
        parsed = argparse.Namespace(data_dir=default)
    return parsed.data_dir


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mesh Comparison Viewer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Dark header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #f0f0f5;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #a0a0b8;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e0e0ff;
        margin: 0;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0.3rem 0 0 0;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .status-badge.success {
        background: rgba(0, 190, 0, 0.15);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.3);
    }
    .status-badge.warning {
        background: rgba(255, 165, 0, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    .status-badge.info {
        background: rgba(74, 144, 217, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.2rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .section-header h3 {
        color: #d0d0e0;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
    }

    /* Info panel */
    .info-panel {
        background: rgba(16, 16, 32, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 0.85rem;
        color: #b0b0c8;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0f0;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Imports (after page config) ──────────────────────────────────────────────
from core.data_loader import discover_dataset, load_trimesh_glb, load_contacts, extract_contact_points
from core.poisson import poisson_reconstruct
from core.voxelizer import (
    voxelize_ground_truth, voxelize_sdf_field, compute_voxel_metrics,
    VOXEL_CUBE_SIZE_M, VOXEL_RESOLUTION
)
from core.visualization import (
    build_comparison_figure, build_voxel_comparison_figure,
    build_sdf_slice_figure
)


# ── Data discovery ───────────────────────────────────────────────────────────
DATA_DIR = get_data_dir()

@st.cache_data
def cached_discover():
    return discover_dataset(DATA_DIR)

dataset_info = cached_discover()

if not dataset_info["objects"]:
    st.error(f"⚠️ No objects found in `{DATA_DIR}`. Make sure --data-dir points to a grasp-dataset-gen repo.")
    st.stop()


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 Mesh Comparison Viewer</h1>
    <p>Ground Truth vs Screened Poisson Reconstruction — Interactive 3D & Voxel Analysis</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Configuration")

    # Object selection
    object_names = list(dataset_info["objects"].keys())
    selected_object = st.selectbox(
        "📦 Object",
        object_names,
        index=0,
        help="Select a 3D object from the dataset"
    )

    # Strategy selection
    strategies = dataset_info["objects"][selected_object]["strategies"]
    if strategies:
        selected_strategy = st.selectbox(
            "🤚 Grasp Strategy",
            strategies,
            index=0,
            help="Select the grasp strategy to use for contact points"
        )
    else:
        selected_strategy = None
        st.warning("No contact data available for this object")

    st.markdown("---")
    st.markdown("### 🔧 Poisson Parameters")

    poisson_resolution = st.slider(
        "Grid Resolution",
        min_value=16, max_value=96, value=48, step=8,
        help="Resolution of the Poisson solver grid (higher = more detail, slower)"
    )

    poisson_padding = st.slider(
        "Bounding Box Padding (mm)",
        min_value=5, max_value=50, value=20, step=5,
        help="Padding around contact points for the reconstruction grid"
    )

    sigma_factor = st.slider(
        "Sigma Factor",
        min_value=2.0, max_value=8.0, value=4.0, step=0.5,
        help="Gaussian kernel spread (lower = tighter, more local)"
    )

    st.markdown("---")
    st.markdown("### 🎨 Display Options")

    gt_opacity = st.slider("GT Mesh Opacity", 0.05, 1.0, 0.30, 0.05)
    recon_opacity = st.slider("Reconstruction Opacity", 0.05, 1.0, 0.55, 0.05)
    show_contacts = st.checkbox("Show Contact Points", value=True)
    voxel_opacity_gt = st.slider("GT Voxel Opacity", 0.05, 1.0, 0.25, 0.05)
    voxel_opacity_pred = st.slider("Poisson Voxel Opacity", 0.05, 1.0, 0.40, 0.05)

    sdf_threshold = st.slider(
        "SDF Threshold Offset",
        min_value=-0.5, max_value=0.5, value=0.0, step=0.01,
        help="Offset from isovalue for voxel thresholding"
    )

    st.markdown("---")
    st.markdown(f"""
    <div class="info-panel">
        <strong>Voxel Grid</strong><br>
        Cube: {VOXEL_CUBE_SIZE_M*100:.0f} × {VOXEL_CUBE_SIZE_M*100:.0f} × {VOXEL_CUBE_SIZE_M*100:.0f} cm<br>
        Resolution: {VOXEL_RESOLUTION}³<br>
        Pitch: {VOXEL_CUBE_SIZE_M/VOXEL_RESOLUTION*1000:.2f} mm/voxel
    </div>
    """, unsafe_allow_html=True)


# ── Load data ────────────────────────────────────────────────────────────────

@st.cache_data
def cached_load_mesh(glb_path):
    return load_trimesh_glb(glb_path)

@st.cache_data
def cached_load_contacts(contacts_path):
    return load_contacts(contacts_path)

@st.cache_data
def cached_poisson(positions_bytes, normals_bytes, resolution, padding, sigma):
    positions = np.frombuffer(positions_bytes, dtype=np.float64).reshape(-1, 3)
    normals = np.frombuffer(normals_bytes, dtype=np.float64).reshape(-1, 3)
    return poisson_reconstruct(positions, normals, resolution, padding, sigma)

@st.cache_data
def cached_voxelize_gt(_mesh_vertices, _mesh_faces):
    """Cache GT voxelization. We pass vertices/faces as hashable inputs."""
    mesh = __import__('trimesh').Trimesh(
        vertices=np.frombuffer(_mesh_vertices, dtype=np.float64).reshape(-1, 3),
        faces=np.frombuffer(_mesh_faces, dtype=np.int64).reshape(-1, 3),
        process=False
    )
    return voxelize_ground_truth(mesh)


# Load the ground truth mesh
glb_path = dataset_info["objects"][selected_object]["glb_path"]
gt_mesh = cached_load_mesh(glb_path)

# Load contacts
positions, normals, fingers = np.zeros((0, 3)), np.zeros((0, 3)), []
contacts_data = None
if selected_strategy:
    contacts_path = os.path.join(
        dataset_info["output_dir"], selected_object,
        f"grasp_{selected_strategy}.json"
    )
    contacts_data = cached_load_contacts(contacts_path)
    positions, normals, fingers = extract_contact_points(contacts_data)


# ── Tab layout ───────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🔍 3D Mesh Comparison",
    "🧊 Voxel Comparison",
    "📊 SDF Analysis",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: 3D Mesh Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("""
    <div class="section-header">
        <h3>🔍 Ground Truth vs Screened Poisson Reconstruction</h3>
    </div>
    """, unsafe_allow_html=True)

    if gt_mesh is None:
        st.error(f"Could not load mesh for '{selected_object}'")
    elif len(positions) < 3:
        st.warning("Need at least 3 contact points for Poisson reconstruction. "
                    "Showing ground truth mesh only.")
        fig = build_comparison_figure(
            gt_mesh=gt_mesh,
            gt_opacity=gt_opacity,
            title=f"{selected_object} — Ground Truth Only",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        with st.spinner("🔄 Running Poisson reconstruction..."):
            recon_mesh, implicit_field, grid_info = cached_poisson(
                positions.tobytes(), normals.tobytes(),
                poisson_resolution, poisson_padding / 1000.0, sigma_factor
            )

        # Status indicator
        if recon_mesh is not None:
            n_verts = len(recon_mesh.vertices)
            n_faces = len(recon_mesh.faces)
            st.markdown(f"""
            <span class="status-badge success">✓ RECONSTRUCTION OK</span>
            &nbsp;&nbsp;
            <span class="status-badge info">{n_verts:,} vertices</span>
            &nbsp;&nbsp;
            <span class="status-badge info">{n_faces:,} faces</span>
            &nbsp;&nbsp;
            <span class="status-badge info">{len(positions)} contacts</span>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <span class="status-badge warning">⚠ RECONSTRUCTION FAILED</span>
            """, unsafe_allow_html=True)

        # 3D comparison figure
        fig = build_comparison_figure(
            gt_mesh=gt_mesh,
            recon_mesh=recon_mesh,
            positions=positions,
            normals=normals,
            fingers=fingers,
            gt_opacity=gt_opacity,
            recon_opacity=recon_opacity,
            show_contacts=show_contacts,
            title=f"{selected_object} — {selected_strategy} | GT + Poisson",
        )
        st.plotly_chart(fig, width="stretch", key="mesh_comparison")

        # Info
        if recon_mesh is not None and grid_info is not None:
            with st.expander("📐 Reconstruction Details"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{grid_info.get('isovalue', 0):.4f}</p>
                        <p class="metric-label">Isovalue</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    ext = grid_info["extent"]
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{ext[0]*1000:.1f}×{ext[1]*1000:.1f}×{ext[2]*1000:.1f}</p>
                        <p class="metric-label">Extent (mm)</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{poisson_resolution}³</p>
                        <p class="metric-label">Grid Resolution</p>
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Voxel Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div class="section-header">
        <h3>🧊 Voxelized Ground Truth vs Poisson SDF</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-panel">
        Voxel cube: <strong>{VOXEL_CUBE_SIZE_M*100:.0f}×{VOXEL_CUBE_SIZE_M*100:.0f}×{VOXEL_CUBE_SIZE_M*100:.0f} cm</strong>
        &nbsp;|&nbsp; Resolution: <strong>{VOXEL_RESOLUTION}³</strong>
        &nbsp;|&nbsp; Pitch: <strong>{VOXEL_CUBE_SIZE_M/VOXEL_RESOLUTION*1000:.2f} mm</strong>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    if gt_mesh is None:
        st.error("Could not load ground truth mesh")
    else:
        # GT voxelization
        with st.spinner("🧊 Voxelizing ground truth mesh..."):
            gt_verts_bytes = gt_mesh.vertices.astype(np.float64).tobytes()
            gt_faces_bytes = gt_mesh.faces.astype(np.int64).tobytes()
            gt_occupancy, gt_grid, gt_vox_points = cached_voxelize_gt(
                gt_verts_bytes, gt_faces_bytes
            )

        # Poisson SDF voxelization
        pred_vox_points = np.zeros((0, 3))
        pred_occupancy = np.zeros((VOXEL_RESOLUTION,)*3, dtype=bool)
        metrics = None

        if len(positions) >= 3:
            with st.spinner("🔄 Computing Poisson SDF & voxelizing..."):
                recon_mesh, implicit_field, grid_info = cached_poisson(
                    positions.tobytes(), normals.tobytes(),
                    poisson_resolution, poisson_padding / 1000.0, sigma_factor
                )

                if implicit_field is not None and grid_info is not None:
                    pred_occupancy, sdf_resampled, pred_vox_points = voxelize_sdf_field(
                        implicit_field, grid_info,
                        threshold=sdf_threshold,
                    )
                    metrics = compute_voxel_metrics(gt_occupancy, pred_occupancy)
        else:
            st.info("ℹ️ Poisson SDF voxelization requires contact points (≥3). "
                    "Showing ground truth voxels only.")

        # Metrics row
        if metrics is not None:
            st.markdown("")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                color = "#4ade80" if metrics["iou"] > 0.5 else "#fbbf24" if metrics["iou"] > 0.2 else "#ef4444"
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value" style="color: {color}">{metrics['iou']:.3f}</p>
                    <p class="metric-label">IoU</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{metrics['dice']:.3f}</p>
                    <p class="metric-label">Dice Score</p>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{metrics['precision']:.3f}</p>
                    <p class="metric-label">Precision</p>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{metrics['recall']:.3f}</p>
                    <p class="metric-label">Recall</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

        # Voxel comparison display
        col_mode = st.radio(
            "Display Mode",
            ["Overlay", "Side by Side"],
            horizontal=True,
            key="voxel_display_mode",
        )

        if col_mode == "Overlay":
            fig = build_voxel_comparison_figure(
                gt_vox_points, pred_vox_points,
                gt_opacity=voxel_opacity_gt,
                pred_opacity=voxel_opacity_pred,
                title=f"{selected_object} — Voxel Overlay (128³, 20cm)",
            )
            st.plotly_chart(fig, width="stretch", key="voxel_overlay")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Ground Truth Voxels**")
                fig_gt = build_voxel_comparison_figure(
                    gt_vox_points, np.zeros((0, 3)),
                    gt_opacity=voxel_opacity_gt,
                    title="GT Voxels",
                )
                st.plotly_chart(fig_gt, width="stretch", key="voxel_gt")
            with col_b:
                st.markdown("**Poisson SDF Voxels**")
                fig_pred = build_voxel_comparison_figure(
                    np.zeros((0, 3)), pred_vox_points,
                    pred_opacity=voxel_opacity_pred,
                    title="Poisson SDF Voxels",
                )
                st.plotly_chart(fig_pred, width="stretch", key="voxel_pred")

        # Voxel counts
        if metrics is not None:
            with st.expander("📊 Voxel Statistics"):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{metrics['n_gt_voxels']:,}</p>
                        <p class="metric-label">GT Voxels</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{metrics['n_pred_voxels']:,}</p>
                        <p class="metric-label">Pred Voxels</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{metrics['n_intersection']:,}</p>
                        <p class="metric-label">Intersection</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{metrics['n_union']:,}</p>
                        <p class="metric-label">Union</p>
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: SDF Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("""
    <div class="section-header">
        <h3>📊 SDF Field Analysis</h3>
    </div>
    """, unsafe_allow_html=True)

    if len(positions) < 3:
        st.warning("Need at least 3 contact points for SDF analysis.")
    else:
        with st.spinner("Computing SDF field..."):
            _, implicit_field, grid_info = cached_poisson(
                positions.tobytes(), normals.tobytes(),
                poisson_resolution, poisson_padding / 1000.0, sigma_factor
            )

        if implicit_field is not None:
            # Axis and slice selection
            col_axis, col_slice = st.columns([1, 3])
            with col_axis:
                axis_name = st.selectbox("Slice Axis", ["X", "Y", "Z"], index=2)
                axis_idx = {"X": 0, "Y": 1, "Z": 2}[axis_name]
            with col_slice:
                max_idx = implicit_field.shape[axis_idx] - 1
                slice_idx = st.slider(
                    f"Slice Index ({axis_name})",
                    0, max_idx, max_idx // 2,
                )

            fig_sdf = build_sdf_slice_figure(
                implicit_field, axis=axis_idx, slice_idx=slice_idx,
                title=f"Poisson SDF — {selected_object} [{selected_strategy}]"
            )
            st.plotly_chart(fig_sdf, width="stretch", key="sdf_slice")

            # Field statistics
            with st.expander("📈 Field Statistics"):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{implicit_field.min():.4f}</p>
                        <p class="metric-label">Min Value</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{implicit_field.max():.4f}</p>
                        <p class="metric-label">Max Value</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{implicit_field.mean():.4f}</p>
                        <p class="metric-label">Mean</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{implicit_field.std():.4f}</p>
                        <p class="metric-label">Std Dev</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Could not compute SDF field")
