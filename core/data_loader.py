"""
data_loader.py
==============
Load GLB meshes and contact JSON files from the grasp-dataset-gen pipeline.
"""
import json
import os
import glob
import numpy as np
import trimesh
from typing import Optional


def load_trimesh_glb(glb_path: str) -> Optional[trimesh.Trimesh]:
    """Load a GLB file and normalize it to fit within [-0.08, 0.08]."""
    if not os.path.exists(glb_path):
        return None
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes) if meshes else None
    else:
        mesh = scene
    if mesh is None:
        return None

    # Align strictly with grasp_dataset_gen/utils.py:normalize_mesh
    # to avoid scale/dimension misalignment between contacts and GT.
    verts = mesh.vertices.copy()
    verts -= verts.mean(axis=0)
    scale = np.max(np.linalg.norm(verts, axis=1))
    if scale > 1e-8:
        verts = verts / scale * 0.08
    
    # Return the perfectly aligned mesh
    return trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)


def load_contacts(contacts_path: str) -> dict:
    """Load a grasp contact JSON file."""
    if not os.path.exists(contacts_path):
        return {"contacts": []}
    with open(contacts_path) as f:
        return json.load(f)


def discover_dataset(data_dir: str) -> dict:
    """
    Scan a grasp-dataset-gen directory for available objects and strategies.

    Returns:
        {
            "glb_dir": str,
            "output_dir": str,
            "objects": {
                "sphere": {
                    "glb_path": str,
                    "strategies": ["front_back", "left_right", ...]
                },
                ...
            }
        }
    """
    glb_dir = os.path.join(data_dir, "data", "glb")
    output_dir = os.path.join(data_dir, "output")

    result = {"glb_dir": glb_dir, "output_dir": output_dir, "objects": {}}

    glb_files = sorted(glob.glob(os.path.join(glb_dir, "*.glb")))
    for glb_path in glb_files:
        name = os.path.splitext(os.path.basename(glb_path))[0]
        obj_output = os.path.join(output_dir, name)
        strategies = []
        if os.path.isdir(obj_output):
            for f in sorted(os.listdir(obj_output)):
                if f.startswith("grasp_") and f.endswith(".json"):
                    strat = f.replace("grasp_", "").replace(".json", "")
                    strategies.append(strat)
        result["objects"][name] = {
            "glb_path": glb_path,
            "strategies": strategies,
        }

    return result


def extract_contact_points(contacts_data: dict):
    """
    Extract positions and normals arrays from contact data.

    Returns:
        positions (N, 3), normals (N, 3), fingers (list of str)
    """
    contacts = contacts_data.get("contacts", [])
    if not contacts:
        return np.zeros((0, 3)), np.zeros((0, 3)), []

    positions = np.array([c["position"] for c in contacts], dtype=np.float64)
    normals = np.array([c["normal"] for c in contacts], dtype=np.float64)
    fingers = [c["finger"] for c in contacts]
    return positions, normals, fingers
