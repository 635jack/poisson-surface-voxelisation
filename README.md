# Mesh Comparison Viewer

Interactive Streamlit app for comparing **ground truth 3D meshes** with **Screened Poisson Surface Reconstructions**, including voxelized SDF comparison.

## 📺 Video Demo

[Watch the demonstration video here](https://youtu.be/TbDWMjjH8l0)

## Features

- 🔍 **3D Mesh Comparison**: Side-by-side view of ground truth mesh vs Poisson reconstruction
- 🧊 **Voxel Comparison**: Voxelized ground truth vs voxelized SDF from Poisson (128³ resolution, 20×20×20 cm cube)
- 📊 **Quantitative Metrics**: Hausdorff distance, Chamfer distance, IoU
- 🎨 **Interactive Controls**: Adjust Poisson resolution, isovalue, voxel opacity

## Setup

```bash
cd mesh-comparison-viewer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py -- --data-dir ../grasp-dataset-gen
```

## Data Format

Expects data from the `grasp-dataset-gen` pipeline:
- GLB meshes in `data/glb/`
- Contact JSON files in `output/<object>/grasp_<strategy>.json`
