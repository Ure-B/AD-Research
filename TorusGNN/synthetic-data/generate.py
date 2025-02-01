import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import pyvista as pv
from meshlib import mrmeshpy as mm

from sdf.sdf import *
from glob import glob
from tqdm import tqdm
from contextlib import contextmanager
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser(description='Synthetic Dataset')
parser.add_argument('--dataset', type=str, default='all', choices=['all', 'box_bump', 'torus_bump', 'torus_color'])
args = parser.parse_args()


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def create_torus_color_files():
    """Create synthetic torus data."""

    if not os.path.exists("torus_color"):
        os.makedirs("torus_color")

    # Create template file
    with suppress_stdout():
        # Build Torus
        filepath = "torus_color/torus_color_template.ply"
        a = torus(1, 0.35)
        a.save(filepath, samples=30000)

    # Create mesh files and save labels
    labels = {}
    feature_range = np.linspace(0, 360, 500, endpoint=False)
    scaler = MinMaxScaler()
    scaler.fit(feature_range.reshape(-1, 1))

    i = 0
    for angle in tqdm(feature_range):
        with suppress_stdout():
            filepath = f'torus_color/torus_color_{i:03d}.ply'
            filename = filepath.split("/")[-1].split(".")[0]
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            labels[filename] = scaler.transform(np.array([angle]).reshape(-1, 1)).item()

            # Add color
            mesh = mm.loadMesh("torus_color/torus_color_template.ply")
            red = mm.Color(0,0,255,255)
            vertColors = mm.VertColors(mesh.points.size(), red)

            # Define the red circle parameters
            circle_center = mm.Vector3f(x, y, 0.0)
            circle_radius = 1.2

            # Blend red color into vertices within the circle
            for j, vertex in enumerate(mesh.points):
                distance = (vertex - circle_center).length()
                if distance <= circle_radius:
                    blend_factor = 1 - (distance / circle_radius)
                    red_component = int(255 * blend_factor)
                    blue_component = int(255 * (1 - blend_factor))

                    c = mm.Color(red_component, 0, blue_component, 255)
                    id = mm.VertId(j)
                    vertColors.autoResizeSet(id, c)

            settings = mm.SaveSettings()
            settings.colors = vertColors
            mm.saveMesh(mesh, filepath, settings)

            i += 1

    torch.save(labels, "torus_color\\torus_color_labels.pt")
    pd.DataFrame(list(labels.items()), columns=['shape', 'label']).to_csv("torus_color\\torus_color_labels.csv")

    # Convert meshes into .vtk file format
    if not os.path.exists("torus_color\\vtk_files"):
        os.makedirs("torus_color\\vtk_files")

    for mesh_path in tqdm(glob("torus_color\\*.ply")):
        filename = mesh_path.split("\\")[-1].split(".")[0]
        mesh = pv.read(mesh_path)
        mesh.save(f"torus_color\\vtk_files\\{filename}.vtk")


if __name__ == "__main__":

    if args.dataset == "all":
        create_torus_color_files()

    elif args.dataset == "torus_color":
        create_torus_color_files()
