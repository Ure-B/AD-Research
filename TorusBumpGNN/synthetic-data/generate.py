import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import pyvista as pv

from sdf.sdf import *
from glob import glob
from tqdm import tqdm
from contextlib import contextmanager
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser(description='Synthetic Dataset')
parser.add_argument('--dataset', type=str, default='torus_bump', choices=['all', 'torus_bump'])
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

def create_torus_bump_files():
    """Create synthetic torus data."""

    if not os.path.exists("torus_bump"):
        os.makedirs("torus_bump")

    # Create template file
    with suppress_stdout():
        filepath = "torus_bump/torus_bump_template.ply"
        a = torus(1, 0.4)
        a.save(filepath, samples=30000)

    # Create mesh files and save labels
    labels = {}
    feature_range_thickness = np.linspace(0.3, 0.5, 15, endpoint=False)
    feature_range_angle = np.linspace(0, 360, 1, endpoint=False)
    scaler = MinMaxScaler()
    scaler.fit(feature_range_angle.reshape(-1, 1))
    #scaler.fit(feature_range_thickness.reshape(-1, 1))

    i = 0
    #for angle, thickness in tqdm(zip(feature_range_angle, feature_range_thickness)):
    for angle in tqdm(feature_range_angle):
        with suppress_stdout():
            filepath = f'torus_bump/torus_bump_{i:03d}.ply'
            filename = filepath.split("/")[-1].split(".")[0]
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            thickness = np.random.choice(feature_range_thickness)
            labels[filename] = np.array([scaler.transform(np.array([angle]).reshape(-1, 1)).item(), thickness])

            #a = torus(1, thickness)
            b = sphere(0.7).translate((x, y, 0))
            f = union(a, b, k=0.2)
            f.save(filepath, samples=20000)
            i += 1

    torch.save(labels, "torus_bump\\torus_bump_labels.pt")

    #pd.DataFrame(list(labels.items()), columns=['shape', 'label']).to_csv("torus_bump\\torus_bump_labels.csv")

    # Create an empty DataFrame
    df = pd.DataFrame(columns=['shape', 'angle', 'thickness'])

    # Iterate over the items in the labels dictionary
    for shape, values in labels.items():
        # Access the individual values from the NumPy array
        value1 = values[0]
        value2 = values[1]
    
        # Append a new row with the values to the DataFrame
        df = df._append({'shape': shape, 'angle': value1, 'thickness': value2}, ignore_index=True)

    # Write the DataFrame to an Excel file
    df.to_csv("torus_bump\\torus_bump_labels.csv")

    # Convert meshes into .vtk file format
    if not os.path.exists("torus_bump\\vtk_files"):
        os.makedirs("torus_bump\\vtk_files")

    for mesh_path in tqdm(glob("torus_bump\\*.ply")):
        filename = mesh_path.split("\\")[-1].split(".")[0]
        mesh = pv.read(mesh_path)
        mesh.save(f"torus_bump\\vtk_files\\{filename}.vtk")


if __name__ == "__main__":

    if args.dataset == "all":
        create_torus_bump_files()

    elif args.dataset == "torus_bump":
        create_torus_bump_files()