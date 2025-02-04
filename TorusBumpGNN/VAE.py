import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data, DataLoader
import trimesh
import numpy as np
from scipy.spatial import KDTree
import os

def normalize(data):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    delta = max_vals - min_vals
    epsilon = 1e-6
    delta = np.where(delta == 0, epsilon, delta)
    return (data - min_vals) / delta, min_vals, max_vals

def denormalize(data, min_vals, max_vals):
    return data * (max_vals - min_vals) + min_vals

def load_ply(filepath):
    mesh = trimesh.load_mesh(filepath)
    vertices = mesh.vertices
    faces = mesh.faces
    features = vertices  # Only use the xyz positions as features
    norm_features, min_vals, max_vals = normalize(features)
    return vertices, faces, norm_features, min_vals, max_vals

def create_graph(vertices, features, min_vals, max_vals, k=8):
    tree = KDTree(vertices)
    edges = []
    for i, v in enumerate(vertices):
        _, idx = tree.query(v, k=k + 1)
        for j in idx[1:]:
            edges.append((i, j))
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)

    # Store additional attributes in the Data object
    data = Data(x=x, edge_index=edge_index)
    data.min_vals = torch.tensor(min_vals, dtype=torch.float)  # Store min/max as tensor
    data.max_vals = torch.tensor(max_vals, dtype=torch.float)
    
    return data

class VariationalGraphAE(GAE):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__(encoder=None)
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.conv_mu = GCNConv(latent_dim, latent_dim)
        self.conv_logvar = GCNConv(latent_dim, latent_dim)
        self.decoder1 = GCNConv(latent_dim, hidden_dim)
        self.decoder2 = GCNConv(hidden_dim, in_channels)  # Reconstruct only vertex positions

    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.conv_mu(h, edge_index), self.conv_logvar(h, edge_index)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        h = F.relu(self.decoder1(z, edge_index))
        return self.decoder2(h, edge_index), mu, logvar

def save_ply(vertices, faces, reconstructed_features, min_vals, max_vals, filename):
    reconstructed_features = denormalize(reconstructed_features, min_vals.numpy(), max_vals.numpy())
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.export(filename)
    print(f"Reconstructed torus saved to: {filename}")

def train_vae(model, data_loader, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data.x, data.edge_index)
            loss = F.mse_loss(recon_x, data.x) + -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch}, Loss: {total_loss / len(data_loader)}')

# Load multiple torus files from a directory
def load_multiple_toruses(directory_path):
    data_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.ply'):
            filepath = os.path.join(directory_path, filename)
            vertices, faces, features, min_vals, max_vals = load_ply(filepath)
            data = create_graph(vertices, features, min_vals, max_vals)  # Pass min/max values
            data.faces = faces  # Store faces separately
            data.vertices = vertices  # Store vertices for reconstruction
            data_list.append(data)
    return data_list

# Set up the training environment
directory_path = "torus_files"  # Folder containing the .ply files
data_list = load_multiple_toruses(directory_path)

# Create DataLoader for batching
data_loader = DataLoader(data_list, batch_size=4, shuffle=True)  # Adjust batch size as needed

# Initialize the VAE model and optimizer
model = VariationalGraphAE(in_channels=3, hidden_dim=64, latent_dim=32)  # in_channels=3 for xyz only
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Train the VAE
train_vae(model, data_loader, optimizer)

# Reconstruct and save the meshes for each input torus
with torch.no_grad():
    for data in data_list:
        reconstructed_features, _, _ = model(data.x, data.edge_index)
        save_ply(data.vertices, data.faces, reconstructed_features.numpy(), data.min_vals, data.max_vals, "reconstructed_torus.ply")