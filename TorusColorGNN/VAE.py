import torch
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, GAE
from torch_geometric.data import Data
import trimesh
import numpy as np
import os
from scipy.spatial import KDTree

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
    colors = mesh.visual.vertex_colors[:, :3] / 255.0
    norm_features, min_vals, max_vals = normalize(colors)
    return mesh.vertices, mesh.faces, norm_features, min_vals, max_vals

def create_graph(vertices, features, k=16): 
    tree = KDTree(vertices)
    edges = []
    for i, v in enumerate(vertices):
        _, idx = tree.query(v, k=k + 1)
        for j in idx[1:]:
            edges.append((i, j))
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

class VAE(GAE):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__(encoder=None)
        self.encoder1 = EdgeConv(torch.nn.Linear(2 * in_channels, hidden_dim), aggr="mean")
        self.encoder2 = EdgeConv(torch.nn.Linear(2 * hidden_dim, latent_dim), aggr="mean")
        self.mu_layer = EdgeConv(torch.nn.Linear(2 * latent_dim, latent_dim), aggr="mean")
        self.logvar_layer = EdgeConv(torch.nn.Linear(2 * latent_dim, latent_dim), aggr="mean")
        self.decoder1 = EdgeConv(torch.nn.Linear(2 * latent_dim, hidden_dim), aggr="mean")
        self.decoder2 = EdgeConv(torch.nn.Linear(2 * hidden_dim, in_channels), aggr="mean")

    def encode(self, x, edge_index):
        h = F.relu(self.encoder1(x, edge_index))
        h = F.relu(self.encoder2(h, edge_index))
        return self.mu_layer(h, edge_index), self.logvar_layer(h, edge_index)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        h = F.relu(self.decoder1(z, edge_index))
        return self.decoder2(h, edge_index), mu, logvar

def laplacian_smoothness_loss(x, edge_index):
    row, col = edge_index
    smoothness = torch.mean((x[row] - x[col]) ** 2)
    return smoothness

def train_vae(model, dataset, optimizer, epochs=10, lambda_smooth=0.1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataset:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data.x, data.edge_index)
            mse_loss = F.mse_loss(recon_x, data.x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            smoothness_loss = laplacian_smoothness_loss(recon_x, data.edge_index)
            loss = mse_loss + kl_loss + lambda_smooth * smoothness_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Avg Loss: {total_loss / len(dataset):.6f}")

def save_ply(vertices, faces, reconstructed_features, min_vals, max_vals, filename):
    reconstructed_features = denormalize(reconstructed_features, min_vals, max_vals)
    colors = np.clip(reconstructed_features, 0.0, 1.0)
    colors = (colors * 255).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.visual.vertex_colors = np.hstack((colors, np.full((colors.shape[0], 1), 255)))
    mesh.export(filename)
    print(f"Color-applied torus saved to: {filename}")

def load_dataset(directory):
    dataset = []
    for filename in os.listdir(directory):
        if filename.endswith(".ply"):
            filepath = os.path.join(directory, filename)
            vertices, faces, features, min_vals, max_vals = load_ply(filepath)
            data = create_graph(vertices, features)
            dataset.append(data)
    return dataset, vertices, faces, min_vals, max_vals

directory = "torus_color"
dataset, vertices, faces, min_vals, max_vals = load_dataset(directory)

model = VAE(in_channels=3, hidden_dim=64, latent_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

train_vae(model, dataset, optimizer)

with torch.no_grad():
    reconstructed_features, _, _ = model(dataset[0].x, dataset[0].edge_index)
save_ply(vertices, faces, reconstructed_features.numpy(), min_vals, max_vals, "reconstructed_torus.ply")
