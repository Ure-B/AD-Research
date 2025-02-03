import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
import trimesh
import numpy as np
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
    # Only use RGB colors, ignore vertices
    colors = mesh.visual.vertex_colors[:, :3] / 255.0  # RGB colors
    features = colors  # Only pass color as feature vector
    norm_features, min_vals, max_vals = normalize(features)  # Normalize color data
    return mesh.vertices, mesh.faces, features, min_vals, max_vals

def create_graph(vertices, features, k=8):
    tree = KDTree(vertices)
    edges = []
    for i, v in enumerate(vertices):
        _, idx = tree.query(v, k=k + 1)
        for j in idx[1:]:
            edges.append((i, j))
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

class VariationalGraphAE(GAE):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__(encoder=None)
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.conv_mu = GCNConv(latent_dim, latent_dim)
        self.conv_logvar = GCNConv(latent_dim, latent_dim)
        self.decoder1 = GCNConv(latent_dim, hidden_dim)
        self.decoder2 = GCNConv(hidden_dim, in_channels)  # Output only color values

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
    # Only use the color data to apply to the fixed geometry (vertices)
    reconstructed_features = denormalize(reconstructed_features, min_vals, max_vals)
    colors = reconstructed_features  # Now just the color data
    colors = np.nan_to_num(colors, nan=0.0, posinf=1.0, neginf=0.0)
    colors = np.clip(colors, 0.0, 1.0)
    colors = (colors * 255).astype(np.uint8)
    
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.visual.vertex_colors = np.hstack((colors, np.full((colors.shape[0], 1), 255)))  # Apply colors
    mesh.export(filename)
    print(f"Color-applied torus saved to: {filename}")

def train_vae(model, data, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon_x, mu, logvar = model(data.x, data.edge_index)
        loss = F.mse_loss(recon_x, data.x) + -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Load the original torus (now using only color data)
filepath = "torus_color_000.ply"
vertices, faces, features, min_vals, max_vals = load_ply(filepath)
data = create_graph(vertices, features)  # Only pass color features

# Initialize the VAE model and optimizer
model = VariationalGraphAE(in_channels=3, hidden_dim=64, latent_dim=32)  # Input channels is now 3 (RGB)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Train the VAE on the color data
train_vae(model, data, optimizer)

# Reconstruct and apply color to the fixed geometry (vertices)
with torch.no_grad():
    reconstructed_features, _, _ = model(data.x, data.edge_index)
save_ply(vertices, faces, reconstructed_features.numpy(), min_vals, max_vals, "reconstructed_torus.ply")
