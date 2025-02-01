import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
import trimesh
import numpy as np
from scipy.spatial import KDTree

# Normalize and denormalize functions
def normalize(data):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    delta = max_vals - min_vals
    # Prevent division by zero by checking if delta is zero, and using a small epsilon value if true
    epsilon = 1e-6
    delta = np.where(delta == 0, epsilon, delta)
    return (data - min_vals) / delta, min_vals, max_vals

def denormalize(data, min_vals, max_vals):
    return data * (max_vals - min_vals) + min_vals


# Load .ply file and extract point cloud (position + color)
def load_ply(filepath):
    mesh = trimesh.load_mesh(filepath)
    vertices = mesh.vertices  # 3D positions
    colors = mesh.visual.vertex_colors[:, :3] / 255.0  # Normalize colors
    features = np.hstack((vertices, colors))  # Combine position and color as features
    
    # Normalize position data
    norm_features, min_vals, max_vals = normalize(features)
    return vertices, norm_features, min_vals, max_vals

# Create graph using k-nearest neighbors
def create_graph(vertices, features, k=8):
    tree = KDTree(vertices)
    edges = []
    for i, v in enumerate(vertices):
        _, idx = tree.query(v, k=k + 1)
        for j in idx[1:]:  # Exclude self-loop
            edges.append((i, j))
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    
    x = torch.tensor(features, dtype=torch.float)
    print(f"Feature shape: {x.shape}, Edge index shape: {edge_index.shape}")
    return Data(x=x, edge_index=edge_index)

# Define Variational Autoencoder (VAE) class
class VariationalGraphAE(GAE):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__(encoder=None)
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.conv_mu = GCNConv(latent_dim, latent_dim)
        self.conv_logvar = GCNConv(latent_dim, latent_dim)
        
        self.decoder1 = GCNConv(latent_dim, hidden_dim)
        self.decoder2 = GCNConv(hidden_dim, in_channels)  # Ensure correct shape

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
        return self.decoder2(h, edge_index), mu, logvar  # Correct shape

# Save reconstructed torus as .ply
def save_ply(vertices, reconstructed_features, min_vals, max_vals, filename):
    reconstructed_features = denormalize(reconstructed_features, min_vals, max_vals)
    
    # Extract colors and handle invalid values
    colors = reconstructed_features[:, 3:]
    
    # Replace any NaN or Inf with 0
    colors = np.nan_to_num(colors, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Clip values to the valid range [0, 1]
    colors = np.clip(colors, 0.0, 1.0)
    
    # Scale colors to 0-255 range and convert to uint8
    colors = (colors * 255).astype(np.uint8)
    
    mesh = trimesh.Trimesh(vertices, process=False)
    mesh.visual.vertex_colors = np.hstack((colors, np.full((colors.shape[0], 1), 255)))  # Add alpha channel
    mesh.export(filename)

# Training function
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

# Example usage
filepath = "torus.ply"
vertices, features, min_vals, max_vals = load_ply(filepath)
data = create_graph(vertices, features)

model = VariationalGraphAE(in_channels=6, hidden_dim=64, latent_dim=32)  # Corrected decoder structure
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

train_vae(model, data, optimizer)

# Reconstruct and save output
with torch.no_grad():
    reconstructed_features, _, _ = model(data.x, data.edge_index)
save_ply(vertices, reconstructed_features.numpy(), min_vals, max_vals, "reconstructed_torus.ply")
