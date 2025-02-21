import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import trimesh
import numpy as np
from scipy.spatial import KDTree
import os
from torch_spline_conv import spline_conv

def load_ply(filepath):
    mesh = trimesh.load_mesh(filepath)
    vertices = mesh.vertices
    faces = mesh.faces
    features = vertices  # Only use the xyz positions as features (no normalization)
    return vertices, faces, features

def create_graph(vertices, features, k=8):
    # Directly use the raw features (no normalization)
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
    
    return data

class VariationalGraphAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, device):
        super().__init__()
        self.device = device  # Store the device
        
        # Define kernel size and degree
        self.kernel_size = 3
        self.degree = 2
        self.is_open_spline = False
        
        # Convert kernel_size to a tensor
        self.kernel_size_tensor = torch.tensor(self.kernel_size, dtype=torch.long).to(device)
        
        # Convert is_open_spline to a tensor (True/False as Tensor)
        self.is_open_spline_tensor = torch.tensor(self.is_open_spline, dtype=torch.bool).to(device)
        
        # Initialize weights for spline conv layers
        self.weight1 = torch.nn.Parameter(torch.randn(in_channels * self.kernel_size, hidden_dim).to(device))
        self.weight2 = torch.nn.Parameter(torch.randn(hidden_dim * self.kernel_size, latent_dim).to(device))
        
        self.mu_weight = torch.nn.Parameter(torch.randn(latent_dim * self.kernel_size, latent_dim).to(device))
        self.logvar_weight = torch.nn.Parameter(torch.randn(latent_dim * self.kernel_size, latent_dim).to(device))

        # Decoder layers
        self.decoder1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.decoder2 = torch.nn.Linear(hidden_dim, in_channels)  # Output is xyz positions

    def encode(self, x, edge_index, pseudo):
        # Use spline_conv for convolutional layers
        h1 = spline_conv(x, edge_index, pseudo, self.weight1, self.kernel_size_tensor, self.is_open_spline_tensor, self.degree)
        h2 = spline_conv(h1, edge_index, pseudo, self.weight2, self.kernel_size_tensor, self.is_open_spline_tensor, self.degree)
        
        mu = spline_conv(h2, edge_index, pseudo, self.mu_weight, self.kernel_size_tensor, self.is_open_spline_tensor, self.degree)
        logvar = spline_conv(h2, edge_index, pseudo, self.logvar_weight, self.kernel_size_tensor, self.is_open_spline_tensor, self.degree)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)  # Ensure eps is on the correct device
        return mu + eps * std

    def forward(self, x, edge_index, pseudo):
        # Ensure inputs are on the correct device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        pseudo = pseudo.to(self.device)
        
        # Encode and get latent variables (mu, logvar)
        mu, logvar = self.encode(x, edge_index, pseudo)
        z = self.reparameterize(mu, logvar)
        
        # Decode from latent variable z to get the learned features (xyz positions)
        h = F.relu(self.decoder1(z))
        reconstructed_features = self.decoder2(h)  # Final output is reconstructed xyz positions
        
        return reconstructed_features, mu, logvar

def save_ply(vertices, faces, reconstructed_features, filename):
    # Save the reconstructed mesh with the learned features (xyz positions)
    mesh = trimesh.Trimesh(reconstructed_features, faces, process=False)  # Use reconstructed features as vertices
    mesh.export(filename)
    print(f"Reconstructed torus saved to: {filename}")

def train_vae(model, data_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data.x, data.edge_index, data.pseudo)
            # Loss function: Reconstruction loss + KL divergence
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
            vertices, faces, features = load_ply(filepath)
            data = create_graph(vertices, features)  # Use raw features (no normalization)
            data.faces = faces  # Store faces separately
            data.vertices = vertices  # Store vertices for reconstruction
            data.pseudo = torch.randn(vertices.shape[0], 1)  # Pseudo tensor (for spline conv)
            data_list.append(data)
    return data_list

# Set up the training environment
directory_path = "torus_files"  # Folder containing the .ply files
data_list = load_multiple_toruses(directory_path)

# Create DataLoader for batching
data_loader = DataLoader(data_list, batch_size=4, shuffle=True)  # Adjust batch size as needed

# Define the device (either CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the VAE model and optimizer
model = VariationalGraphAE(in_channels=3, hidden_dim=64, latent_dim=32, device=device)  # in_channels=3 for xyz only
model.to(device)  # Ensure the model is on the correct device
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Train the VAE
train_vae(model, data_loader, optimizer)

# Reconstruct and save the meshes for each input torus
with torch.no_grad():
    for data in data_list:
        # Ensure that data.x, data.edge_index, and data.pseudo are on the same device as the model
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.pseudo = data.pseudo.to(device)
        
        # Use the learned latent features for reconstruction
        reconstructed_features, _, _ = model(data.x, data.edge_index, data.pseudo)
        save_ply(data.vertices, data.faces, reconstructed_features.cpu().numpy(), "reconstructed_torus.ply")
