import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
from torch_geometric.data import Data
import trimesh
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
import os

def load_ply(filepath):
    mesh = trimesh.load_mesh(filepath)
    vertices = mesh.vertices
    faces = mesh.faces
    features = vertices
    return vertices, faces, features

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

def save_ply(faces, reconstructed_features, filename):
    mesh = trimesh.Trimesh(reconstructed_features, faces, process=False)
    mesh.export(filename)
    print(f"Reconstructed torus saved to: {filename}")

class GraphVAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, kernel_size=2):
        super(GraphVAE, self).__init__()
        self.conv1 = SplineConv(in_channels, hidden_dim, dim=3, kernel_size=kernel_size)
        self.conv2_mu = SplineConv(hidden_dim, latent_dim, dim=3, kernel_size=kernel_size)
        self.conv2_logvar = SplineConv(hidden_dim, latent_dim, dim=3, kernel_size=kernel_size)
        
        self.decoder_fc1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = torch.nn.Linear(hidden_dim, in_channels)
    
    def encode(self, x, edge_index):
        device = x.device  
        pseudo = torch.zeros(edge_index.shape[1], 3, device=device) 
        h = F.relu(self.conv1(x, edge_index, pseudo))
        mu = self.conv2_mu(h, edge_index, pseudo)
        logvar = self.conv2_logvar(h, edge_index, pseudo)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.decoder_fc1(z))
        return self.decoder_fc2(h)
    
    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=0.01):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

def train_vae(model, train_data, optimizer, epochs=5):
    model.train()
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        total_loss = 0
        for data in train_data:
            recon_x, mu, logvar = model(data.x, data.edge_index)
            loss = vae_loss(recon_x, data.x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch}, Loss: {total_loss / len(train_data)}')

train_dir = "training_data"
test_dir = "testing_data"
num_of_training_data = 4
num_of_testing_data = 1
train_files = sorted(os.listdir(train_dir))[:num_of_training_data]
test_files = sorted(os.listdir(test_dir))[:num_of_testing_data]

train_data = []
for file in train_files:
    vertices, faces, features = load_ply(os.path.join(train_dir, file))
    train_data.append(create_graph(vertices, features, k=16))

test_data = []
for file in test_files:
    vertices, faces, features = load_ply(os.path.join(test_dir, file))
    test_data.append((vertices, faces, create_graph(vertices, features, k=16)))

model = GraphVAE(in_channels=3, hidden_dim=64, latent_dim=64, kernel_size=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_vae(model, train_data, optimizer)

with torch.no_grad():
    for i in range(len(test_data)):
        vertices, faces, data = test_data[i]
        reconstructed_features, _, _ = model(data.x, data.edge_index)
        print("Original:", data.x[:5]) 
        print("Reconstructed:", reconstructed_features[:5])
        save_ply(faces, reconstructed_features.numpy(), f"reconstructed_torus_VAE_{i}.ply")
