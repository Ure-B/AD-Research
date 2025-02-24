import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
from torch_geometric.data import Data
import trimesh
import numpy as np
from scipy.spatial import KDTree

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

class GraphVAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, kernel_size=2):
        super(GraphVAE, self).__init__()
        self.conv1 = SplineConv(in_channels, hidden_dim, dim=3, kernel_size=kernel_size)
        self.conv_mu = SplineConv(hidden_dim, latent_dim, dim=3, kernel_size=kernel_size)
        self.conv_logvar = SplineConv(hidden_dim, latent_dim, dim=3, kernel_size=kernel_size)
        
        self.decoder_fc1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = torch.nn.Linear(hidden_dim, in_channels)

    def encode(self, x, edge_index):
        device = x.device  
        pseudo = torch.zeros(edge_index.shape[1], 3, device=device) 

        h = F.relu(self.conv1(x, edge_index, pseudo))
        mu = self.conv_mu(h, edge_index, pseudo)
        logvar = self.conv_logvar(h, edge_index, pseudo)
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

def save_ply(vertices, faces, reconstructed_features, filename):
    mesh = trimesh.Trimesh(reconstructed_features, faces, process=False)
    mesh.export(filename)
    print(f"Reconstructed torus saved to: {filename}")

def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x)
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.1 * kl_div

def train_vae(model, data, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon_x, mu, logvar = model(data.x, data.edge_index)
        loss = loss_function(recon_x, data.x, mu, logvar)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

filepath = "torus_bump_000.ply"
vertices, faces, features = load_ply(filepath)
data = create_graph(vertices, features, k=16)

model = GraphVAE(in_channels=3, hidden_dim=64, latent_dim=32, kernel_size=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_vae(model, data, optimizer)

with torch.no_grad():
    reconstructed_features, _, _ = model(data.x, data.edge_index)
    reconstructed_features = reconstructed_features.numpy()

save_ply(vertices, faces, reconstructed_features, "reconstructed_torus.ply")
