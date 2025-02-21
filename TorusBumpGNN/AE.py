import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
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

class GraphAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super(GraphAE, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.decoder_fc1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = torch.nn.Linear(hidden_dim, in_channels)

    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h

    def decode(self, z):
        h = F.relu(self.decoder_fc1(z))
        return self.decoder_fc2(h)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z)

def save_ply(vertices, faces, reconstructed_features, filename):
    mesh = trimesh.Trimesh(reconstructed_features, faces, process=False)
    mesh.export(filename)
    print(f"Reconstructed torus saved to: {filename}")

def train_ae(model, data, optimizer, epochs=300):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon_x = model(data.x, data.edge_index)
        loss = F.mse_loss(recon_x, data.x) + 0.1
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

filepath = "torus_bump_000.ply"
vertices, faces, features = load_ply(filepath)
data = create_graph(vertices, features, k=16)

model = GraphAE(in_channels=3, hidden_dim=64, latent_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_ae(model, data, optimizer, epochs=300)

with torch.no_grad():
    reconstructed_features = model(data.x, data.edge_index).numpy()
save_ply(vertices, faces, reconstructed_features, "reconstructed_torus.ply")
