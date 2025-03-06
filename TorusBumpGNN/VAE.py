import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
from torch_geometric.data import Data
import trimesh
import numpy as np
from tqdm import tqdm
import os
import wandb
from torch_geometric.nn import knn_graph

def load_ply(filepath):
    """
    Loads .ply file using trimesh and returns vertices, faces, and features
    """
    mesh = trimesh.load_mesh(filepath)
    vertices = mesh.vertices
    faces = mesh.faces
    features = vertices
    features = normalize_vertices(features) 
    return vertices, faces, features

def normalize_vertices(vertices):
    """
    Normalize the vertices using Z-score normalization (zero mean, unit variance)
    """
    mean = vertices.mean(axis=0)
    std = vertices.std(axis=0)
    return (vertices - mean) / std

def create_graph(vertices, features, k=16):
    """
    Creates graph using k-nearest neighbors for edges
    """
    x = torch.tensor(features, dtype=torch.float)
    edge_index = knn_graph(x, k=k, batch=None, loop=False)
    return Data(x=x, edge_index=edge_index)


class GraphVAE(torch.nn.Module):
    """
    Graph VAE using SplineConv layers.
    """

    def __init__(self, in_channels, hidden_dim, latent_dim, kernel_size=2):
        super(GraphVAE, self).__init__()
        # Encoder
        self.conv1 = SplineConv(in_channels, hidden_dim, dim=3, kernel_size=kernel_size)
        self.conv2 = SplineConv(hidden_dim, hidden_dim, dim=3, kernel_size=kernel_size) 
        self.conv3 = SplineConv(hidden_dim, 2 * latent_dim, dim=3, kernel_size=kernel_size)

        # Decoder
        self.decoder_fc1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.decoder_fc3 = torch.nn.Linear(hidden_dim, in_channels)

        self.latent_dim = latent_dim

    def encode(self, x, edge_index):
        """
        Creates mean (mu) and log-variance (logvar)
        """
        device = x.device
        pseudo = torch.zeros(edge_index.shape[1], 3, device=device)

        h = F.relu(self.conv1(x, edge_index, pseudo))
        h = F.relu(self.conv2(h, edge_index, pseudo))
        h = self.conv3(h, edge_index, pseudo)

        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
        z = mu + sigma * eps, where eps ~ N(0, I)
        sigma = exp(0.5 * logvar)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode to reconstruct original shape"""
        h = F.relu(self.decoder_fc1(z))
        h = F.relu(self.decoder_fc2(h))
        return self.decoder_fc3(h)

    def forward(self, x, edge_index):
        """
        Forward encode -> reparameterize -> decode
        """
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def save_ply(faces, reconstructed_features, filename):
    """
    Save reconstruction as .ply
    """
    mesh = trimesh.Trimesh(reconstructed_features, faces, process=False)
    mesh.export(filename)
    print(f"Reconstructed torus saved to: {filename}")


def train_vae(model, train_data, optimizer, epochs=50, beta=3.0):

    model.train()

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        for data in train_data:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data.x, data.edge_index)

            # Reconstruction loss
            recon_loss = F.mse_loss(recon_x, data.x)
            #recon_loss = F.l1_loss(recon_x, data.x)

            # KL Divergence
            #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

            loss = recon_loss + beta * kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        # Average losses for wandb tracking
        avg_loss = total_loss / len(train_data)
        avg_recon_loss = total_recon_loss / len(train_data)
        avg_kl_loss = total_kl_loss / len(train_data)

        print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Recon: {avg_recon_loss:.6f} | KL: {avg_kl_loss:.6f}")

        wandb.log({
            'epoch': epoch,
            'train/loss': avg_loss,
            'train/recon_loss': avg_recon_loss,
            'train/kl_loss': avg_kl_loss,
        })


if __name__ == "__main__":
    # Initialize wandb run
    wandb.init(project='VAE')

    config = wandb.config
    config.epochs = 5
    config.beta = 0.001
    config.learning_rate = 0.001
    config.hidden_dim = 64
    config.latent_dim = 16

    train_dir = "training_data"
    test_dir = "testing_data"

    # Number of files to use
    num_of_training_data = 1
    num_of_testing_data = 1

    train_files = sorted(os.listdir(train_dir))[:num_of_training_data]
    test_files = sorted(os.listdir(test_dir))[:num_of_testing_data]

    # Load training data
    train_data = []
    for file in train_files:
        vertices, faces, features = load_ply(os.path.join(train_dir, file))
        train_data.append(create_graph(vertices, features, k=16))

    # Load test data
    test_data = []
    for file in test_files:
        vertices, faces, features = load_ply(os.path.join(test_dir, file))
        test_data.append((vertices, faces, create_graph(vertices, features, k=16)))

    model = GraphVAE(
        in_channels=3,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        kernel_size=2
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0)

    # Train VAE
    train_vae(model, train_data, optimizer, epochs=config.epochs, beta=config.beta)

    # Testing / Reconstruction
    model.eval()
    with torch.no_grad():
        for i, (vertices, faces, data) in enumerate(test_data):
            
            recon_x, mu, logvar = model(data.x, data.edge_index)
            reconstructed_features = recon_x.cpu().numpy()

            output_filename = f"reconstructed_torus_VAE_{i}.ply"
            save_ply(faces, reconstructed_features, output_filename)

    wandb.finish()