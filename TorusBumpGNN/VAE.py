import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
from torch_geometric.data import Data
import trimesh
import numpy as np
from tqdm import tqdm
import os
# import wandb
from torch_geometric.nn import knn_graph

def load_ply(filepath):
    """
    Loads .ply file using trimesh and returns vertices, faces, and features
    """
    mesh = trimesh.load_mesh(filepath)
    vertices = mesh.vertices
    faces = mesh.faces
    features, mean, std = normalize_vertices(vertices) 
    return vertices, faces, features, mean, std

def normalize_vertices(vertices):
    """
    Normalize the vertices using Z-score normalization (zero mean, unit variance)
    """
    mean = vertices.mean(axis=0)
    std = vertices.std(axis=0)
    return (vertices - mean) / std, mean, std

def denormalize_vertices(vertices, mean, std):
    """
    Denormalize the vertices using Z-score normalization (zero mean, unit variance)
    """
    return vertices * std + mean

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

    def __init__(self, in_channels, hidden_dim, latent_dim, kernel_size):
        super(GraphVAE, self).__init__()
        # Encoder
        self.conv1 = SplineConv(in_channels, hidden_dim, dim=3, kernel_size=kernel_size)
        self.conv2 = SplineConv(hidden_dim, hidden_dim, dim=3, kernel_size=kernel_size)
        self.conv3 = SplineConv(hidden_dim, 2 * latent_dim, dim=3, kernel_size=kernel_size)

        # Decoder
        self.decoder_fc1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.deconv1 = SplineConv(hidden_dim, hidden_dim, dim=3, kernel_size=kernel_size)
        self.deconv2 = SplineConv(hidden_dim, in_channels, dim=3, kernel_size=kernel_size)

        self.latent_dim = latent_dim

    def encode(self, x, edge_index):
        """
        Creates mean (mu) and log-variance (logvar)
        """
        row, col = edge_index
        pseudo = x[row] - x[col]
        pseudo = self._normalize_pseudo(pseudo)

        h = F.leaky_relu(self.conv1(x, edge_index, pseudo))
        h = F.leaky_relu(self.conv2(h, edge_index, pseudo))
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

    def decode(self, z, edge_index, pseudo):
        """Decode to reconstruct original shape"""
        h = F.leaky_relu(self.decoder_fc1(z))
        h = F.leaky_relu(self.deconv1(h, edge_index, pseudo))
        return self.deconv2(h, edge_index, pseudo)

    def forward(self, x, edge_index):
        """
        Forward encode -> reparameterize -> decode
        """
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        row, col = edge_index
        pseudo = x[row] - x[col]
        pseudo = self._normalize_pseudo(pseudo)
        recon_x = self.decode(z, edge_index, pseudo)
        return recon_x, mu, logvar

    def _normalize_pseudo(self, pseudo):
        """
        Normalize pseudo coordinates to [0, 1] for spline kernel input
        """
        pseudo_min = pseudo.min(dim=0, keepdim=True)[0]
        pseudo_max = pseudo.max(dim=0, keepdim=True)[0]
        return (pseudo - pseudo_min) / (pseudo_max - pseudo_min + 1e-8)

def save_ply(faces, reconstructed_features, mean, std, filename):
    """
    Save reconstruction as .ply
    """
    reconstructed_features = denormalize_vertices(reconstructed_features, mean, std)
    mesh = trimesh.Trimesh(reconstructed_features, faces, process=False)
    mesh.export(filename)
    print(f"Reconstructed torus saved to: {filename}")

def train_vae(model, train_data, optimizer, epochs=50, beta=1.0, device="cpu"):
    model.train()
    for epoch in tqdm(range(epochs)):

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        for data in train_data:
            data = data.to(device)  # Move data to GPU
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data.x, data.edge_index)

            # Reconstruction loss
            #recon_loss = F.l1_loss(recon_x, data.x)
            recon_loss = F.mse_loss(recon_x, data.x)
            # recon_loss = 0.5 * F.l1_loss(recon_x, data.x) + 0.5 * F.mse_loss(recon_x, data.x)

            # KL Divergence calculation
            kl_loss = torch.mean(
                -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1),
                dim=0
            )

            loss = recon_loss + beta * kl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        avg_loss = total_loss / len(train_data)
        avg_recon_loss = total_recon_loss / len(train_data)
        avg_kl_loss = total_kl_loss / len(train_data)

        print(
            f"Epoch {epoch} | Loss: {avg_loss:.6f} | Recon: {avg_recon_loss:.6f} | "
            f"KL: {avg_kl_loss:.6f}"
        )

        # wandb.log({
        #     'epoch': epoch,
        #     'train/loss': avg_loss,
        #     'train/recon_loss': avg_recon_loss,
        #     'train/kl_loss': avg_kl_loss,
        # })

if __name__ == "__main__":
    # Initialize wandb run
    # wandb.init(project='VAE')

    #config = wandb.config

    class Config():
        def __init__(self):
            self.epochs = 100
            self.beta = 0.001
            self.learning_rate = 0.001
            self.hidden_dim = 128
            self.latent_dim = 16
    
    config = Config()
    config.epochs = 200
    config.beta = 0.0001
    config.learning_rate = 0.001
    config.hidden_dim = 64
    config.latent_dim = 16

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dir = "better_training_data"
    test_dir = "testing_data"

    # Number of files to use
    num_of_training_data = 100
    num_of_testing_data = 4

    train_files = sorted(os.listdir(train_dir))[:num_of_training_data]
    test_files = sorted(os.listdir(test_dir))[:num_of_testing_data]

    # Load training data
    train_data = []
    for file in train_files:
        vertices, faces, features, mean, std = load_ply(os.path.join(train_dir, file))
        train_data.append(create_graph(vertices, features, k=16))

    # Load test data
    test_data = []
    for file in test_files:
        vertices, faces, features, mean, std = load_ply(os.path.join(test_dir, file))
        test_data.append((vertices, faces, mean, std, create_graph(vertices, features, k=16)))

    model = GraphVAE(
        in_channels=3,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        kernel_size=2
    ).to(device)  # Move model to GPU

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0)

    # Train VAE on GPU
    train_vae(model, train_data, optimizer, epochs=config.epochs, beta=config.beta, device=device)

    # Testing / Reconstruction
    model.eval()
    with torch.no_grad():
        for i, (vertices, faces, mean, std, data) in enumerate(test_data):
            data = data.to(device)  # Move test data to GPU
            recon_x, mu, logvar = model(data.x, data.edge_index)

            print("Input range:", data.x.min().item(), data.x.max().item())
            print("Output range:", recon_x.min().item(), recon_x.max().item())

            reconstructed_features = recon_x.cpu().numpy()  # Move back to CPU for saving
            output_filename = f"reconstructed_torus_VAE_{i}.ply"
            save_ply(faces, reconstructed_features, mean, std, output_filename)
    
    #wandb.finish()