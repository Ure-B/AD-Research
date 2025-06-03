import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
# import meshplot as mp
from tqdm import tqdm
from collections import deque
from torch_scatter import scatter_add

def build_spiral(adj_list, seq_length=9):
    spirals = []
    for neighbors in adj_list:
        spiral = []
        visited = set()
        queue = deque()
        queue.append((0, neighbors))

        while queue and len(spiral) < seq_length:
            depth, current = queue.popleft()
            for v in current:
                if v not in visited:
                    visited.add(v)
                    spiral.append(v)
                    if len(spiral) == seq_length:
                        break
            if len(spiral) < seq_length:
                new_neighbors = []
                for v in current:
                    new_neighbors.extend(adj_list[v])
                queue.append((depth + 1, new_neighbors))

        if len(spiral) < seq_length:
            spiral += [spiral[-1]] * (seq_length - len(spiral))
        spirals.append(spiral[:seq_length])
    return torch.LongTensor(spirals)

#Normalize the vertices value in range [0,1] 
def normalize_n1p1(points, keep_aspect_ratio=True, eps=0.01):
    # coords, norm_params = normalize_center_scale(points=points, scale=1.0)
    coords = points
    if keep_aspect_ratio:
        coord_max = np.amax(coords)+eps
        coord_min = np.amin(coords)
    else:
        coord_max = np.amax(coords, axis=0, keepdims=True)+eps
        coord_min = np.amin(coords, axis=0, keepdims=True)
    coords = (coords - coord_min) / (coord_max - coord_min)
    coords -= 0.5
    coords *= 2.0
    return coords, {'scale_min': coord_min,
                    'scale_max': coord_max,
                    }

def preprocess_torus_meshes(folder, spiral_len=9):
    file_list = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith('.ply')
    ])
    
    print(f"Loading {len(file_list)} meshes...")

    first_mesh = trimesh.load(file_list[0], process=False)
    num_vertices = len(first_mesh.vertices)
    print(f"Each mesh has {num_vertices} vertices")

    # Build spiral indices from first mesh (template)
    graph = first_mesh.vertex_adjacency_graph
    adj_list = [list(graph.adj[i]) for i in range(num_vertices)]
    spiral_indices = build_spiral(adj_list, spiral_len)
    torch.save(spiral_indices, 'spiral_indices.pt')  #This is the spiral template 
    print("Saved: spiral_indices.pt")

    all_verts = []
    for file in tqdm(file_list):
        mesh = trimesh.load(file, process=False)
        if len(mesh.vertices) != num_vertices:
            raise ValueError(f"{file} has inconsistent vertex count.")
        coords = mesh.vertices
        coords_normed, _ = normalize_n1p1(coords)
        verts = torch.tensor(coords_normed, dtype=torch.float32)
        all_verts.append(verts)

    all_verts = torch.stack(all_verts)  # [N, V, 3]
    torch.save(all_verts, 'torus_vertices.pt') 
    print("Saved: torus_vertices.pt")

class spiralconv_plus(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super().__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1) #9 

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)

def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out

class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = spiralconv_plus(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = spiralconv_plus(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out


class SpiralNet_Plus(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 latent_channels,
                 spiral_indices,
                 down_transform,
                 up_transform):
        super().__init__()
        self.in_channels = in_channels # =3 
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],   #After: [B, 3496, 32]
                                  self.spiral_indices[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],  #After: [B, 3496, 64]
                                  self.spiral_indices[idx]))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], 2*latent_channels)) #Take in: [B, 3496 × 64] => [B, 128]

        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1])) #Take in: [B,128] => [B,3496x64]
        for idx in range(len(out_channels)): 
            if idx == 0:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx - 1],                #After: [B, 3496, 64]
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1])) 
            else:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx], out_channels[-idx - 1], #After: [B, 3496, 32]
                                  self.spiral_indices[-idx - 1]))
        self.de_layers.append(
            spiralconv_plus(out_channels[0], in_channels, self.spiral_indices[0]))  #After: [B, 3496, 3]

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        
        mu = x[:, :self.latent_channels]
        log_var = x[:, self.latent_channels:]

        return mu, log_var
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
        z = mu + sigma * eps, where eps ~ N(0, I)
        sigma = exp(0.5 * logvar)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mu + eps * std
        return sample

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x

    def forward(self, data):
        x = data['x']
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        re_x = out
        return {"re_x": re_x, "mu": mu, "logvar": logvar}
    
class TorusMeshDataset(Dataset):
    def __init__(self, vertex_path='torus_vertices.pt'):
        self.data = torch.load(vertex_path)  # [N, V, 3]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {'x': self.data[idx]}  # for SpiralNet++

def loss_function(original, reconstruction, mu, log_var, beta):
    reconstruction_loss = F.l1_loss(reconstruction, original, reduction='mean')
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    return reconstruction_loss + beta*kld_loss

def save_ply(faces, reconstructed_features, filename):
    """
    Save reconstruction as .ply
    """
    mesh = trimesh.Trimesh(reconstructed_features, faces, process=False)
    mesh.export(filename)
    print(f"Reconstructed torus saved to: {filename}")

if __name__ == '__main__':

    # Pre-Processing
    preprocess_torus_meshes(
        folder=os.path.expanduser('better_training_data'), #replace with your folder with 500 files
        spiral_len=9
    )

    # Training
    BATCH_SIZE = 8
    NUM_EPOCHS = 1000
    LEARNING_RATE = 1e-4
    BETA = 0.0001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Data
    dataset = TorusMeshDataset('torus_vertices.pt')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    spiral_indices = torch.load('spiral_indices.pt').to(DEVICE)
    identity = torch.eye(spiral_indices.size(0)).to_sparse().to(DEVICE)

    # Model
    model = SpiralNet_Plus(
        in_channels=3,
        out_channels=[32, 64],
        latent_channels=128,
        spiral_indices=[spiral_indices, spiral_indices],  # duplicated for 2-layer model
        down_transform=[identity, identity],   # currently no pooling, just identity 
        up_transform=[identity, identity]
    ).to(DEVICE)

    #Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    #Training Loop
    PATIENCE = 10
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            x = batch['x'].to(DEVICE)
            optimizer.zero_grad()
            output = model({'x': x})
            loss = loss_function(x, output['re_x'], output['mu'], output['logvar'], BETA) 

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'spiralnet_torus_autoencoder_best.pt')
            print("Improved. Saved model.")
        else:
            patience_counter += 1
            print(f" No improvement. Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
    
    #Testing
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = TorusMeshDataset('torus_vertices.pt')
    spiral_indices = torch.load('spiral_indices.pt').to(DEVICE)
    identity = torch.eye(spiral_indices.size(0)).to_sparse().to(DEVICE)

    # Load model
    model = SpiralNet_Plus(
        in_channels=3,
        out_channels=[32, 64],
        latent_channels=128,
        spiral_indices=[spiral_indices, spiral_indices],
        down_transform=[identity, identity],
        up_transform=[identity, identity]
    ).to(DEVICE)

    original_tensor = torch.load('torus_vertices.pt')[100]

    model.load_state_dict(torch.load('spiralnet_torus_autoencoder_best.pt'))
    model.eval()

    # Load your template mesh for faces
    mesh = trimesh.load('training_data/template.ply')
    faces = mesh.faces

    x = dataset[100]['x'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        recon_tensor = model({'x': x})['re_x'].squeeze(0).cpu()

    original_vertices = np.array(original_tensor.tolist())
    reconstructed_vertices = np.array(recon_tensor.tolist())

    save_ply(faces, original_vertices, "original_torus_spiralnet.ply")
    save_ply(faces, reconstructed_vertices, "reconstructed_torus_spiralnet.ply")

    # Plot side-by-side
    # p = mp.subplot(original_vertices, faces, s=[1, 2, 0],
    #            shading={"color": "lightblue", "wireframe": True}, title="Original")
    # mp.subplot(reconstructed_vertices, faces, plot=p, s=[1, 2, 1],
    #        shading={"color": "salmon", "wireframe": True}, title="Reconstruction")