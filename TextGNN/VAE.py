import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import string
from collections import Counter
import glob

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char if char not in string.punctuation else ' ' for char in text])
    words = text.split()
    
    if len(words) == 0:
        raise ValueError("Input text is empty after preprocessing.")
    
    word_counts = Counter(words)
    word_to_idx = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    return words, word_to_idx, idx_to_word


def text_to_graph(words, word_to_idx):
    nodes = torch.tensor([word_to_idx[word] for word in words], dtype=torch.long)
    edge_list = [(word_to_idx[words[i]], word_to_idx[words[i + 1]]) for i in range(len(words) - 1)]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=nodes, edge_index=edge_index)


class VAE(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, vocab_size):
        super(VAE, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, num_features)
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, vocab_size)
    
    def encode(self, x, edge_index):
        x = self.embeddings(x)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = torch.relu(self.fc3(z))
        return self.fc4(z)

    def forward(self, data):
        mu, logvar = self.encode(data.x, data.edge_index)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def train_model(text_files, template_text, word_to_idx, idx_to_word, epochs=20, latent_dim=16):
    template_words, _, _ = preprocess_text(template_text)
    template_graph = text_to_graph(template_words, word_to_idx)
    
    model = VAE(num_features=32, hidden_dim=64, latent_dim=latent_dim, vocab_size=len(word_to_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        loss = 0
        for file in text_files:
            words, _, _ = preprocess_text(open(file, 'r').read())
            data = text_to_graph(words, word_to_idx)
            recon_x, mu, logvar = model(data)
            loss += F.cross_entropy(recon_x, data.x) + (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

def generate_text(model, idx_to_word, template_words, capitalize_index=None):
    model.eval()
    
    generated_words = template_words[:]
    if capitalize_index is not None and 0 <= capitalize_index < len(generated_words):
        generated_words[capitalize_index] = generated_words[capitalize_index].upper()
    
    return ' '.join(generated_words)

def process_text_files(template_file, text_files, output_file, capitalize_index=None):
    template_text = open(template_file, 'r').read()
    words, word_to_idx, idx_to_word = preprocess_text(template_text)
    model = train_model(text_files, template_text, word_to_idx, idx_to_word)
    generated_text = generate_text(model, idx_to_word, words, capitalize_index=capitalize_index)
    open(output_file, 'w').write(generated_text)
    print(f"Generated text saved to {output_file}")

template_file = 'template.txt'
text_files = sorted(glob.glob('file_*.txt')) 
output_file = 'generated_output.txt'
process_text_files(template_file, text_files, output_file, capitalize_index=4)
