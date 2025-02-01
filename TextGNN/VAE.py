import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import string
from collections import Counter

# Sample Text Preprocessing
def preprocess_text(text, capitalize_word=None):
    # Remove punctuation and tokenize
    text = text.lower()
    text = ''.join([char if char not in string.punctuation else ' ' for char in text])
    words = text.split()
    
    if len(words) == 0:
        raise ValueError("Input text is empty after preprocessing.")
    
    # Create word-to-index mapping
    word_counts = Counter(words)
    word_to_idx = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # If we have a word to capitalize, mark it (otherwise, None)
    capitalize_idx = None
    if capitalize_word:
        capitalize_idx = word_to_idx.get(capitalize_word.lower())  # Get the index of the word to capitalize
    
    return words, word_to_idx, idx_to_word, capitalize_idx

# Convert text to graph data
def text_to_graph(text, word_to_idx, capitalize_idx=None):
    words, _, _, _ = preprocess_text(text, capitalize_word=None)
    
    # Create nodes (one for each word)
    nodes = torch.tensor([word_to_idx[word] for word in words], dtype=torch.long)
    
    if len(nodes) == 0:
        raise ValueError("No nodes (words) were created from the text.")
    
    # Create edges based on word co-occurrence (window size of 1 for simplicity)
    edge_list = []
    for i in range(len(words) - 1):
        edge_list.append((word_to_idx[words[i]], word_to_idx[words[i + 1]]))
    
    if len(edge_list) == 0:
        raise ValueError("No edges were created. The input might be too small.")
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    return Data(x=nodes, edge_index=edge_index, capitalize_idx=capitalize_idx)

# Variational Autoencoder Model
class VAE(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, vocab_size):
        super(VAE, self).__init__()
        
        # Encoder
        self.embeddings = nn.Embedding(vocab_size, num_features)
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, vocab_size)  # Output layer for word prediction
        self.fc_capitalize = nn.Linear(latent_dim, 1)  # Capitalization decision (1 or 0)

    def encode(self, x, edge_index):
        x = self.embeddings(x)  # Convert word indices to embeddings
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = torch.relu(self.fc3(z))
        return self.fc4(z)  # Output raw logits for word prediction

    def decide_capitalization(self, z):
        # Output 1 or 0, where 1 means capitalize
        return torch.sigmoid(self.fc_capitalize(z))

    def forward(self, data):
        mu, logvar = self.encode(data.x, data.edge_index)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        capitalize_decision = self.decide_capitalization(z)
        return recon_x, capitalize_decision, mu, logvar

# VAE Loss Function (Reconstruction + KL Divergence)
def vae_loss(recon_x, x, capitalize_decision, target_capitalized, mu, logvar):
    # Reconstruction loss (cross-entropy)
    recon_loss = F.cross_entropy(recon_x, x, reduction='sum')
    
    # Capitalization loss (binary cross-entropy)
    capitalize_loss = F.binary_cross_entropy(capitalize_decision.squeeze(), target_capitalized.float())
    
    # KL divergence (regularization)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss + capitalize_loss

# Training the model
def train_model(text, word_to_idx, idx_to_word, capitalize_idx=None, epochs=20, latent_dim=16):
    data = text_to_graph(text, word_to_idx, capitalize_idx)
    
    # Initialize model
    model = VAE(num_features=32, hidden_dim=64, latent_dim=latent_dim, vocab_size=len(word_to_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # For simplicity, assume that the word to capitalize is known
        target_capitalized = torch.zeros_like(data.x)  # No word is capitalized initially
        if data.capitalize_idx is not None:
            target_capitalized[data.capitalize_idx] = 1  # Capitalize the designated word
        
        # Forward pass
        recon_x, capitalize_decision, mu, logvar = model(data)
        
        # Compute loss
        loss = vae_loss(recon_x, data.x, capitalize_decision, target_capitalized, mu, logvar)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model

# Reconstruct the text from the embeddings (decode latent code)
def generate_text(model, idx_to_word, vocab_size, capitalize_idx=None, max_len=50):
    model.eval()
    
    # Start with a fixed latent vector (no randomness for generation)
    z = torch.randn(1, model.fc_mu.out_features)  # Fixed random latent vector
    
    generated_words = []
    for _ in range(max_len):
        with torch.no_grad():
            output = model.decode(z)
            next_word_idx = torch.argmax(output, dim=1).item()
            generated_word = idx_to_word[next_word_idx]
            
            # If we know the index to capitalize, capitalize the word at that position
            if capitalize_idx is not None and len(generated_words) == capitalize_idx:
                generated_word = generated_word.capitalize()  # Capitalize designated word
            
            generated_words.append(generated_word)
            
            # Update z to reflect the generation of the next word
            z = torch.randn(1, model.fc_mu.out_features)  # Keep z fixed for consistency

    return ' '.join(generated_words)

# Function to read the input file and save the reconstructed output
def process_text_file(input_file, output_file, capitalize_word=None):
    # Read input file
    with open(input_file, 'r') as f:
        text = f.read()
    
    if not text.strip():
        raise ValueError("Input text file is empty.")
    
    # Preprocess the text and train the model
    words, word_to_idx, idx_to_word, capitalize_idx = preprocess_text(text, capitalize_word)
    model = train_model(text, word_to_idx, idx_to_word, capitalize_idx)
    
    # Generate new text
    generated_text = generate_text(model, idx_to_word, vocab_size=len(word_to_idx), capitalize_idx=capitalize_idx)
    
    # Write the generated text to the output file
    with open(output_file, 'w') as f:
        f.write(generated_text)
    print(f"Generated text saved to {output_file}")

# Example Usage
input_file = 'input.txt'  # Specify the input .txt file here (for training)
output_file = 'generated_output.txt'  # Specify the output .txt file here

process_text_file(input_file, output_file, capitalize_word="two")
