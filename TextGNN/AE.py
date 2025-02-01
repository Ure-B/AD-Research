import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from collections import Counter
import string

# Sample Text Preprocessing
def preprocess_text(text):
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
    
    return words, word_to_idx, idx_to_word

# Convert text to graph data
def text_to_graph(text, word_to_idx):
    words, _, _ = preprocess_text(text)
    
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
    
    return Data(x=nodes, edge_index=edge_index)

# Graph Neural Network model with PyG
class GNN(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim, vocab_size):
        super(GNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, num_features)  # Embedding layer for word features
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, vocab_size)  # Output layer for word prediction

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embeddings(x)  # Convert word indices to embeddings
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)  # Predict the word indices
        return x

# Training the model
def train_model(text, word_to_idx, idx_to_word, epochs=20):
    data = text_to_graph(text, word_to_idx)
    
    # Initialize model
    model = GNN(num_features=32, hidden_dim=64, output_dim=32, vocab_size=len(word_to_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data)
        
        # Reconstruction task: predict the word indices for each node
        loss = criterion(out, data.x)  # The target is the word indices (data.x)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model

# Reconstruct the text from the embeddings
def reconstruct_text(model, idx_to_word, data):
    model.eval()
    with torch.no_grad():
        embeddings = model.embeddings(data.x)
        x = model.conv1(embeddings, data.edge_index)
        x = torch.relu(x)
        x = model.conv2(x, data.edge_index)
        x = torch.relu(x)
        output = model.fc(x)
        
        # Get the word indices with the highest probability
        reconstructed_indices = torch.argmax(output, dim=1)
        words = [idx_to_word[idx.item()] for idx in reconstructed_indices]
        return ' '.join(words)

# Function to read the input file and save the reconstructed output
def process_text_file(input_file, output_file):
    # Read input file
    with open(input_file, 'r') as f:
        text = f.read()
    
    if not text.strip():
        raise ValueError("Input text file is empty.")
    
    # Preprocess the text and train the model
    words, word_to_idx, idx_to_word = preprocess_text(text)
    model = train_model(text, word_to_idx, idx_to_word)
    
    # Convert the text to graph data
    data = text_to_graph(text, word_to_idx)
    
    # Reconstruct the text
    reconstructed_text = reconstruct_text(model, idx_to_word, data)
    
    # Write the reconstructed text to the output file
    with open(output_file, 'w') as f:
        f.write(reconstructed_text)
    print(f"Reconstructed text saved to {output_file}")

# Example Usage
input_file = 'input.txt'  # Specify the input .txt file here
output_file = 'reconstructed_output.txt'  # Specify the output .txt file here

process_text_file(input_file, output_file)
