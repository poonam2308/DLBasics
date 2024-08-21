# Example data
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from transformers.seq_transformer.custom_dataset import ToyDataset
from transformers.seq_transformer.transformer import Transformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


src_data = [[1, 2, 3, 4], [5, 6, 7, 8]]
tgt_data = [[2, 3, 4, 0], [6, 7, 8, 0]]  # Target sequences with padding (0)

dataset = ToyDataset(src_data, tgt_data)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

d_model = 64        # Dimension of the model
n_heads = 8         # Number of attention heads
d_ff = 256          # Dimension of the feed-forward network
n_layers = 4        # Number of encoder/decoder layers
vocab_size = 10     # Vocabulary size (including padding)
max_len = 10        # Maximum length of input sequences

# Initialize the transformer model
model = Transformer(d_model, n_heads, d_ff, n_layers, vocab_size, max_len)
# Move model to device (CPU/GPU)
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token in loss calculation
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train(model, data_loader, criterion, optimizer, device):
    model.train()
    for src, tgt in data_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Create target padding mask and look-ahead mask
        src_mask = None  # Implement source mask if needed
        tgt_mask = None  # Implement target mask if needed

        if src_mask is not None:
            src_mask = src_mask.to(device)  # Move src_mask to the same device as src
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device)  # Move tgt_mask to the same device as tgt

        optimizer.zero_grad()
        output = model(src, tgt, src_mask, tgt_mask)

        # Compute loss (assuming tgt is shifted right by one position)
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")


# Training the model
num_epochs = 5
for epoch in range(num_epochs):
    train(model, data_loader, criterion, optimizer, device)


def evaluate(model, src, device):
    model.eval()
    with torch.no_grad():
        src = torch.tensor(src).unsqueeze(0).to(device)  # Ensure src is on the correct device
        tgt = torch.zeros_like(src).to(device)  # Ensure tgt is on the correct device

        # Implement target generation and mask creation
        src_mask = None  # Implement source mask if needed
        tgt_mask = None  # Implement target mask if needed

        if src_mask is not None:
            src_mask = src_mask.to(device)  # Move src_mask to the same device as src
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device)  # Move tgt_mask to the same device as tgt

        output = model(src, tgt, src_mask, tgt_mask)
        predicted = output.argmax(dim=-1)
        return predicted



# Example evaluation
src_example = [1, 2, 3, 4]  # Example source sequence
predicted = evaluate(model, src_example, device)
print(f"Predicted sequence: {predicted}")



