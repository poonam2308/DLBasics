import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from pathlib import Path

from transformers.vision_transformer.train import train, evaluate
from transformers.vision_transformer.vision_transformer import VisionTransformer


def main():
    # Paths
    data_path = Path("data")
    model_path = Path("model")

    # Create directories if they don't exist
    data_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Hyperparameters
    patch_size = 4
    embed_dim = 256
    num_heads = 8
    num_layers = 6
    ff_dim = 512
    num_classes = 10
    lr = 1e-4
    num_epochs = 10

    # Model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(patch_size, embed_dim, num_heads, num_layers, ff_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save model
    torch.save(model.state_dict(), model_path / "vit.pth")

if __name__ == "__main__":
    main()
