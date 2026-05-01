import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitCNN


def build_loaders(data_dir, batch_size):
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=12,
                translate=(0.10, 0.10),
                scale=(0.85, 1.15),
                shear=8,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Batch {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    return running_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            prediction = model(data).argmax(dim=1)
            total += target.size(0)
            correct += (prediction == target).sum().item()

    return 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train a CNN digit classifier on MNIST.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-path", default="mnist_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = build_loaders(args.data_dir, args.batch_size)

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_accuracy = 0.0
    model_path = Path(args.model_path)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.epochs,
        )
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch} finished | Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "DigitCNN",
                    "accuracy": best_accuracy,
                    "epochs": epoch,
                },
                model_path,
            )
            print(f"Saved best model to {model_path}")

    print(f"\nBest test accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
