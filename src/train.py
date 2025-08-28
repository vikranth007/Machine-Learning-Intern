import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
from src.dataset import get_dataloaders
from src.utils import save_model
from models.model import SimpleCNN

def train_model(data_dir, epochs, batch_size, lr, num_classes, device):
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size)

    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, "checkpoints/best_model.pth")

    print("Training complete.")

# ✅ This is crucial — define the function, do not put main logic here.
