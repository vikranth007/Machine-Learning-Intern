import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
from src.dataset import get_dataloaders
from models.model import SimpleCNN, get_resnet18
from src.evaluate import evaluate

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 2
MODEL_PATH = "checkpoints/best_model.pth"
USE_RESNET = False  # Set to True if using ResNet

def main():
    # Load test data
    _, _, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=2
    )
    class_names = test_loader.dataset.classes

    # Load model
    model = get_resnet18(num_classes=NUM_CLASSES) if USE_RESNET else SimpleCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    # Evaluate
    evaluate(model, test_loader, class_names)

if __name__ == "__main__":
    main()
