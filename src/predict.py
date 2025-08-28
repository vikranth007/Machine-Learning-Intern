import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

from models.model import SimpleCNN

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
NUM_CLASSES = 2
MODEL_PATH = "checkpoints/best_model.pth"
USE_RESNET = False
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']  # Make sure this matches your dataset

# Preprocessing for prediction
def preprocess_image(image_path):
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ✅ Convert RGB to grayscale
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

    image = Image.open(image_path).convert("RGB")  # ✅ Convert to RGB
    image = transform(image)
    image = image.unsqueeze(0)  # [1, 3, 224, 224]
    return image.to(DEVICE)

def predict(image_path):
    # Load model
    model = SimpleCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Preprocess image
    image = preprocess_image(image_path)

    # Inference
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    predicted_label = CLASS_NAMES[predicted_class.item()]
    print(f"\n📷 Predicted Label: **{predicted_label}**")

    return predicted_label

if __name__ == "__main__":
    test_image_path = "sample_images/inference.jpg"  # Change this to your test image path
    if not os.path.exists(test_image_path):
        print(f"❌ Image not found: {test_image_path}")
    else:
        predict(test_image_path)
