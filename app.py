import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ------------------------------
# 1. Model Setup
# ------------------------------
CHECKPOINT_PATH = "checkpoints/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 👇 Same model architecture as used in train.py
def load_model(num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ------------------------------
# 2. Image Transform
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------------------
# 3. Streamlit UI
# ------------------------------
st.title("🩻 Lung Disease Detector")
st.write("Upload a chest X-ray image to classify lung disease")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Load model dynamically (number of classes = your dataset classes)
    # ⚠️ Replace with your actual class names
    class_names = ["Normal", "Pneumonia", "COVID-19"]  
    model = load_model(len(class_names))

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        result = class_names[predicted.item()]

    st.success(f"🎯 Prediction: **{result}**")
