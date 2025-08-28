import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add root directory to path

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
from torchvision import transforms

from models.model import SimpleCNN  # ✅ Import works now

# Initialize app and model
app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=2)
model_path = os.path.join(os.path.dirname(__file__), "../checkpoints/best_model.pth")

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded successfully.")
except RuntimeError as e:
    print("❌ Model loading failed:", e)

model.to(device)
model.eval()

CLASS_NAMES = ["Normal", "Pneumonia"]

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # ✅ 1 channel for grayscale
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])             # ✅ normalization for 1 channel
])

# Template and static file setup
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Home page route
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict/", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image"):
        return "Prediction: Invalid file type. Please upload an image."

    img = Image.open(file.file).convert("L")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()

    return f"Prediction: {CLASS_NAMES[pred]}"
