# Lung Disease Detector

This repository contains a modular PyTorch pipeline to train, evaluate, and deploy models for lung disease detection using chest X-ray images.

## Project Structure

- `data/` - Training, validation, and test images organized by class folders.
- `models/` - Model definitions (Custom CNN and ResNet18).
- `src/` - Core project logic for dataset loading, training, evaluation, prediction, and utilities.
- `checkpoints/` - Folder to save trained model weights.
- `notebooks/` - Jupyter notebooks for exploratory data analysis.
- `app/` - FastAPI backend for model inference serving.
- `train.py`, `eval.py` - Entrypoint scripts for training and evaluation.

## Quickstart

1. Install dependencies:  
   `pip install -r requirements.txt`

2. Prepare data folders inside `data/` as `train/`, `val/`, and `test/` grouped by class labels.

3. Train the model:  
   `python train.py --data_dir data/ --epochs 10 --batch_size 32`

4. Evaluate the model:  
   `python eval.py --data_dir data/`

5. Predict single images:  
   `python src/predict.py --img path/to/image.jpg`

6. Run API server:  
