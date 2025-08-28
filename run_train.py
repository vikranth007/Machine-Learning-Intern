import sys
import os

# Ensure 'src' folder is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.train import train_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lung Disease Detection Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory of train/val/test folders")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_classes=args.num_classes,
        device=args.device
    )
