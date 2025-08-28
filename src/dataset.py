import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(image_size=224):
    """
    Define preprocessing transformations for chest X-ray images.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),           # Convert RGB to Grayscale
        transforms.Resize((image_size, image_size)),           # Resize images to a uniform size
        transforms.ToTensor(),                                 # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5])            # Normalize grayscale image
    ])
    return transform

def get_dataloaders(data_dir, batch_size=32, image_size=224, num_workers=2):
    """
    Load train, validation, and test data using torchvision.datasets.ImageFolder.
    
    Parameters:
    - data_dir: path to the 'chest_xray' folder.
    - batch_size: batch size for training.
    - image_size: image resize dimensions.
    - num_workers: number of subprocesses to use for data loading.

    Returns:
    - train_loader, val_loader, test_loader: PyTorch DataLoaders.
    - class_names: List of class labels.
    """

    transform = get_transforms(image_size)

    train_path = os.path.join(data_dir, 'train')
    val_path   = os.path.join(data_dir, 'val')
    test_path  = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset   = datasets.ImageFolder(root=val_path, transform=transform)
    test_dataset  = datasets.ImageFolder(root=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes  # ['NORMAL', 'PNEUMONIA']

    return train_loader, val_loader, test_loader
