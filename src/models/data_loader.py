import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    try:
        full_dataset = datasets.Food101(
            root=data_dir,
            split='train',
            download=False,  # Manual download
            transform=train_transform
        )
    except RuntimeError as e:
        raise RuntimeError(f"Dataset not found in {data_dir}. Please ensure: \n"
                         "1. food-101.tar.gz is extracted in the data directory\n"
                         "2. Directory structure: data/food-101/[images|meta]") from e

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Test dataset
    test_dataset = datasets.Food101(
        root=data_dir,
        split='test',
        download=False,
        transform=test_transform
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, full_dataset.classes