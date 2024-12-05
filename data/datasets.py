# data/datasets.py

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataloaders(batch_size=64, validation_split=0.2, shuffle=True, num_workers=2, download=True, normalize=True):
    # Define transformations
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010]),
        ])
        denorm_params = {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010],
        }
    else:
        transform = transforms.ToTensor()
        denorm_params = None

    # Download and load the training dataset
    full_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=download, transform=transform)

    # Split the dataset into training and validation sets
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)

    # Load the test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=download, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, denorm_params
