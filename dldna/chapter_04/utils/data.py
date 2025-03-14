
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple

def get_device() -> str:
    """Returns the device to use ('cuda' if GPU is available, otherwise 'cpu')."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_data_loaders(
    dataset: str = "FashionMNIST",
    batch_size: int = 128
) -> Tuple[DataLoader, DataLoader]:
    """Creates and returns data loaders for training and testing.

    Args:
        dataset: Name of the dataset ("FashionMNIST" or "CIFAR100").
        batch_size: Batch size for the data loaders.

    Returns:
        Tuple: Training data loader, testing data loader.

    Raises:
        ValueError: If an unsupported dataset is specified.
    """
    if dataset == "FashionMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_class = datasets.FashionMNIST

    elif dataset == "CIFAR100":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_class = datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_dataset = dataset_class(
        root="data", train=True, download=True, transform=transform)
    test_dataset = dataset_class(
        root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def get_dataset(
    dataset: str = "FashionMNIST"
) -> Tuple[Dataset, Dataset]:
    """Creates and returns the training and testing datasets.

    Args:
        dataset: Name of the dataset ("FashionMNIST" or "CIFAR100").

    Returns:
        Tuple: Training dataset, testing dataset.

    Raises:
      ValueError: If an unsupported dataset name is provided.
    """
    if dataset == "FashionMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_class = datasets.FashionMNIST

    elif dataset == "CIFAR100":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_class = datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_dataset = dataset_class(
        root="data", train=True, download=True, transform=transform)
    test_dataset = dataset_class(
        root="data", train=False, download=True, transform=transform)

    return train_dataset, test_dataset