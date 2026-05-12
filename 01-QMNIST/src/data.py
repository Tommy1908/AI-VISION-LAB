from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple

def get_dataloaders() -> Tuple[DataLoader,DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.QMNIST('data', train=True, download = True, transform=transform)
    test_set = datasets.QMNIST('data', train=False, download = True, transform=transform)

    return (DataLoader(train_set, batch_size = 64), DataLoader(test_set, batch_size = 64))