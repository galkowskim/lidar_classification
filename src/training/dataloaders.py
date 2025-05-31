import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(config):
    transform = transforms.Compose(
        [transforms.Resize(config["img_size"]), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    dataset = datasets.ImageFolder(config["data_dir"], transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader
