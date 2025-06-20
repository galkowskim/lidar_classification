from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def compute_mean_std(dataset, batch_size=64):
    """Computes the mean and standard deviation of the dataset.
    Args:
        dataset: The dataset to compute mean and std for.
        batch_size: The batch size to use for DataLoader.
    Returns:
        mean: The mean of the dataset.
        std: The standard deviation of the dataset.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mean = 0.0
    std = 0.0
    nb_samples = 0.0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return mean.tolist(), std.tolist()


def get_dataloaders(config):
    # First, load train set with basic transform for mean/std calculation
    base_transform = transforms.Compose([transforms.Resize(config["img_size"]), transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(f"{config['data_dir']}/train", transform=base_transform)
    test_dataset = datasets.ImageFolder(f"{config['data_dir']}/test", transform=base_transform)

    mean, std = compute_mean_std(train_dataset, batch_size=config["batch_size"])

    train_transform = transforms.Compose(
        [
            transforms.Resize(config["img_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_transform = transforms.Compose(
        [transforms.Resize(config["img_size"]), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    train_dataset = datasets.ImageFolder(f"{config['data_dir']}/train", transform=train_transform)
    test_dataset = datasets.ImageFolder(f"{config['data_dir']}/test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, test_loader
