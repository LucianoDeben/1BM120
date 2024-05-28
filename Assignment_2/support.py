from torchvision import datasets, transforms
import torch


def load_dataset():
    
    # # Set the mean and std for the dataset to 0 mean and 1 std
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]

    transform = transforms.Compose(
        [
            transforms.Resize([105, 78]),
            transforms.CenterCrop(size=[60, 30]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_path = r"WF-data/train"
    train_dataset = datasets.ImageFolder(
        train_path, transform=transform, target_transform=None
    )

    test_path = r"WF-data/test"
    test_dataset = datasets.ImageFolder(
        test_path, transform=transform, target_transform=None
    )

    return train_dataset, test_dataset

