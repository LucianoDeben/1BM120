from torchvision import datasets, transforms
import torch


def load_dataset():
    
    mean = [0.3699, 0.2424, 0.2564]
    std = [0.0888, 0.1932, 0.1779]

    transform = transforms.Compose(
        [
            transforms.Resize([105, 78]),
            transforms.CenterCrop(size=[60, 30]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
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

def calculate_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    mean = 0.0
    for images, _ in dataloader:
        batch_samples = images.size(0) 
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dataloader.dataset)

    var = 0.0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(dataloader.dataset)*images.size(2)))

    return mean, std

