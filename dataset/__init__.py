
# Make datasets directly accessible
from .shapes3d import Shapes3dDataset
from .xygrid import XYDataset


def split_dataset(dataset, train_ratio=0.8):
    import torch.utils.data
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])


def make_datasets(dataset):
    import torchvision

    # MNIST
    if dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=False)

    # 3D SHAPES
    elif dataset == '3dshapes':
        shapes_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(28),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
        ])
        train_dataset, test_dataset = split_dataset(
            dataset=Shapes3dDataset('data/3dshapes.h5', transform=shapes_transform),
            train_ratio=0.8
        )

    # XYGRID
    elif dataset == 'xygrid':
        train_dataset, test_dataset = split_dataset(
            dataset=XYDataset(size=28, arch='full', transform=torchvision.transforms.ToTensor()),
            train_ratio=0.8
        )

    # UNKNOWN
    else:
        raise KeyError(f'Unknown dataset: {dataset}')

    return train_dataset, test_dataset