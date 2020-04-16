
# ========================================================================= #
# make - optimizers                                                         #
# ========================================================================= #


def make_optimizer(optimizer, parameters, lr=0.001):
    # SGD
    if optimizer == 'sgd':
        import torch.optim
        return torch.optim.SGD(parameters, lr=lr)
    # ADAM
    elif optimizer == 'adam':
        import torch.optim
        return torch.optim.Adam(parameters, lr=lr)
    # RADAM
    elif optimizer == 'radam':
        import torch_optimizer
        return torch_optimizer.RAdam(parameters, lr=lr)
    # UNKNOWN
    else:
        raise KeyError(f'Unsupported Optimizer: {optimizer}')


# ========================================================================= #
# make - datasets                                                           #
# ========================================================================= #


def make_datasets(dataset):
    import torchvision.transforms
    from disent.dataset import split_dataset

    # MNIST
    if dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=False)
    # 3D SHAPES
    elif dataset == '3dshapes':
        from disent.dataset import Shapes3dDataset
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
        from disent.dataset import XYDataset
        train_dataset, test_dataset = split_dataset(
            dataset=XYDataset(size=28, arch='full', transform=torchvision.transforms.ToTensor()),
            train_ratio=0.8
        )
    # UNKNOWN
    else:
        raise KeyError(f'Unknown dataset: {dataset}')

    return train_dataset, test_dataset


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
