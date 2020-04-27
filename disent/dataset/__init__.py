
# Make datasets directly accessible
from .ground_truth.base import GroundTruthDataset
from .ground_truth.dataset_cars3d import Cars3dDataset   # TODO: implement
from .ground_truth.dataset_dsprites import DSpritesDataset
from .ground_truth.dataset_mpi3d import Mpi3dDataset     # TODO: implement
from .ground_truth.dataset_norb import SmallNorbDataset  # TODO: implement
from .ground_truth.dataset_shapes3d import Shapes3dDataset
from .ground_truth.dataset_xygrid import XYDataset


# ========================================================================= #
# shapes3d                                                                   #
# ========================================================================= #


def split_dataset(dataset, train_ratio=0.8):
    """
    splits a dataset randomly into a training (train_ratio) and test set (1-train_ratio).
    """
    import torch.utils.data
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])


def make_ground_truth_dataset(name, data_dir='data', try_in_memory=True) -> GroundTruthDataset:
    import torchvision
    import torch

    # all datasets are in the range [0, 1] to correspond
    # to the sigmoid activation function.

    if '3dshapes' == name:
        return Shapes3dDataset(data_dir=data_dir, transform=torchvision.transforms.ToTensor())

    elif 'dsprites' == name:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),  # add extra channels
        ])
        return DSpritesDataset(data_dir=data_dir, transform=transforms, in_memory=try_in_memory)

    elif 'xygrid' == name:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0))  # add extra channels
        ])
        return XYDataset(width=64, transform=transforms)

    # elif 'mnist' == name:
    #     # this dataset is quite pointless, just compatible with the others...
    #     from PIL import Image
    #     transforms = torchvision.transforms.Compose([
    #         torchvision.transforms.Resize(64, interpolation=Image.NEAREST),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),  # add extra channels
    #     ])
    #
    #     # MNIST that only returns images, without labels
    #     # torchvision.datasets.MNIST names data folder based off of class name
    #     class MNIST(torchvision.datasets.MNIST):
    #         def __getitem__(self, index):
    #             return super().__getitem__(index)[0]
    #
    #     # return instance
    #     return MNIST(root=data_dir, train=True, transform=transforms, download=True)

    else:
        raise KeyError(f'Unsupported Ground Truth Dataset: {name}')


def make_paired_dataset(name, k='uniform'):
    from disent.dataset.ground_truth.base import PairedVariationDataset
    dataset = make_ground_truth_dataset(name)
    return PairedVariationDataset(dataset, k=k)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
