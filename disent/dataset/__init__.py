
# Make datasets directly accessible
from .ground_truth.base import GroundTruthData, GroundTruthDataset
from .ground_truth.data_cars3d import Cars3dData   # TODO: implement
from .ground_truth.data_dsprites import DSpritesData
from .ground_truth.data_mpi3d import Mpi3dData     # TODO: implement
from .ground_truth.data_norb import SmallNorbData  # TODO: implement
from .ground_truth.data_shapes3d import Shapes3dData
from .ground_truth.data_xygrid import XYData


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


def make_ground_truth_data(name, try_in_memory=True) -> GroundTruthData:
    if '3dshapes' == name:
        data = Shapes3dData()
    elif 'dsprites' == name:
        data = DSpritesData(in_memory=try_in_memory)
    elif 'xygrid' == name:
        data = XYData(grid_size=64, square_size=2)
    elif 'smallnorb' == name:
        data = SmallNorbData()
    elif 'cars3d' == name:
        data = Cars3dData()
    elif 'mpi3d_toy' == name:
        data = Mpi3dData(subset='toy', in_memory=try_in_memory)
    elif 'mpi3d_realistic' == name:
        data = Mpi3dData(subset='realistic', in_memory=try_in_memory)
    elif 'mpi3d_real' == name:
        data = Mpi3dData(subset='real', in_memory=try_in_memory)
    else:
        raise KeyError(f'Unsupported Ground Truth Dataset: {name}')
    return data

def make_ground_truth_data_transform(name):
    import torchvision
    import torch

    transforms = []

    if '3dshapes' == name:  # (1, 64, 64)
        pass
    elif 'dsprites' == name:  # (1, 64, 64)
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),  # add extra channels
        ])
    elif 'xygrid' == name:  # (1, 64, 64)
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0))  # add extra channels
        ])
    elif 'smallnorb' == name:
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),  # add extra channels
        ])
    elif 'cars3d' == name:
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),
            torchvision.transforms.ToTensor(),
        ])
    # elif 'mpi3d_toy' == name:
    #     pass
    # elif 'mpi3d_realistic' == name:
    #     pass
    # elif 'mpi3d_real' == name:
    #     pass
    else:
        raise KeyError(f'Unsupported Ground Truth Dataset: {name}')


def make_ground_truth_dataset(name, try_in_memory=True) -> GroundTruthDataset:
    # all datasets are in the range [0, 1] to correspond
    # to the sigmoid activation function.
    return GroundTruthDataset(
        ground_truth_data=make_ground_truth_data(name, try_in_memory=try_in_memory),
        transform=make_ground_truth_data_transform(name)
    )


def make_paired_dataset(name, k='uniform'):
    from disent.dataset.ground_truth.base import PairedVariationDataset
    dataset = make_ground_truth_dataset(name)
    return PairedVariationDataset(dataset, k=k)


# ========================================================================= #
# strings to datasets only if necessary                                     #
# ========================================================================= #


# TODO: merge into above methods
def as_data(data) -> GroundTruthData:
    if isinstance(data, str):
        data = make_ground_truth_data(data, try_in_memory=False)
    assert isinstance(data, GroundTruthData), 'data not an instance of GroundTruthData'
    return data


# TODO: merge into above methods
def as_dataset(dataset):
    if isinstance(dataset, str):
        dataset = make_ground_truth_dataset(dataset, try_in_memory=False)
    assert isinstance(dataset, GroundTruthDataset), 'dataset not an instance of GroundTruthDataset'
    return dataset


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

