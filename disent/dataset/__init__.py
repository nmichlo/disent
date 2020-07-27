
# Make datasets directly accessible
from .ground_truth.base import GroundTruthData, GroundTruthDataset
from .ground_truth.data_cars3d import Cars3dData
from .ground_truth.data_dsprites import DSpritesData
from .ground_truth.data_mpi3d import Mpi3dData
from .ground_truth.data_norb import SmallNorbData
from .ground_truth.data_shapes3d import Shapes3dData
from .ground_truth.data_xygrid import XYData
from .ground_truth.data_xyscalegrid import XYScaleData
from .ground_truth.data_xyscalecolorgrid import XYScaleColorData


# ========================================================================= #
# util                                                                      #
# ========================================================================= #


def split_dataset(dataset, train_ratio=0.8):
    """
    splits a dataset randomly into a training (train_ratio) and test set (1-train_ratio).
    """
    import torch.utils.data
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])


def DEPRICATED_make_ground_truth_dataset(name, try_in_memory=True) -> GroundTruthDataset:
    raise NotImplementedError('Use hydra-config instead')
    # all datasets are in the range [0, 1] to correspond
    # to the sigmoid activation function.
    # return GroundTruthDataset(
    #     ground_truth_data=DEPRICATED_make_ground_truth_data(name, try_in_memory=try_in_memory),
    #     transform=DEPRICATED_make_ground_truth_data_transform(name)
    # )

# ========================================================================= #
# strings to datasets only if necessary                                     #
# ========================================================================= #


# TODO: merge into above methods
def DEPRICATED_as_data(data) -> GroundTruthData:
    raise NotImplementedError('Use hydra-config instead')
    # if isinstance(data, str):
    #     data = DEPRICATED_make_ground_truth_data(data, try_in_memory=False)
    # assert isinstance(data, GroundTruthData), 'data not an instance of GroundTruthData'
    # return data


# TODO: merge into above methods
def DEPRICATED_as_dataset(dataset):
    raise NotImplementedError('Use hydra-config instead')
    # if isinstance(dataset, str):
    #     dataset = DEPRICATED_make_ground_truth_dataset(dataset, try_in_memory=False)
    # assert isinstance(dataset, GroundTruthDataset), 'dataset not an instance of GroundTruthDataset'
    # return dataset


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

