
# Make data directly accessible
from .ground_truth_data.base_data import GroundTruthData
from .ground_truth_data.data_cars3d import Cars3dData
from .ground_truth_data.data_dsprites import DSpritesData
from .ground_truth_data.data_mpi3d import Mpi3dData
from .ground_truth_data.data_norb import SmallNorbData
from .ground_truth_data.data_shapes3d import Shapes3dData
from .ground_truth_data.data_xygrid import XYGridData
from .ground_truth_data.data_xysquares import XYSquaresData

# make datasets directly accessible
from .single import GroundTruthDataset
from .pairs import PairedVariationDataset, RandomPairDataset
from .triplets import SupervisedTripletDataset

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


# ========================================================================= #
# strings to datasets only if necessary                                     #
# ========================================================================= #


# TODO: merge into above methods
def DEPRICATED_as_data(data) -> GroundTruthData:
    raise NotImplementedError('convert to hydra-config')
    # if isinstance(data, str):
    #     data = DEPRICATED_make_ground_truth_data(data, try_in_memory=False)
    # assert isinstance(data, GroundTruthData), 'data not an instance of GroundTruthData'
    # return data


# TODO: merge into above methods
def DEPRICATED_as_dataset(dataset):
    raise NotImplementedError('convert to hydra-config')
    # if isinstance(dataset, str):
    #     dataset = DEPRICATED_make_ground_truth_dataset(dataset, try_in_memory=False)
    # assert isinstance(dataset, GroundTruthDataset), 'dataset not an instance of GroundTruthDataset'
    # return dataset


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

