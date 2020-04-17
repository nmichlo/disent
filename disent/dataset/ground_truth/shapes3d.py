from typing import Optional, Tuple

import h5py
import torch
from PIL import Image
from torch.utils.data import Dataset

from disent.dataset.ground_truth.ground_truth import GroundTruthData

# ========================================================================= #
# shapes3d                                                                  #
# ========================================================================= #


class Shapes3dData(GroundTruthData, Dataset):
    """
    3D Shapes Dataset:
    https://github.com/deepmind/3d-shapes
    """

    factor_names = ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')
    factor_sizes = (10, 10, 10, 8, 4, 15)
    observation_shape = (64, 64, 3)
    used_factors = None

    def __init__(self, shapes_file='3dshapes.h5', transform=None):
        super().__init__()
        self.transform = transform

        dataset = h5py.File(shapes_file, 'r')

        self.images = dataset['images']      # array shape [480000,64,64,3], uint8 in range(256)
        self.labels = dataset['labels']      # array shape [480000,6], float64
        assert self.images.shape == (480000, 64, 64, 3)
        assert self.labels.shape == (480000, 6)

    def get_observations_from_indices(self, indices):
        return self.images[indices]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(self.images[idx])
        if self.transform:
            image = self.transform(image)

        # label = self.labels[idx]
        # if self.target_transform:
        #     label = self.target_transform(label)

        return image


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
