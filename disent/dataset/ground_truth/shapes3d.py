from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import h5py
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# ========================================================================= #
# shapes3d                                                                  #
# ========================================================================= #


class Shapes3D(object):
    """
    Shapes3D dataset.
    The data set was originally introduced in "Disentangling by Factorising".
    The ground-truth factors of variation are:
    0 - floor color (10 different values)
    1 - wall color (10 different values)
    2 - object color (10 different values)
    3 - object size (8 different values)
    4 - object type (4 different values)
    5 - azimuth (15 different values)

    https://github.com/google-research/disentanglement_lib/blob/adb2772b599ea55c60d58fd4b47dff700ef9233b/disentanglement_lib/data/ground_truth/shapes3d.py
    """

    def __init__(self):
        # with tf.gfile.GFile(SHAPES3D_PATH, "rb") as f:
        #     Data was saved originally using python2, so we need to set the encoding.
            # data = np.load(f, encoding="latin1")

        # images = data["images"]
        # labels = data["labels"]
        # n_samples = np.prod(images.shape[0:6])
        # self.images = (images.reshape([n_samples, 64, 64, 3]).astype(np.float32) / 255.)
        # features = labels.reshape([n_samples, 6])

        # self.factor_indices = list(range(6))
        # self.state_space = SplitDiscreteStateSpace(self.factor_dimensions, self.factor_indices)

        # self.num_total_factors = features.shape[1]
        self.factor_bases = np.prod(self.factor_dimensions) // np.cumprod(self.factor_dimensions)

    @property
    def num_factors(self):
        return len(self.factor_dimensions)

    @property
    def factor_dimensions(self):
        return 10, 10, 10, 8, 4, 15

    @property
    def observation_shape(self):
        return 64, 64, 3

    def sample_factors(self, num):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num)

    def sample_observations_from_factors(self, factors):
        all_factors = self.state_space.sample_all_factors(factors)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return self.images[indices]




class GroundTruthDataset(Dataset, metaclass=ABCMeta):

    def __init__(self):
        assert len(self.FACTOR_NAMES) == len(self.FACTOR_DIMS), 'Dimensionality mismatch of FACTOR_NAMES and FACTOR_DIMS'
        self._num_samples = np.product(self.FACTOR_DIMS)

    @property
    def num_factors(self):
        return len(self.FACTOR_NAMES)

    @property
    def __len__(self):
        return self._num_samples

    @property
    @abstractmethod
    def FACTOR_NAMES(self) -> Tuple[str, ...]: pass

    @property
    @abstractmethod
    def FACTOR_DIMS(self) -> Tuple[int, ...]: pass

    @property
    @abstractmethod
    def OBSERVATION_SHAPE(self) -> Tuple[int, ...]: pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class Shapes3dDataset(GroundTruthDataset):
    """
    3D Shapes Dataset:
    https://github.com/deepmind/3d-shapes
    """

    FACTOR_NAMES = ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')
    FACTOR_DIMS = (10, 10, 10, 8, 4, 15)
    OBSERVATION_SHAPE = (64, 64, 3)

    def __init__(self, shapes_file='3dshapes.h5', transform=None, target_transform=None):
        """
        Args:
            shapes_file (string): Path to the 3D Shapes h5 DATASET file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.transform, self.target_transform = transform, target_transform
        dataset = h5py.File(shapes_file, 'r')
        self.images = dataset['images']      # array shape [480000,64,64,3], uint8 in range(256)
        self.labels = dataset['labels']      # array shape [480000,6], float64

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(self.images[idx])
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    # def get_index(factors):
    #     """ Converts factors to indices in range(num_data)
    #     Args:
    #         factors: np array shape [6,BATCH_SIZE].
    #                 factors[i]=factors[i,:] takes integer values in
    #                 range(FACTOR_DIMS[FACTOR_NAMES[i]]).
    #     Returns:
    #         indices: np array shape [BATCH_SIZE].
    #     """
    #     indices = 0
    #     base = 1
    #     for factor, name in reversed(list(enumerate(FACTOR_NAMES))):
    #         indices += factors[factor] * base
    #         base *= FACTOR_DIMS[name]
    #     return indices

    # def sample_random_batch(BATCH_SIZE):
    #     """ Samples a random batch of images.
    #     Args:
    #         BATCH_SIZE: number of images to sample.
    #     Returns:
    #         batch: images shape [BATCH_SIZE,64,64,3].
    #     """
    #     indices = np.random.choice(n_samples, BATCH_SIZE)
    #     ims = []
    #     for ind in indices:
    #         im = images[ind]
    #         im = np.asarray(im)
    #         ims.append(im)
    #     ims = np.stack(ims, axis=0)
    #     ims = ims / 255. # normalise values to range [0,1]
    #     ims = ims.astype(np.float32)
    #     return ims.reshape([BATCH_SIZE, 64, 64, 3])

    # def sample_batch(BATCH_SIZE, fixed_factor, fixed_factor_value):
    #     """ Samples a batch of images with fixed_factor=fixed_factor_value, but with
    #         the other factors varying randomly.
    #     Args:
    #         BATCH_SIZE: number of images to sample.
    #         fixed_factor: index of factor that is fixed in range(6).
    #         fixed_factor_value: integer value of factor that is fixed
    #         in range(FACTOR_DIMS[FACTOR_NAMES[fixed_factor]]).
    #     Returns:
    #         batch: images shape [BATCH_SIZE,64,64,3]
    #     """
    #     factors = np.zeros([len(FACTOR_NAMES), BATCH_SIZE], dtype=np.int32)
    #     for factor, name in enumerate(FACTOR_NAMES):
    #         num_choices = FACTOR_DIMS[name]
    #         factors[factor] = np.random.choice(num_choices, BATCH_SIZE)
    #     factors[fixed_factor] = fixed_factor_value
    #     indices = get_index(factors)
    #     ims = []
    #     for ind in indices:
    #         im = images[ind]
    #         im = np.asarray(im)
    #         ims.append(im)
    #     ims = np.stack(ims, axis=0)
    #     ims = ims / 255. # normalise values to range [0,1]
    #     ims = ims.astype(np.float32)
    #     return ims.reshape([BATCH_SIZE, 64, 64, 3])


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
