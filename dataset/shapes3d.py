import h5py
import torch
from PIL import Image
from torch.utils.data import Dataset


# ========================================================================= #
# shapes3d                                                                   #
# ========================================================================= #


class Shapes3dDataset(Dataset):
    """3D Shapes Dataset"""

    _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8, 'shape': 4, 'orientation': 15}

    def __init__(self, shapes_file='3dshapes.h5', transform=None, target_transform=None):
        """
        Args:
            shapes_file (string): Path to the 3D Shapes h5 DATASET file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.target_transform = target_transform

        dataset = h5py.File(shapes_file, 'r')

        self.images = dataset['images']      # array shape [480000,64,64,3], uint8 in range(256)
        self.labels = dataset['labels']      # array shape [480000,6], float64
        self.image_shape = self.images.shape[1:]  # [64,64,3]
        self.label_shape = self.labels.shape[1:]  # [6]
        self.n_samples = self.labels.shape[0]     # 10*10*10*8*4*15=480000

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, label = self.images[idx], self.labels[idx]

        # https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    # def get_index(factors):
    #     """ Converts factors to indices in range(num_data)
    #     Args:
    #         factors: np array shape [6,BATCH_SIZE].
    #                 factors[i]=factors[i,:] takes integer values in
    #                 range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).
    #     Returns:
    #         indices: np array shape [BATCH_SIZE].
    #     """
    #     indices = 0
    #     base = 1
    #     for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
    #         indices += factors[factor] * base
    #         base *= _NUM_VALUES_PER_FACTOR[name]
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
    #         in range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[fixed_factor]]).
    #     Returns:
    #         batch: images shape [BATCH_SIZE,64,64,3]
    #     """
    #     factors = np.zeros([len(_FACTORS_IN_ORDER), BATCH_SIZE], dtype=np.int32)
    #     for factor, name in enumerate(_FACTORS_IN_ORDER):
    #         num_choices = _NUM_VALUES_PER_FACTOR[name]
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
