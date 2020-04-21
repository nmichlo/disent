import h5py
import torch
from PIL import Image
from disent.dataset.util import GroundTruthDataset, PairedVariationDataset


# ========================================================================= #
# shapes3d                                                                  #
# ========================================================================= #


class Shapes3dDataset(GroundTruthDataset):
    """
    3D Shapes Dataset:
    https://github.com/deepmind/3d-shapes
    """

    factor_names = ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')
    factor_sizes = (10, 10, 10, 8, 4, 15)
    observation_shape = (64, 64, 3)

    def __init__(self, shapes_file='data/processed/3dshapes_gzip4_32-64-3_4096__4700.43eps.h5', transform=None):
        # TODO: add automatic conversion on first run

        super().__init__()
        self.transform = transform
        self.hdf5file = shapes_file

        # dataset = h5py.File(shapes_file, 'r')
        # self.images = dataset['images']      # array shape [480000,64,64,3], uint8 in range(256)
        # self.labels = dataset['labels']      # array shape [480000,6], float64
        # assert self.images.shape == (480000, 64, 64, 3)
        # assert self.labels.shape == (480000, 6)

    def __getitem__(self, indices):
        if torch.is_tensor(indices):
            indices = indices.tolist()

        with h5py.File(self.hdf5file, 'r') as db:
            image = db['images'][indices]

        # https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
        # PIL Image so that this is consistent with other datasets
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    import numpy as np

    dataset = Shapes3dDataset()
    pair_dataset = PairedVariationDataset(dataset, k='uniform')

    for i in range(10):
        idx = np.random.randint(len(dataset))
        a, b = pair_dataset.sample_pair_factors(idx)
        print(all(dataset.idx_to_pos(idx) == a), '|', a, '&', b, ':', [int(v) for v in (a == b)])
        a, b = dataset.pos_to_idx([a, b])
        print(a, b)
        dataset[a], dataset[b]
