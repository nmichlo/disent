import gzip
import numpy as np
from disent.data.groundtruth.base import DownloadableGroundTruthData

# ========================================================================= #
# dataset_norb                                                              #
# ========================================================================= #


class SmallNorbData(DownloadableGroundTruthData):
    """
    Small NORB Dataset
    - https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/

    Files:
        - direct hdf5: https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5
        - direct npz: https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz

    # reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/norb.py
    """

    NORB_TYPES = {
        0x1E3D4C55: 'uint8',      # byte matrix
        0x1E3D4C54: 'int32',      # integer matrix
        # 0x1E3D4C56: 'int16',    # short matrix
        # 0x1E3D4C51: 'float32',  # single precision matrix
        # 0x1E3D4C53: 'float64',  # double precision matrix
    }

    # ordered training data (dat, cat, info)
    NORB_TRAIN_URLS = [
        'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz',
        'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz',
        'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz',
    ]

    # ordered testing data (dat, cat, info)
    NORB_TEST_URLS = [
        'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz',
        'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz',
        'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz',
    ]

    dataset_urls = [*NORB_TRAIN_URLS, *NORB_TEST_URLS]

    # TODO: add ability to randomly sample the instance so
    #       that this corresponds to disentanglement_lib
    factor_names = ('category', 'instance', 'elevation', 'azimuth', 'lighting_condition')
    factor_sizes = (5, 5, 9, 18, 6)  # TOTAL: 24300
    observation_shape = (96, 96, 1)

    def __init__(self, data_dir='data/dataset/smallnorb', force_download=False, is_test=False):
        super().__init__(data_dir=data_dir, force_download=force_download)
        assert not is_test, 'Test set not yet supported'

        if not hasattr(self.__class__, '_DATA'):
            images, features = self._read_norb_set(is_test)
            # sort by features
            indices = np.lexsort(features[:, [4, 3, 2, 1, 0]].T)
            # store data on class
            self.__class__._DATA = images[indices]

    def __getitem__(self, idx):
        return self.__class__._DATA[idx]

    def _read_norb_set(self, is_test):
        # get file data corresponding to urls
        dat, cat, info = [
            self._read_norb_file(self.dataset_paths[self.dataset_urls.index(url)])
            for url in (self.NORB_TEST_URLS if is_test else self.NORB_TRAIN_URLS)
        ]
        features = np.column_stack([cat, info])  # append info to categories
        features[:, 3] = features[:, 3] / 2  # azimuth values are even numbers, convert to indices
        images = dat[:, 0]  # images are in pairs, we only extract the first one of each
        return images, features

    @staticmethod
    def _read_norb_file(filename):
        """Read the norb data from the compressed file - modified from disentanglement_lib"""
        with gzip.open(filename, "rb") as f:
            s = f.read()
            magic = int(np.frombuffer(s, "int32", 1, 0))
            ndim = int(np.frombuffer(s, "int32", 1, 4))
            eff_dim = max(3, ndim)  # stores minimum of 3 dimensions even for 1D array
            dims = np.frombuffer(s, "int32", eff_dim, 8)[0:ndim]
            data = np.frombuffer(s, SmallNorbData.NORB_TYPES[magic], offset=8 + eff_dim * 4)
            data = data.reshape(tuple(dims))
        return data


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
