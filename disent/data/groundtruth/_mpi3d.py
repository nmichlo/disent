#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import logging
import numpy as np
from disent.data.groundtruth.base import DownloadableGroundTruthData

log = logging.getLogger(__name__)

# ========================================================================= #
# mpi3d                                                                     #
# ========================================================================= #


class Mpi3dData(DownloadableGroundTruthData):
    """
    MPI3D Dataset
    - https://github.com/rr-learning/disentanglement_dataset

    Files:
        - toy:       https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz
        - realistic: https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_realistic.npz
        - real:      https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz

    reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/mpi3d.py
    """

    URLS = {
        'toy': 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz',
        'realistic': 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_realistic.npz',
        'real': 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz',
    }

    factor_names = ("object_color", "object_shape", "object_size", "camera_height", "background_color", "first_dof", "second_dof")
    factor_sizes = (4, 4, 2, 3, 3, 40, 40)  # TOTAL: 460800
    observation_shape = (64, 64, 3)

    @property
    def dataset_urls(self):
        return [Mpi3dData.URLS[self.subset]]

    def __init__(self, data_dir='data/dataset/mpi3d', force_download=False, subset='realistic', in_memory=False):
        # check subset
        assert subset in Mpi3dData.URLS, f'Invalid subset: {subset=} must be one of: {set(Mpi3dData.URLS.values())}'
        self.subset = subset

        # TODO: add support for converting to h5py for fast disk access
        assert in_memory, f'{in_memory=} is not yet supported'
        if in_memory:
            log.warning('[WARNING]: mpi3d files are extremely large (over 11GB), you are trying to load these into memory.')

        # initialise
        super().__init__(data_dir=data_dir, force_download=force_download)

        # load data
        if not hasattr(self.__class__, '_DATA'):
            self.__class__._DATA = {}
        if subset not in self.__class__._DATA:
            self.__class__._DATA[subset] = np.load(self.dataset_paths[0])
        self._data = self.__class__._DATA[subset]

    def __getitem__(self, idx):
        return self._data[idx]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
