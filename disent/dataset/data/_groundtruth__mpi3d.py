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
from typing import Optional

from disent.dataset.util.datafile import DataFileHashedDl
from disent.dataset.data._groundtruth import NumpyFileGroundTruthData


log = logging.getLogger(__name__)

# ========================================================================= #
# mpi3d                                                                     #
# ========================================================================= #


class Mpi3dData(NumpyFileGroundTruthData):
    """
    MPI3D Dataset
    - https://github.com/rr-learning/disentanglement_dataset

    reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/mpi3d.py
    """

    MPI3D_DATASETS = {
        'toy':        DataFileHashedDl(uri='https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz',       uri_hash={'fast': '146138e36ff495e77ceacdc8cf14c37e', 'full': '55889cb7c7dfc655d6e0277beee88868'}),
        'realistic':  DataFileHashedDl(uri='https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_realistic.npz', uri_hash={'fast': '96c8ff1155dd61f79d3493edef9f19e9', 'full': '59a6225b88b635365f70c91b3e52f70f'}),
        'real':       DataFileHashedDl(uri='https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz',      uri_hash={'fast': 'e2941bba6f4a2b130edc5f364637b39e', 'full': '0f33f609918fb5c97996692f91129802'}),
    }

    factor_names = ('object_color', 'object_shape', 'object_size', 'camera_height', 'background_color', 'first_dof', 'second_dof')
    factor_sizes = (4, 4, 2, 3, 3, 40, 40)  # TOTAL: 460800
    img_shape = (64, 64, 3)

    # override
    data_key = 'images'

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, subset='realistic', in_memory=False, transform=None):
        # check subset is correct
        assert subset in self.MPI3D_DATASETS, f'Invalid MPI3D subset: {repr(subset)} must be one of: {set(self.MPI3D_DATASETS.keys())}'
        self._subset = subset
        # handle different cases
        if in_memory:
            log.warning('[WARNING]: mpi3d files are extremely large (over 11GB), you are trying to load these into memory.')
        else:
            raise NotImplementedError('TODO: add support for converting to h5py for fast disk access')  # TODO!
        # initialise
        super().__init__(data_root=data_root, prepare=prepare, transform=transform)

    @property
    def datafile(self) -> DataFileHashedDl:
        return self.MPI3D_DATASETS[self._subset]

    @property
    def name(self) -> str:
        return f'mpi3d_{self._subset}'


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    Mpi3dData(prepare=True, in_memory=False)
