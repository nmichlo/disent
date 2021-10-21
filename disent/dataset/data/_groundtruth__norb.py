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

import gzip
import logging
import os
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

from disent.dataset.util.datafile import DataFileHashedDl
from disent.dataset.data._groundtruth import DiskGroundTruthData


# ========================================================================= #
# Binary Matrix Helper Functions                                            #
# - https://cs.nyu.edu/~ylclab/data/norb-v1.0-small                         #
# ========================================================================= #


_BINARY_MATRIX_TYPES = {
    0x1E3D4C55: 'uint8',    # byte matrix
    0x1E3D4C54: 'int32',    # integer matrix
    0x1E3D4C56: 'int16',    # short matrix
    0x1E3D4C51: 'float32',  # single precision matrix
    0x1E3D4C53: 'float64',  # double precision matrix
    # 0x1E3D4C52: '???',    # packed matrix -- not sure what this is?
}


def read_binary_matrix_bytes(bytes):
    """
    Read the binary matrix data
    - modified from disentanglement_lib

    Binary Matrix File Format Specification
        * The Header:
            - dtype:      4 bytes
            - ndim:       4 bytes, little endian
            - dim_sizes: (4 * min(3, ndim)) bytes
        * Handling the number of dimensions:
            - If there are less than 3 dimensions, then dim[1] and dim[2] are both: 1
            - Elif there are 3 or more dimensions, then the header will contain further size information.
        * Handling Matrix Data:
            - Little endian matrix data comes after the header,
              the index of the last dimension changes the fastest.
    """
    # header: dtype, ndim, dim_sizes
    dtype = int(np.frombuffer(bytes, dtype='int32', count=1, offset=0))  # bytes [0, 4)
    ndim  = int(np.frombuffer(bytes, dtype='int32', count=1, offset=4))  # bytes [4, 8)
    stored_ndim = max(3, ndim)  # stores minimum of 3 dimensions even for 1D array
    dims = np.frombuffer(bytes, dtype='int32', count=stored_ndim, offset=8)[0:ndim]  # bytes [8, 8 + eff_dim * 4)
    # matrix: data
    data = np.frombuffer(bytes, dtype=_BINARY_MATRIX_TYPES[dtype], count=-1, offset=8 + stored_ndim * 4)
    data = data.reshape(tuple(dims))
    # done
    return data


def read_binary_matrix_file(file, gzipped: bool = True):
    # this does not seem to copy the bytes, which saves memory
    with (gzip.open if gzipped else open)(file, "rb") as f:
        return read_binary_matrix_bytes(bytes=f.read())


# ========================================================================= #
# Norb Functions                                                            #
# ========================================================================= #


def read_norb_dataset(dat_path: str, cat_path: str, info_path: str, gzipped=True, sort=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load The Normalised Dataset
    * dat:
        - images (5 categories, 5 instances, 6 lightings, 9 elevations, and 18 azimuths)
    * cat:
        - initial ground truth factor:
            0. category of images (0 for animal, 1 for human, 2 for plane, 3 for truck, 4 for car).
    * info:
        - additional ground truth factors:
            1. the instance in the category (0 to 9)
            2. the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70 degrees from the horizontal respectively)
            3. the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
            4. the lighting condition (0 to 5)
    """
    # read the dataset
    dat = read_binary_matrix_file(dat_path, gzipped=gzipped)
    cat = read_binary_matrix_file(cat_path, gzipped=gzipped)
    info = read_binary_matrix_file(info_path, gzipped=gzipped)
    # collect the ground truth factors
    factors = np.column_stack([cat, info])  # append info to categories
    factors[:, 3] = factors[:, 3] / 2       # azimuth values are even numbers, convert to indices
    images = dat[:, 0]                      # images are in pairs, only use the first. TODO: what is the second of each?
    # order the images and factors
    if sort:
        indices = np.lexsort(factors[:, [4, 3, 2, 1, 0]].T)
        images = images[indices]
        factors = factors[indices]
    # done!
    return images, factors


# ========================================================================= #
# dataset_norb                                                              #
# ========================================================================= #


class SmallNorbData(DiskGroundTruthData):
    """
    Small NORB Dataset
    - https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/

    # reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/norb.py
    # TODO: add ability to randomly sample the instance so that this corresponds to disentanglement_lib
    """

    name = 'smallnorb'

    factor_names = ('category', 'instance', 'elevation', 'rotation', 'lighting')
    factor_sizes = (5, 5, 9, 18, 6)  # TOTAL: 24300
    img_shape = (96, 96, 1)

    TRAIN_DATA_FILES = {
        'dat': DataFileHashedDl(uri='https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz', uri_hash={'fast': '92560cccc7bcbd6512805e435448b62d', 'full': '66054832f9accfe74a0f4c36a75bc0a2'}),
        'cat': DataFileHashedDl(uri='https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz', uri_hash={'fast': '348fc3ccefd651d69f500611988b5dcd', 'full': '23c8b86101fbf0904a000b43d3ed2fd9'}),
        'info': DataFileHashedDl(uri='https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz', uri_hash={'fast': 'f1b170c16925867c05f58608eb33ba7f', 'full': '51dee1210a742582ff607dfd94e332e3'}),
    }

    TEST_DATA_FILES = {
        'dat': DataFileHashedDl(uri='https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz', uri_hash={'fast': '9aee0b474a4fc2a2ec392b463efb8858', 'full': 'e4ad715691ed5a3a5f138751a4ceb071'}),
        'cat': DataFileHashedDl(uri='https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz', uri_hash={'fast': '8cfae0679f5fa2df7a0aedfce90e5673', 'full': '5aa791cd7e6016cf957ce9bdb93b8603'}),
        'info': DataFileHashedDl(uri='https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz', uri_hash={'fast': 'd2703a3f95e7b9a970ad52e91f0aaf6a', 'full': 'a9454f3864d7fd4bb3ea7fc3eb84924e'}),
    }

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, is_test=False, transform=None):
        self._is_test = is_test
        # initialize
        super().__init__(data_root=data_root, prepare=prepare, transform=transform)
        # read dataset and sort by features
        dat_path, cat_path, info_path = (os.path.join(self.data_dir, obj.out_name) for obj in self.datafiles)
        self._data, _ = read_norb_dataset(dat_path=dat_path, cat_path=cat_path, info_path=info_path)

    def _get_observation(self, idx):
        return self._data[idx][:, :, None]  # data is missing channel dim

    @property
    def datafiles(self) -> Sequence[DataFileHashedDl]:
        norb_objects = self.TEST_DATA_FILES if self._is_test else self.TRAIN_DATA_FILES
        return norb_objects['dat'], norb_objects['cat'], norb_objects['info']


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    SmallNorbData(prepare=True)
