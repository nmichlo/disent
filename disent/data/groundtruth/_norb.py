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
import dataclasses
import gzip
from typing import Dict

import numpy as np

from disent.data.groundtruth import GroundTruthData
from disent.data.groundtruth.base import DlDataObject


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


def read_binary_matrix_buffer(buffer):
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
    dtype = int(np.frombuffer(buffer, "int32", 1, 0))          # bytes [0, 4)
    ndim  = int(np.frombuffer(buffer, "int32", 1, 4))          # bytes [4, 8)
    eff_dim = max(3, ndim)                                     # stores minimum of 3 dimensions even for 1D array
    dims = np.frombuffer(buffer, "int32", eff_dim, 8)[0:ndim]  # bytes [8, 8 + eff_dim * 4)
    data = np.frombuffer(buffer, _BINARY_MATRIX_TYPES[dtype], offset=8 + eff_dim * 4)
    data = data.reshape(tuple(dims))
    return data


def read_binary_matrix_file(file, gzipped: bool = True):
    with (gzip.open if gzipped else open)(file, "rb") as f:
        return read_binary_matrix_buffer(buffer=f)


def resave_binary_matrix_file(inp_path, out_path, gzipped: bool = True):
    with AtomicFileContext(out_path, open_mode=None) as temp_out_path:
        data = read_binary_matrix_file(file=inp_path, gzipped=gzipped)
        np.savez(temp_out_path, data=data)


# ========================================================================= #
# Norb Data Tasks                                                           #
# ========================================================================= #


@dataclasses.dataclass
class BinaryMatrixDataObject(DlDataObject):
    file_name: str
    file_hashes: Dict[str, str]
    # download file/link
    uri: str
    uri_hashes: Dict[str, str]
    # hash settings
    hash_mode: str
    hash_type: str

    def _make_h5_job(self, load_path: str, save_path: str):
        return CachedJobFile(
            make_file_fn=lambda path: resave_binary_matrix_file(
                inp_path=load_path,
                out_path=path,
                gzipped=True,
            ),
            path=save_path,
            hash=self.file_hashes[self.hash_mode],
            hash_type=self.hash_type,
            hash_mode=self.hash_mode,
        )

    def prepare(self, data_dir: str):
        dl_path = self.get_file_path(data_dir=data_dir, variant='ORIG')
        h5_path = self.get_file_path(data_dir=data_dir)
        dl_job = self._make_dl_job(save_path=dl_path)
        h5_job = self._make_h5_job(load_path=dl_path, save_path=h5_path)
        dl_job.set_child(h5_job).run()


# ========================================================================= #
# dataset_norb                                                              #
# ========================================================================= #


class SmallNorbData(GroundTruthData):
    """
    Small NORB Dataset
    - https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/

    # reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/norb.py
    """

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
        # read dataset and sort by features
        images, features = self._read_norb_set(is_test)
        indices = np.lexsort(features[:, [4, 3, 2, 1, 0]].T)
        self._data = images[indices]

    def __getitem__(self, idx):
        return self._data[idx]

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


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

