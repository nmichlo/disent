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
import os
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import loadmat

from disent.dataset.util.datafile import DataFileHashedDlGen
from disent.dataset.data._groundtruth import NumpyFileGroundTruthData
from disent.util.inout.files import AtomicSaveFile


log = logging.getLogger(__name__)


# ========================================================================= #
# cars 3d data processing                                                   #
# ========================================================================= #


def load_cars3d_folder(raw_data_dir):
    """
    nips2015-analogy-data.tar.gz contains:
        1. /data/cars
            - list.txt: [ordered list of mat files "car_***_mesh" without the extension]
            - car_***_mesh.mat: [MATLAB file with keys: "im" (128, 128, 3, 24, 4), "mask" (128, 128, 24, 4)]
        2. /data/sprites
        3. /data/shapes48.mat
    """
    # load image paths
    with open(os.path.join(raw_data_dir, 'cars/list.txt'), 'r') as img_names:
        img_paths = [os.path.join(raw_data_dir, f'cars/{name.strip()}.mat') for name in img_names.readlines()]
    # load images
    images = np.stack([loadmat(img_path)['im'] for img_path in img_paths], axis=0)
    # check size
    assert images.shape == (183, 128, 128, 3, 24, 4)
    # reshape & transpose: (183, 128, 128, 3, 24, 4) -> (4, 24, 183, 128, 128, 3) -> (17568, 128, 128, 3)
    return images.transpose([5, 4, 0, 1, 2, 3]).reshape([-1, 128, 128, 3])


def resave_cars3d_archive(orig_zipped_file, new_save_file, overwrite=False):
    """
    Convert a cars3d archive 'nips2015-analogy-data.tar.gz' to a numpy file,
    uncompressing the contents of the archive into a temporary directory in the same folder.
    """
    with TemporaryDirectory(prefix='raw_cars3d_', dir=os.path.dirname(orig_zipped_file)) as temp_dir:
        # extract zipfile and get path
        log.info(f"Extracting into temporary directory: {temp_dir}")
        shutil.unpack_archive(filename=orig_zipped_file, extract_dir=temp_dir)
        # load image paths & resave
        with AtomicSaveFile(new_save_file, overwrite=overwrite) as temp_file:
            images = load_cars3d_folder(raw_data_dir=os.path.join(temp_dir, 'data'))
            np.savez(temp_file, images=images)


# ========================================================================= #
# cars3d data object                                                        #
# ========================================================================= #


class DataFileCars3d(DataFileHashedDlGen):
    """
    download the cars3d dataset and convert it to a numpy file.
    """
    def _generate(self, inp_file: str, out_file: str):
        resave_cars3d_archive(orig_zipped_file=inp_file, new_save_file=out_file, overwrite=True)


# ========================================================================= #
# dataset_cars3d                                                            #
# ========================================================================= #


class Cars3dData(NumpyFileGroundTruthData):
    """
    Cars3D Dataset
    - Deep Visual Analogy-Making (https://papers.nips.cc/paper/5845-deep-visual-analogy-making)
      http://www.scottreed.info

    # reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/cars3d.py
    """

    name = 'cars3d'

    factor_names = ('elevation', 'azimuth', 'object_type')
    factor_sizes = (4, 24, 183)  # TOTAL: 17568
    img_shape = (128, 128, 3)

    datafile = DataFileCars3d(
        uri='http://www.scottreed.info/files/nips2015-analogy-data.tar.gz',
        uri_hash={'fast': 'fe77d39e3fa9d77c31df2262660c2a67', 'full': '4e866a7919c1beedf53964e6f7a23686'},
        file_name='cars3d.npz',
        file_hash={'fast': 'ef5d86d1572ddb122b466ec700b3abf2', 'full': 'dc03319a0b9118fbe0e23d13220a745b'},
        hash_mode='fast'
    )

    # override
    data_key = 'images'


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    Cars3dData(prepare=True)
