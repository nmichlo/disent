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
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np

from disent.dataset.data._groundtruth import NumpyFileGroundTruthData
from disent.dataset.util.datafile import DataFileHashed
from disent.dataset.util.datafile import DataFileHashedDlGen
from disent.util.inout.paths import modify_name_keep_ext


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
    from scipy.io import loadmat
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
        # load images
        images = load_cars3d_folder(raw_data_dir=os.path.join(temp_dir, 'data'))
    # save the array
    from disent.dataset.util.formats.npz import save_dataset_array
    save_dataset_array(images, new_save_file, overwrite=overwrite, save_key='images')


def resave_cars3d_resized(orig_converted_file: str, new_resized_file: str, overwrite=False, size: int = 64):
    # load the array
    cars3d_array = np.load(orig_converted_file)['images']
    assert cars3d_array.shape == (17568, 128, 128, 3)
    # save the array
    from disent.dataset.util.formats.npz import save_resized_dataset_array
    save_resized_dataset_array(cars3d_array, new_resized_file, overwrite=overwrite, size=size, save_key='images')


# ========================================================================= #
# cars3d data object                                                        #
# ========================================================================= #


class DataFileCars3d(DataFileHashedDlGen):
    """
    download the cars3d dataset and convert it to a numpy file.
    """
    def _generate(self, inp_file: str, out_file: str):
        resave_cars3d_archive(orig_zipped_file=inp_file, new_save_file=out_file, overwrite=True)


class DataFileCars3dResized(DataFileHashed):

    def __init__(
        self,
        cars3d_datafile: DataFileCars3d,
        # - convert file name
        out_hash: Optional[Union[str, Dict[str, str]]],
        out_name: Optional[str] = None,
        out_size: int = 64,
        # - hash settings
        hash_type: str = 'md5',
        hash_mode: str = 'fast',
    ):
        self._out_size = out_size
        self._cars3dfile = cars3d_datafile
        super().__init__(
            file_name=modify_name_keep_ext(self._cars3dfile.out_name, suffix=f'_x{out_size}') if (out_name is None) else out_name,
            file_hash=out_hash,
            hash_type=hash_type,
            hash_mode=hash_mode,
        )

    def _prepare(self, out_dir: str, out_file: str):
        log.debug('Preparing Orig Cars3d Data:')
        cars3d_path = self._cars3dfile.prepare(out_dir)
        log.debug('Generating Resized Cars3d Data:')
        resave_cars3d_resized(orig_converted_file=cars3d_path, new_resized_file=out_file, overwrite=True, size=self._out_size)


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
        file_hash={'fast': '204ecb6852216e333f1b022903f9d012', 'full': '46ad66acf277897f0404e522460ba7e5'},
        hash_mode='fast'
    )

    # override
    data_key = 'images'


# TODO: this is very slow compared to other datasets for some reason!
#       - in memory benchmark are equivalent, eg. against Shapes3D, but when we run the
#         experiment/run.py with this its about twice as slow? Why is this?
class Cars3d64Data(Cars3dData):
    """
    Optimized version of Cars3dOrigData, that has already been re-sized to 64x64
    - This can improve run times dramatically!
    """

    img_shape = (64, 64, 3)

    datafile = DataFileCars3dResized(
        cars3d_datafile=Cars3dData.datafile,
        out_name='cars3d_x64.npz',
        out_hash={'fast': '5a85246b6f555bc6e3576ee62bf6d19e', 'full': '2b900b3c5de6cd9b5df87bfc02f01f03'},
        hash_mode='fast',
        out_size=64,
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    def main():
        import torch
        from tqdm import tqdm
        from disent.dataset.transform import ToImgTensorF32

        logging.basicConfig(level=logging.DEBUG)

        # original dataset
        data_128 = Cars3dData(prepare=True, transform=ToImgTensorF32(size=64))
        for i in tqdm(data_128, desc='cars3d_x128 -> 64'):
            pass
        # resized dataset
        data_64 = Cars3d64Data(prepare=True, transform=ToImgTensorF32(size=64))
        for i in tqdm(data_64, desc='cars3d_x64'):
            pass
        # check equivalence
        for obs_128, obs_64 in tqdm(zip(data_128, data_64), desc='equivalence'):
            assert torch.allclose(obs_128, obs_64)

    main()
