#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2022 Nathan Juraj Michlo
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
from typing import Dict
from typing import NoReturn
from typing import Optional
from typing import Union

import numpy as np

from disent.dataset.data import NumpyFileGroundTruthData
from disent.dataset.util.datafile import DataFileHashed
from disent.util.inout.paths import modify_name_keep_ext


log = logging.getLogger(__name__)


# ========================================================================= #
# teapots 3d data processing                                                #
# ========================================================================= #


def resave_teapots3d_as_uint8(orig_file: str, new_file: str, overwrite: bool = False):
    # load data into memory ~10GB
    # -- by default this array is stored as uint32, instead of uint8
    log.debug('loading teapots data into memory, this may take a while...')
    imgs = np.load(orig_file)['images']
    log.debug('loaded teapots data into memory!')
    # checks
    log.debug('checking teapots data...')
    assert imgs.dtype == 'int32'
    assert imgs.shape == (200_000, 64, 64, 3)
    assert imgs.max() == 255
    assert imgs.min() == 0
    log.debug('checked teapots data!')
    # convert the values
    log.debug('converting teapots data to uint8...')
    imgs = imgs.astype('uint8')
    log.debug('converted teapots data!')
    # save the array
    from disent.dataset.util.formats.npz import save_dataset_array
    log.debug('saving convert teapots data...')
    save_dataset_array(imgs, new_file, overwrite=overwrite, save_key='images')
    log.debug('saved convert teapots data!')


# ========================================================================= #
# teapots 3d data files                                                     #
# ========================================================================= #


class DataFileTeapots3dInt32(DataFileHashed):

    # TODO: add a version of this file that automatically unpacks the original zip file?

    def _prepare(self, out_dir: str, out_file: str) -> NoReturn:
        if not os.path.exists(out_file):
            raise FileNotFoundError(
                f'Please download the Teapots3D dataset to: {repr(out_file)}'
                f'\nThe original repository is: {repr("https://github.com/cianeastwood/qedr")}'
                f'\nThe original download link is: {repr("https://www.dropbox.com/s/woeyomxuylqu7tx/edinburgh_teapots.zip?dl=0")}'
            )


class DataFileTeapots3dUint8(DataFileHashed):

    def __init__(
        self,
        teapots3d_datafile: DataFileTeapots3dInt32,
        # - convert file name
        out_hash: Optional[Union[str, Dict[str, str]]],
        out_name: Optional[str] = None,
        # - hash settings
        hash_type: str = 'md5',
        hash_mode: str = 'fast',
    ):
        self._teapots3dfile = teapots3d_datafile
        super().__init__(
            file_name=modify_name_keep_ext(self._teapots3dfile.out_name, suffix=f'_uint8') if (out_name is None) else out_name,
            file_hash=out_hash,
            hash_type=hash_type,
            hash_mode=hash_mode,
        )

    def _prepare(self, out_dir: str, out_file: str):
        log.debug('Preparing Orig Teapots3d Data:')
        cars3d_path = self._teapots3dfile.prepare(out_dir)
        log.debug('Converting Teapots3d Data to Uint8:')
        resave_teapots3d_as_uint8(orig_file=cars3d_path, new_file=out_file, overwrite=True)


# ========================================================================= #
# teapots 3d dataset                                                        #
# ========================================================================= #


class Teapots3dData(NumpyFileGroundTruthData):
    """
    Teapots3D Dataset
    -  A Framework for the Quantitative Evaluation of Disentangled Representations
       * https://openreview.net/forum?id=By-7dz-AZ
       * https://github.com/cianeastwood/qedr

    Manual Download Link:
    - https://www.dropbox.com/s/woeyomxuylqu7tx/edinburgh_teapots.zip?dl=0

    NOTE:
    - This dataset is generated from ground-truth factors, HOWEVER, each datapoint
      is randomly sampled. This dataset is NOT a typical grid-search over ground-truth factors
      which means that we cannot create a StateSpace object over this dataset.
    """

    name = 'edinburgh_teapots'

    factor_names = ('azimuth', 'elevation', 'red', 'green', 'blue')
    factor_sizes = (..., ..., ..., ..., ...)  # TOTAL: 200_000 -- TODO: this is invalid, we cannot actually generate a StateSpace object over this dataset!
    img_shape = (64, 64, 3)

    datafile = DataFileTeapots3dUint8(
        teapots3d_datafile=DataFileTeapots3dInt32(
            file_name='teapots.npz',
            file_hash={'full': '9b58d66a382d01f4477e33520f1fa503', 'fast': '12c889e001c205d0bafa59dfff114102'},
        ),
        out_hash={'full': 'e64207ee443030d310500d762f0d1dfd', 'fast': '7fbca0223c27e055d35b6d5af720f108'},
        out_name='teapots_uint8.npz',
    )

    # override
    data_key = 'images'


# ========================================================================= #
# main                                                                      #
# ========================================================================= #



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    data = Teapots3dData(data_root='~/Downloads', prepare=True)
