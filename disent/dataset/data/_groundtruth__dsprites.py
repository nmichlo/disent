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

from disent.dataset.util.datafile import DataFileHashedDlH5
from disent.dataset.data._groundtruth import Hdf5GroundTruthData


# ========================================================================= #
# dataset_dsprites                                                          #
# ========================================================================= #


class DSpritesData(Hdf5GroundTruthData):
    """
    DSprites Dataset
    - beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
      (https://github.com/deepmind/dsprites-dataset)

    Files:
        - direct npz: https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
                      approx 2.5 GB loaded into memory
        - direct hdf5: https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5
                       default chunk size is (23040, 2, 4), dataset is (737280, 64, 64) uint8.

    # reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/dsprites.py
    """

    name = 'dsprites'

    # TODO: reference implementation has colour variants
    factor_names = ('shape', 'scale', 'orientation', 'position_x', 'position_y')
    factor_sizes = (3, 6, 40, 32, 32)  # TOTAL: 737280
    img_shape = (64, 64, 1)

    datafile = DataFileHashedDlH5(
        # download file/link
        uri='https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5',
        uri_hash={'fast': 'd6ee1e43db715c2f0de3c41e38863347', 'full': 'b331c4447a651c44bf5e8ae09022e230'},
        # processed dataset file
        file_hash={'fast': '25013c85aebbf4b1023d72564f9413f0', 'full': '4611d1a03e709cd5d0f6fdcdc221ca0e'},
        # h5 re-save settings
        hdf5_dataset_name='imgs',
        hdf5_chunk_size=(1, 64, 64, 1),
        hdf5_dtype='uint8',
        hdf5_mutator=lambda x: (x * 255)[..., None],  # data is of shape (-1, 64, 64), so we add the channel dimension
        hdf5_obs_shape=(64, 64, 1),
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    DSpritesData(prepare=True)
