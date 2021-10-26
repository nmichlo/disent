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
# shapes3d                                                                  #
# ========================================================================= #


class Shapes3dData(Hdf5GroundTruthData):
    """
    3D Shapes Dataset:
    - https://github.com/deepmind/3d-shapes

    Files:
        - direct:   https://storage.googleapis.com/3d-shapes/3dshapes.h5
          redirect: https://storage.cloud.google.com/3d-shapes/3dshapes.h5
          info:     https://console.cloud.google.com/storage/browser/_details/3d-shapes/3dshapes.h5
    """

    # TODO: name should be `shapes3d` so that it is a valid python identifier
    name = '3dshapes'

    factor_names = ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')
    factor_sizes = (10, 10, 10, 8, 4, 15)  # TOTAL: 480000
    img_shape = (64, 64, 3)

    datafile = DataFileHashedDlH5(
        # download file/link
        uri='https://storage.googleapis.com/3d-shapes/3dshapes.h5',
        uri_hash={'fast': '85b20ed7cc8dc1f939f7031698d2d2ab', 'full': '099a2078d58cec4daad0702c55d06868'},
        # processed dataset file
        file_hash={'fast': 'e3a1a449b95293d4b2c25edbfcb8e804', 'full': 'b5187ee0d8b519bb33281c5ca549658c'},
        # h5 re-save settings
        hdf5_dataset_name='images',
        hdf5_chunk_size=(1, 64, 64, 3),
        hdf5_obs_shape=img_shape,
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    Shapes3dData(prepare=True)
