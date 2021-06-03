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
import numpy as np
import pytest

from disent.data.groundtruth import Shapes3dData
from disent.data.groundtruth import XYSquaresData
from disent.data.groundtruth._xysquares import XYSquaresMinimalData


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #
from disent.data.groundtruth.base import Hdf5GroundTruthData


def test_xysquares_similarity():
    data_org = XYSquaresData()
    data_min = XYSquaresMinimalData()
    # check lengths
    assert len(data_org) == len(data_min)
    n = len(data_min)
    # check items
    for i in np.random.randint(0, n, size=100):
        assert np.allclose(data_org[i], data_min[i])
    # check bounds
    assert np.allclose(data_org[0], data_min[0])
    assert np.allclose(data_org[n-1], data_min[n-1])





@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_hdf5_multiproc_dataset(num_workers):
    from disent.dataset.random import RandomDataset
    from torch.utils.data import DataLoader

    xysquares = XYSquaresData(square_size=2, image_size=4)


    # class TestHdf5Dataset(Hdf5GroundTruthData):
    #
    #
    #     factor_names = ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')
    #     factor_sizes = (10, 10, 10, 8, 4, 15)  # TOTAL: 480000
    #     observation_shape = (64, 64, 3)
    #
    #     data_object = DlH5DataObject(
    #         # processed dataset file
    #         file_name='3dshapes.h5',
    #         file_hashes={'fast': 'e3a1a449b95293d4b2c25edbfcb8e804', 'full': 'b5187ee0d8b519bb33281c5ca549658c'},
    #         # download file/link
    #         uri='https://storage.googleapis.com/3d-shapes/3dshapes.h5',
    #         uri_hashes={'fast': '85b20ed7cc8dc1f939f7031698d2d2ab', 'full': '099a2078d58cec4daad0702c55d06868'},
    #         # hash settings
    #         hash_mode='fast',
    #         hash_type='md5',
    #         # h5 re-save settings
    #         hdf5_dataset_name='images',
    #         hdf5_chunk_size=(1, 64, 64, 3),
    #         hdf5_compression='gzip',
    #         hdf5_compression_lvl=4,
    #     )
    #
    #
    #
    # Shapes3dData()
    # dataset = RandomDataset(Shapes3dData(prepare=True))
    #
    # dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=2, shuffle=True)
    #
    # with tqdm(total=len(dataset)) as progress:
    #     for batch in dataloader:
    #         progress.update(256)



# ========================================================================= #
# END                                                                       #
# ========================================================================= #

