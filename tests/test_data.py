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

from concurrent.futures import ProcessPoolExecutor
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pytest

from disent.data.groundtruth import XYSquaresData
from disent.data.groundtruth._xysquares import XYSquaresMinimalData
from disent.data.hdf5 import PickleH5pyFile


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


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


def _iterate_over_data(data, indices):
    i = -1
    for i, idx in enumerate(indices):
        img = data[i]
    return i + 1


def test_hdf5_pickle_dataset():
    with NamedTemporaryFile('r') as temp_file:
        # create temporary dataset
        with h5py.File(temp_file.name, 'w') as file:
            file.create_dataset(
                name='data',
                shape=(64, 4, 4, 3),
                dtype='uint8',
                data=np.stack([img for img in XYSquaresData(square_size=2, image_size=4)], axis=0)
            )
        # load the data
        # - ideally we want to test this with a pytorch
        #   DataLoader, but that is quite slow to initialise
        with PickleH5pyFile(temp_file.name, 'data') as data:
            indices = list(range(len(data)))
            # test locally
            assert _iterate_over_data(data=data, indices=indices) == 64
            # test multiprocessing
            executor = ProcessPoolExecutor(2)
            future_0 = executor.submit(_iterate_over_data, data=data, indices=indices[0::2])
            future_1 = executor.submit(_iterate_over_data, data=data, indices=indices[1::2])
            assert future_0.result() == 32
            assert future_1.result() == 32
            # test multiprocessing on invalid data
            with h5py.File(temp_file.name, 'r', swmr=True) as file:
                with pytest.raises(TypeError, match='h5py objects cannot be pickled'):
                    future_2 = executor.submit(_iterate_over_data, data=file['data'], indices=indices)
                    future_2.result()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

