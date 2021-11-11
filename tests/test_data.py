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

import contextlib
import time
from concurrent.futures import ProcessPoolExecutor
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pytest

from disent.dataset.data import Hdf5Dataset
from disent.dataset.data import XYObjectData
from disent.dataset.util.hdf5 import hdf5_resave_file
from disent.dataset.util.hdf5 import hdf5_test_speed
from disent.util.inout.hashing import hash_file
from disent.util.function import wrapped_partial

from tests.util import no_stderr
from tests.util import no_stdout


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


# factors=(3, 3, 2, 3), len=54
TestXYObjectData = wrapped_partial(XYObjectData, grid_size=4, grid_spacing=1, min_square_size=1, max_square_size=2, square_size_spacing=1, palette='rgb_1')
_TEST_LEN = 54


def _iterate_over_data(data, indices):
    i = -1
    for i, idx in enumerate(indices):
        img = data[i]
    return i + 1


@contextlib.contextmanager
def create_temp_h5data(track_times=False, **h5py_dataset_kwargs):
    # generate data
    data = np.stack([img for img in TestXYObjectData()], axis=0)
    # create temp file
    with NamedTemporaryFile('r') as out_file:
        # create temp files
        with h5py.File(out_file.name, 'w', libver='earliest') as file:
            file.create_dataset(name='data', shape=(_TEST_LEN, 4, 4, 3), dtype='uint8', data=data, track_times=track_times, **h5py_dataset_kwargs)
        # return the data & file
        yield out_file.name, data


def test_hdf5_pickle_dataset():
    with create_temp_h5data() as (tmp_path, raw_data):
        # load the data
        # - ideally we want to test this with a pytorch
        #   DataLoader, but that is quite slow to initialise
        with Hdf5Dataset(tmp_path, 'data') as data:
            indices = list(range(len(data)))
            # test locally
            assert _iterate_over_data(data=data, indices=indices) == _TEST_LEN
            # test multiprocessing
            executor = ProcessPoolExecutor(2)
            future_0 = executor.submit(_iterate_over_data, data=data, indices=indices[0::2])
            future_1 = executor.submit(_iterate_over_data, data=data, indices=indices[1::2])
            assert future_0.result() == _TEST_LEN // 2
            assert future_1.result() == _TEST_LEN // 2
            # test multiprocessing on invalid data
            with h5py.File(tmp_path, 'r', swmr=True) as file:
                with pytest.raises(TypeError, match='h5py objects cannot be pickled'):
                    future_2 = executor.submit(_iterate_over_data, data=file['data'], indices=indices)
                    future_2.result()


@pytest.mark.parametrize(['hash_mode', 'target_hash'], [
    ('full', 'a3b60a9e248b4b66bdbf4f87a78bf7cc'),
    ('fast', 'a20d554d4912a39e7654b4dc98207490'),
])
def test_hdf5_determinism(hash_mode: str, target_hash: str):
    # check hashing a
    def make_and_hash(track_times=False, **h5py_dataset_kwargs):
        with create_temp_h5data(track_times=track_times, **h5py_dataset_kwargs) as (path_a, raw_data_a):
            a = hash_file(path_a, hash_type='md5', hash_mode=hash_mode, missing_ok=False)
        # track times only has a resolution of 1 second
        # TODO: this test is slow ~4.4 seconds of sleeping...
        time.sleep(1.1)
        # redo the same task
        with create_temp_h5data(track_times=track_times, **h5py_dataset_kwargs) as (path_b, raw_data_b):
            b = hash_file(path_b, hash_type='md5', hash_mode=hash_mode, missing_ok=False)
        return a, b
    # compute hashes
    deterministic_hash_a, deterministic_hash_b = make_and_hash(track_times=False)
    stochastic_hash_a,    stochastic_hash_b    = make_and_hash(track_times=True)
    # check hashes
    assert deterministic_hash_a == deterministic_hash_b
    assert stochastic_hash_a != stochastic_hash_b
    # check against target
    assert deterministic_hash_a == target_hash
    assert deterministic_hash_b == target_hash
    assert stochastic_hash_a != target_hash
    assert stochastic_hash_b != target_hash


def test_hdf5_resave_dataset():
    with no_stdout(), no_stderr():
        with create_temp_h5data(chunks=(_TEST_LEN, 4, 4, 3)) as (inp_path, raw_data), create_temp_h5data(chunks=None) as (out_path, _):
            # convert dataset
            hdf5_resave_file(
                inp_path=inp_path,
                out_path=out_path,
                dataset_name='data',
                chunk_size=(1, 4, 4, 3),
                compression=None,
                compression_lvl=None,
                batch_size=None,
                out_dtype=None,
                out_mutator=None,
                obs_shape=None,
                write_mode='w',
            )
            # check datasets
            with h5py.File(inp_path, 'r') as inp:
                assert np.all(inp['data'][...] == raw_data)
                assert inp['data'].chunks == (_TEST_LEN, 4, 4, 3)
            with h5py.File(out_path, 'r') as out:
                assert np.all(out['data'][...] == raw_data)
                assert out['data'].chunks == (1, 4, 4, 3)


def test_hdf5_speed_test():
    with create_temp_h5data(chunks=(_TEST_LEN, 4, 4, 3)) as (path, _):
        hdf5_test_speed(path, dataset_name='data', access_method='random')
    with create_temp_h5data(chunks=(1, 4, 4, 3)) as (path, _):
        hdf5_test_speed(path, dataset_name='data', access_method='sequential')
    with create_temp_h5data(chunks=None) as (path, _):
        hdf5_test_speed(path, dataset_name='data', access_method='sequential')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
