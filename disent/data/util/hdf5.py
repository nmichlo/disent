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
"""
Utilities for converting and testing different chunk sizes of hdf5 files
"""

import logging
import math
import os

import h5py
import numpy as np
from tqdm import tqdm

from disent.data.util.in_out import AtomicFileContext
from disent.util import iter_chunks
from disent.util import Timer


log = logging.getLogger(__name__)


# ========================================================================= #
# hdf5                                                                   #
# ========================================================================= #


# TODO: cleanup
def bytes_to_human(size_bytes, decimals=3, color=True):
    if size_bytes == 0:
        return "0B"
    size_name = ("B  ", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    size_color = (None,     92,   93,    91,    91,    91,    91,    91,    91)
    i = int(math.floor(math.log(size_bytes, 1024)))
    s = round(size_bytes / math.pow(1024, i), decimals)
    name = f'\033[{size_color[i]}m{size_name[i]}\033[0m' if color else size_name[i]
    return f"{s:{4+decimals}.{decimals}f} {name}"


# TODO: cleanup
def hdf5_print_entry_data_stats(data, dataset, label='STATISTICS', print_mode='all'):
    dtype = data[dataset].dtype
    itemsize = data[dataset].dtype.itemsize
    # chunk
    chunks = np.array(data[dataset].chunks)
    data_per_chunk = np.prod(chunks) * itemsize
    # entry
    shape = np.array([1, *data[dataset].shape[1:]])
    data_per_entry = np.prod(shape) * itemsize
    # chunks per entry
    chunks_per_dim = np.ceil(shape / chunks).astype('int')
    chunks_per_entry = np.prod(chunks_per_dim)
    read_data_per_entry = data_per_chunk * chunks_per_entry
    # print info
    if print_mode == 'all':
        if label:
            tqdm.write(f'[{label}]: \033[92m{dataset}\033[0m')
        tqdm.write(
            f'\t\033[90mentry shape:\033[0m      {str(list(shape)):18s} \033[93m{bytes_to_human(data_per_entry)}\033[0m'
            f'\n\t\033[90mchunk shape:\033[0m      {str(list(chunks)):18s} \033[93m{bytes_to_human(data_per_chunk)}\033[0m'
            f'\n\t\033[90mchunks per entry:\033[0m {str(list(chunks_per_dim)):18s} \033[93m{bytes_to_human(read_data_per_entry)}\033[0m (\033[91m{chunks_per_entry}\033[0m)'
        )
    elif print_mode == 'minimal':
        tqdm.write(
            f'[{label:3s}] entry: {str(list(shape)):18s} ({str(dtype):8s}) \033[93m{bytes_to_human(data_per_entry)}\033[0m chunk: {str(list(chunks)):18s} \033[93m{bytes_to_human(data_per_chunk)}\033[0m chunks per entry: {str(list(chunks_per_dim)):18s} \033[93m{bytes_to_human(read_data_per_entry)}\033[0m (\033[91m{chunks_per_entry}\033[0m)'
        )


# TODO: cleanup
def hd5f_print_dataset_info(data, dataset, label='DATASET'):
    if label:
        tqdm.write(f'[{label}]: \033[92m{dataset}\033[0m')
    tqdm.write(
          f'\t\033[90mraw:\033[0m                {data[dataset]}'
          f'\n\t\033[90mchunks:\033[0m           {data[dataset].chunks}'
          f'\n\t\033[90mcompression:\033[0m      {data[dataset].compression}'
          f'\n\t\033[90mcompression lvl:\033[0m  {data[dataset].compression_opts}'
    )


def hdf5_resave_dataset(inp_h5: h5py.File, out_h5: h5py.File, dataset_name, chunk_size=None, compression=None, compression_lvl=None, batch_size=None, print_mode='minimal'):
    # create new dataset
    out_h5.create_dataset(
        name=dataset_name,
        shape=inp_h5[dataset_name].shape,
        dtype=inp_h5[dataset_name].dtype,
        chunks=chunk_size,
        compression=compression,
        compression_opts=compression_lvl,
        # non-deterministic time stamps are added to the file if this is not
        # disabled, resulting in different hash sums when the file is re-generated!
        # https://github.com/h5py/h5py/issues/225
        # https://stackoverflow.com/questions/16019656
        track_times=False,
    )
    # print stats
    hdf5_print_entry_data_stats(inp_h5, dataset_name, label=f'IN', print_mode=print_mode)
    hdf5_print_entry_data_stats(out_h5, dataset_name, label=f'OUT', print_mode=print_mode)
    # choose batch size for copying data
    if batch_size is None:
        batch_size = inp_h5[dataset_name].chunks[0]
        log.debug(f're-saving h5 dataset using automatic batch size of: {batch_size}')
    # batched copy | loads could be parallelized!
    entries = len(inp_h5[dataset_name])
    with tqdm(total=entries) as progress:
        for i in range(0, entries, batch_size):
            out_h5[dataset_name][i:i + batch_size] = inp_h5[dataset_name][i:i + batch_size]
            progress.update(batch_size)


def hdf5_resave_file(inp_path: str, out_path: str, dataset_name, chunk_size=None, compression=None, compression_lvl=None, batch_size=None, print_mode='minimal'):
    # re-save datasets
    with h5py.File(inp_path, 'r') as inp_h5:
        with AtomicFileContext(out_path, open_mode=None, overwrite=True) as tmp_h5_path:
            with h5py.File(tmp_h5_path, 'w') as out_h5:
                hdf5_resave_dataset(
                    inp_h5=inp_h5,
                    out_h5=out_h5,
                    dataset_name=dataset_name,
                    chunk_size=chunk_size,
                    compression=compression,
                    compression_lvl=compression_lvl,
                    batch_size=batch_size,
                    print_mode=print_mode,
                )
    # file size:
    log.info(f'[FILE SIZES] IN: {bytes_to_human(os.path.getsize(inp_path))} OUT: {bytes_to_human(os.path.getsize(out_path))}')


def hdf5_test_speed(h5_path: str, dataset_name: str, access_method: str = 'random'):
    with h5py.File(h5_path, 'r') as out_h5:
        log.info('[TESTING] Access Speed...')
        log.info(f'Random Accesses Per Second: {hdf5_test_entries_per_second(out_h5, dataset_name, access_method=access_method, max_entries=5_000):.3f}')


def hdf5_test_entries_per_second(h5_data: h5py.File, dataset_name, access_method='random', max_entries=48000, timeout=10, batch_size: int = 256):
    data = h5_data[dataset_name]
    # get access method
    if access_method == 'sequential':
        indices = np.arange(len(data))
    elif access_method == 'random':
        indices = np.arange(len(data))
        np.random.shuffle(indices)
    else:
        raise KeyError('Invalid access method')
    # num entries to test
    n = min(len(data), max_entries)
    indices = indices[:n]
    # iterate through dataset, exit on timeout or max_entries
    t = Timer()
    for chunk in iter_chunks(enumerate(indices), chunk_size=batch_size):
        with t:
            for i, idx in chunk:
                entry = data[idx]
        if t.elapsed > timeout:
            break
    # calculate score
    entries_per_sec = (i + 1) / t.elapsed
    return entries_per_sec


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
