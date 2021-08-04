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
import os
import warnings
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

from disent.util.strings import colors as c
from disent.util.inout.files import AtomicSaveFile
from disent.util.iters import iter_chunks
from disent.util.profiling import Timer
from disent.util.strings.fmt import bytes_to_human


log = logging.getLogger(__name__)


# ========================================================================= #
# hdf5 - resave                                                             #
# ========================================================================= #


# def _get_normalized_factor_names(gt_dataset, max_len=32):
#     # check inputs
#     if (factor_names is not None) and (factor_sizes is not None):
#         factor_names = tuple(s.encode('ascii') for s in factor_names)
#         factor_sizes = tuple(factor_sizes)
#         if any(len(s) > max_len for s in factor_names):
#             raise ValueError(f'factor names must be at most 32 ascii characters long')
#         if len(factor_names) != len(factor_sizes):
#             raise ValueError(f'length of factor names must be length of factor sizes: len({factor_names}) != len({factor_sizes})')
#         return factor_names, factor_sizes
#     elif not ((factor_names is None) and (factor_sizes is None)):
#         raise ValueError('factor_names and factor_sizes must both be given together.')
#     return None


def _normalize_dtype(dtype: Union[torch.dtype, np.dtype, str]) -> np.dtype:
    if isinstance(dtype, torch.dtype):
        dtype: str = torch.finfo(torch.float32).dtype
    return np.dtype(dtype)


def _normalize_out_array(array: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.cpu().detach().numpy()
    return np.array(array)


def hdf5_save_array(
    inp_data: Union[h5py.Dataset, np.ndarray, 'torch.Tensor'],
    out_h5: h5py.File,
    dataset_name: str,  # input and output dataset name
    chunk_size: Optional[Union[Tuple[int, ...], Literal[True]]] = None,  # True: auto determine, Tuple: specific chunk size, None: disable chunking
    compression: Optional[Union[Literal['gzip'], Literal['lzf']]] = None,  # compression type, only works if chunks is specified
    compression_lvl: Optional[int] = None,  # 0 through 9
    batch_size: Optional[int] = None,  # batch size to process / save at a time
    out_dtype: Optional[Union[np.dtype, str]] = None,  # output dtype of the dataset
    out_mutator: Optional[Callable[[np.ndarray], np.ndarray]] = None,  # mutate batches before saving
    obs_shape: Optional[Tuple[int, ...]] = None,  # resize batches to this shape
):
    # TODO: this should take in an array object and output the file!
    # check out_h5 version compatibility
    if (isinstance(out_h5.libver, str) and out_h5.libver != 'earliest') or (out_h5.libver[0] != 'earliest'):
        raise RuntimeError(f'hdf5 out file has an incompatible libver: {repr(out_h5.libver)} libver should be set to: "earliest"')
    # get observation size
    if obs_shape is None:
        obs_shape = inp_data.shape[1:]
    # create new dataset
    out_data = out_h5.create_dataset(
        name=dataset_name,
        shape=(inp_data.shape[0], *obs_shape),
        dtype=out_dtype if (out_dtype is not None) else _normalize_dtype(inp_data.dtype),
        chunks=chunk_size,
        compression=compression,
        compression_opts=compression_lvl,
        # non-deterministic time stamps are added to the file if this is not
        # disabled, resulting in different hash sums when the file is re-generated!
        # - https://github.com/h5py/h5py/issues/225
        # - https://stackoverflow.com/questions/16019656
        # other properties:
        # - https://docs.h5py.org/en/stable/high/group.html#h5py.Group.create_dataset
        track_times=False,
        # track_order=False,
        # fletcher32=True,  # checksum for each chunk
        # shuffle=True,     # reorder chunk values to possibly help compression
        # scaleoffset=<int> # enable lossy compression, ints: number of bits to keep (0 is automatic lossless), floats: number of digits after decimal
    )
    # print stats
    tqdm.write('')
    hdf5_print_entry_data_stats(inp_data, label=f'IN')
    hdf5_print_entry_data_stats(out_data, label=f'OUT')
    # choose batch size for copying data
    if batch_size is None:
        batch_size = inp_data.chunks[0] if (hasattr(inp_data, 'chunks') and inp_data.chunks) else 32
        log.debug(f'saving h5 dataset using automatic batch size of: {batch_size}')
    # get default
    if out_mutator is None:
        out_mutator = lambda x: x
    # save data
    with tqdm(total=len(inp_data)) as progress:
        for i in range(0, len(inp_data), batch_size):
            # load and modify the batch
            batch = inp_data[i:i + batch_size]
            batch = _normalize_out_array(batch)
            batch = out_mutator(batch)
            assert batch.shape == (batch_size, *obs_shape), f'batch shape: {tuple(batch.shape)} from processed input data does not match required obs shape: {(batch_size, *obs_shape)}, try changing the `obs_shape` or resizing the batch in the `out_mutator`.'
            # save the batch
            out_data[i:i + batch_size] = batch
            progress.update(batch_size)


def hdf5_resave_file(
    inp_path: Union[str, torch.Tensor, np.ndarray],
    out_path: str,
    dataset_name: str,  # input and output dataset name
    chunk_size: Optional[Union[Tuple[int, ...], Literal[True]]] = None,  # True: auto determine, Tuple: specific chunk size, None: disable chunking
    compression: Optional[Union[Literal['gzip'], Literal['lzf']]] = None,  # compression type, only works if chunks is specified
    compression_lvl: Optional[int] = None,  # 0 through 9
    batch_size: Optional[int] = None,  # batch size to process / save at a time
    out_dtype: Optional[Union[np.dtype, str]] = None,  # output dtype of the dataset
    out_mutator: Optional[Callable[[np.ndarray], np.ndarray]] = None,  # mutate batches before saving
    obs_shape: Optional[Tuple[int, ...]] = None,  # resize batches to this shape
    write_mode: Union[Literal['atomic_w'], Literal['w'], Literal['a']] = 'atomic_w',
):
    if isinstance(inp_path, str):
        inp_context = h5py.File(inp_path, 'r')
    else:
        import contextlib
        inp_context = contextlib.nullcontext(inp_path)
    # re-save datasets
    with inp_context as inp_data:
        # get input dataset from h5 file
        if isinstance(inp_data, h5py.File):
            inp_data = inp_data[dataset_name]
        # get context manager
        if write_mode == 'atomic_w':
            save_context = AtomicSaveFile(out_path, open_mode=None, overwrite=True)
            write_mode = 'w'
        else:
            import contextlib
            save_context = contextlib.nullcontext(out_path)
        # handle saving to file
        with save_context as tmp_h5_path:
            with h5py.File(tmp_h5_path, write_mode, libver='earliest') as out_h5:  # TODO: libver='latest' is not deterministic, even with track_times=False
                hdf5_save_array(
                    inp_data=inp_data,
                    out_h5=out_h5,
                    dataset_name=dataset_name,
                    chunk_size=chunk_size,
                    compression=compression,
                    compression_lvl=compression_lvl,
                    batch_size=batch_size,
                    out_dtype=out_dtype,
                    out_mutator=out_mutator,
                    obs_shape=obs_shape,
                )
    # file size:
    log.info(f'[FILE SIZES] IN: {bytes_to_human(os.path.getsize(inp_path)) if isinstance(inp_path, str) else "N/A"} OUT: {bytes_to_human(os.path.getsize(out_path))}')


# ========================================================================= #
# hdf5 - speed tests                                                        #
# ========================================================================= #


def hdf5_test_entries_per_second(h5_dataset: h5py.Dataset, access_method='random', max_entries=48000, timeout=10, batch_size: int = 256):
    # get access method
    if access_method == 'sequential':
        indices = np.arange(len(h5_dataset))
    elif access_method == 'random':
        indices = np.arange(len(h5_dataset))
        np.random.shuffle(indices)
    else:
        raise KeyError('Invalid access method')
    # num entries to test
    n = min(len(h5_dataset), max_entries)
    indices = indices[:n]
    # iterate through dataset, exit on timeout or max_entries
    t = Timer()
    for chunk in iter_chunks(enumerate(indices), chunk_size=batch_size):
        with t:
            for i, idx in chunk:
                entry = h5_dataset[idx]
        if t.elapsed > timeout:
            break
    # calculate score
    entries_per_sec = (i + 1) / t.elapsed
    return entries_per_sec


def hdf5_test_speed(h5_path: str, dataset_name: str, access_method: str = 'random'):
    with h5py.File(h5_path, 'r') as out_h5:
        log.info('[TESTING] Access Speed...')
        log.info(f'Random Accesses Per Second: {hdf5_test_entries_per_second(out_h5[dataset_name], access_method=access_method, max_entries=5_000):.3f}')


# ========================================================================= #
# hdf5 - stats                                                              #
# ========================================================================= #


# TODO: cleanup
def hdf5_print_entry_data_stats(h5_dataset: h5py.Dataset, label='STATISTICS'):
    if not isinstance(h5_dataset, h5py.Dataset):
        tqdm.write(
            f'[{label:3s}] '
            f'array: {tuple(h5_dataset.shape)} ({str(h5_dataset.dtype):8s})'
            + (f' ({h5_dataset.device})' if isinstance(h5_dataset, torch.Tensor) else '')
        )
        return
    # get info
    itemsize = _normalize_dtype(h5_dataset.dtype).itemsize
    # entry
    shape = np.array([1, *h5_dataset.shape[1:]])
    data_per_entry = np.prod(shape) * itemsize
    # chunk
    chunks = np.array(h5_dataset.chunks) if (h5_dataset.chunks is not None) else np.ones(h5_dataset.ndim, dtype='int')
    data_per_chunk = np.prod(chunks) * itemsize
    # chunks per entry
    chunks_per_dim = np.ceil(shape / chunks).astype('int')
    chunks_per_entry = np.prod(chunks_per_dim)
    read_data_per_entry = data_per_chunk * chunks_per_entry
    # format
    chunks              = f'{str(list(chunks)):18s}'
    data_per_chunk      = f'{bytes_to_human(data_per_chunk):20s}'
    chunks_per_dim      = f'{str(list(chunks_per_dim)):18s}'
    chunks_per_entry    = f'{chunks_per_entry:5d}'
    read_data_per_entry = f'{bytes_to_human(read_data_per_entry):20s}'
    # format remaining
    entry          = f'{str(list(shape)):18s}'
    data_per_entry = f'{bytes_to_human(data_per_entry)}'
    # print info
    tqdm.write(
        f'[{label:3s}] '
        f'entry: {entry} ({str(h5_dataset.dtype):8s}) {c.lYLW}{data_per_entry}{c.RST} '
        f'chunk: {chunks} {c.YLW}{data_per_chunk}{c.RST} '
        f'chunks per entry: {chunks_per_dim} {c.YLW}{read_data_per_entry}{c.RST} ({c.RED}{chunks_per_entry}{c.RST})  |  '
        f'compression: {repr(h5_dataset.compression)} compression lvl: {repr(h5_dataset.compression_opts)}'
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
