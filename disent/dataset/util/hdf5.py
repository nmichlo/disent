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

import contextlib
import logging
import os
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from disent.util.deprecate import deprecated
from disent.util.strings import colors as c
from disent.util.inout.files import AtomicSaveFile
from disent.util.iters import iter_chunks
from disent.util.profiling import Timer
from disent.util.strings.fmt import bytes_to_human


log = logging.getLogger(__name__)


# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!
# TODO: this file needs to be cleaned up!!!


# ========================================================================= #
# hdf5 - arguments                                                          #
# ========================================================================= #


AnyDType = Union[torch.dtype, np.dtype, str]


def _normalize_dtype(dtype: AnyDType) -> np.dtype:
    if isinstance(dtype, torch.dtype):
        dtype: str = torch.finfo(torch.float32).dtype
    return np.dtype(dtype)


ChunksType = Union[Tuple[int, ...], Literal['auto'], Literal['batch']]


def _normalize_chunks(chunks: ChunksType, shape: Tuple[int, ...]):
    if chunks == 'auto':
        return True
    elif chunks == 'batch':
        return (1, *shape[1:])
    elif isinstance(chunks, tuple):
        return chunks
    else:
        raise ValueError(f'invalid chunks value: {repr(chunks)}')


def _normalize_compression(compression_lvl: Optional[int]):
    if compression_lvl is None:
        return None, None  # compression, compression_lvl
    # check compression level
    if compression_lvl not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        raise ValueError('compression_lvl must be an interger in the range [0, 9]')
    # get values
    return 'gzip', compression_lvl  # compression, compression_lvl


# ========================================================================= #
# hdf5 - helper                                                             #
# ========================================================================= #


class H5IncompatibleError(Exception):
    pass


def h5_assert_deterministic(h5_file: h5py.File) -> h5py.File:
    # check the version
    if (isinstance(h5_file.libver, str) and h5_file.libver != 'earliest') or (h5_file.libver[0] != 'earliest'):
        raise H5IncompatibleError(f'hdf5 out file has an incompatible libver: {repr(h5_file.libver)} libver should be set to: "earliest"')
    return h5_file


# ========================================================================= #
# hdf5 - resave                                                             #
# ========================================================================= #


@contextlib.contextmanager
def h5_open(path: str, mode: str = 'r') -> h5py.File:
    """
    MODES:
        atomic_w Create temp file, then move and overwrite existing when done
        atomic_x Create temp file, then try move or fail if existing when done
        r        Readonly, file must exist (default)
        r+       Read/write, file must exist
        w        Create file, truncate if exists
        w- or x  Create file, fail if exists
        a        Read/write if exists, create otherwise
    """
    assert str.endswith(path, '.h5') or str.endswith(path, '.hdf5'), f'hdf5 file path does not end with extension: `.h5` or `.hdf5`, got: {path}'
    # get atomic context manager
    if mode == 'atomic_w':
        save_context, mode = AtomicSaveFile(path, open_mode=None, overwrite=True), 'w'
    elif mode == 'atomic_x':
        save_context, mode = AtomicSaveFile(path, open_mode=None, overwrite=False), 'x'
    else:
        save_context = contextlib.nullcontext(path)
    # handle saving to file
    with save_context as tmp_h5_path:
        with h5py.File(tmp_h5_path, mode, libver='earliest') as h5_file:
            yield h5_file


class H5Builder(object):

    def __init__(self, path: Union[str, Path], mode: str = 'x'):
        super().__init__()
        # make sure that the file is deterministic
        # - we might be missing some of the properties that control this
        # - should we add a recursive option?
        if not isinstance(path, (str, Path)):
            raise TypeError(f'the given h5py path must be of type: `str`, `pathlib.Path`, got: {type(path)}')
        self._h5_path = path
        self._h5_mode = mode
        self._context_manager = None
        self._open_file = None

    def __enter__(self):
        self._context_manager = h5_open(self._h5_path, self._h5_mode)
        self._open_file = h5_assert_deterministic(self._context_manager.__enter__())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context_manager.__exit__(exc_type, exc_val, exc_tb)
        self._open_file = None
        self._context_manager = None

    @property
    def _h5_file(self) -> h5py.File:
        if self._open_file is None:
            raise 'The H5Builder has not been opened in a new context, use `with H5Builder(...) as builder: ...`'
        return self._open_file

    def add_dataset(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: AnyDType,
        chunk_shape: ChunksType = 'batch',
        compression_lvl: Optional[int] = 9,
        attrs: Optional[Dict[str, Any]] = None
    ) -> 'H5Builder':
        # normalize chunk_shape
        compression, compression_lvl = _normalize_compression(compression_lvl=compression_lvl)
        # create new dataset
        dataset = self._h5_file.create_dataset(
            name=name,
            shape=shape,
            dtype=_normalize_dtype(dtype),
            chunks=_normalize_chunks(chunk_shape, shape=shape),
            compression=compression,
            compression_opts=compression_lvl,
            # non-deterministic time stamps are added to the file if this is not
            # disabled, resulting in different hash sums when the file is re-generated!
            # - https://github.com/h5py/h5py/issues/225
            # - https://stackoverflow.com/questions/16019656
            # other properties:
            # - https://docs.h5py.org/en/stable/high/group.html#h5py.Group.create_dataset
            track_times=False,
            # how do these affect determinism:
            # track_order=False,
            # fletcher32=True,  # checksum for each chunk
            # shuffle=True,     # reorder chunk values to possibly help compression
            # scaleoffset=<int> # enable lossy compression, ints: number of bits to keep (0 is automatic lossless), floats: number of digits after decimal
        )
        # add atttributes & convert
        if attrs is not None:
            for key, value in attrs.items():
                if isinstance(value, str):
                    value = np.array(value, dtype='S')
                dataset.attrs[key] = value
        # done!
        return self

    def fill_dataset(
        self,
        name: str,
        get_batch_fn: Callable[[int, int], np.ndarray],  # i_start, i_end
        batch_size: Union[int, Literal['auto']] = 'auto',
        show_progress: bool = False,
    ) -> 'H5Builder':
        dataset: h5py.Dataset = self._h5_file[name]
        # determine batch size for copying data
        # get smallest multiple less than 32, otherwise original number
        if batch_size == 'auto':
            if dataset.chunks:
                batch_size = dataset.chunks[0]
                batch_size = max((32 // batch_size) * batch_size, batch_size)
            else:
                batch_size = 32
        else:
            if dataset.chunks:
                if batch_size % dataset.chunks[0] != 0:
                    log.warning(f'batch_size={batch_size} is not divisible by the first dimension of the dataset chunk size: {dataset.chunks[0]} {tuple(dataset.chunks)}')
        # check batch size!
        assert isinstance(batch_size, int) and (batch_size >= 1), f'invalid batch_size: {repr(batch_size)}, expected: "auto" or an integer `>= 1`'
        # loop variables
        n = len(dataset)
        # save data
        with tqdm(total=n, disable=not show_progress, desc=f'saving {name}') as progress:
            for i in range(0, n, batch_size):
                j = min(i + batch_size, n)
                assert j > i, f'this is a bug! {repr(j)} > {repr(i)}, len(dataset)={repr(n)}, batch_size={repr(batch_size)}'
                # load and modify the batch
                batch = get_batch_fn(i, j)  # i_start, i_end
                assert isinstance(batch, np.ndarray), f'returned batch is not an `np.ndarray`, got: {repr(type(batch))}'
                assert batch.shape == (j-i, *dataset.shape[1:]), f'returned batch has incorrect shape: {tuple(batch.shape)}, expected: {(j-i, *dataset.shape[1:])}'
                # save the batch & update progress
                dataset[i:j] = batch
                progress.update(j-i)
        # done!
        return self

    def fill_dataset_from_array(
        self,
        name: str,
        array,
        batch_size: Union[int, Literal['auto']] = 'auto',
        show_progress: bool = False,
        mutator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> 'H5Builder':
        # get the array extractor
        if isinstance(array, torch.Tensor):
            @torch.no_grad()
            def _extract_fn(i, j): return array[i:j].cpu().numpy()
        elif isinstance(array, (np.ndarray, h5py.Dataset)):  # chunk sizes will be missmatched
            def _extract_fn(i, j): return array[i:j]
        elif isinstance(array, (tuple, list)):
            def _extract_fn(i, j): return array[i:j]
        elif isinstance(array, Sequence):
            def _extract_fn(i, j): return [array[k] for k in range(i, j)]
        else:
            # last ditch effort, try as an iterator
            try:
                array = iter(array)
            except:
                raise TypeError(f'`fill_dataset_from_array` only supports arrays of type: `np.ndarray` or `torch.Tensor`')
            # get iterator function
            def _extract_fn(i, j): return [next(array) for k in range(i, j)]

        # get the batch fn
        def get_batch_fn(i, j):
            batch = _extract_fn(i, j)
            if mutator:
                batch = mutator(batch)
            return np.array(batch)

        # get the batch size
        if batch_size == 'auto' and isinstance(array, h5py.Dataset):
            batch_size = array.chunks[0]

        # copy into the dataset
        self.fill_dataset(
            name=name,
            get_batch_fn=get_batch_fn,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        return self

    def fill_dataset_from_batches(
        self,
        name: str,
        batch_iter,
        batch_size: Union[int, Literal['auto']] = 'auto',
        show_progress: bool = False,
        mutator: Optional[Callable[[Any], np.ndarray]] = None,
    ) -> 'H5Builder':
        try:
            batches = iter(batch_iter)
        except:
            raise TypeError(f'`fill_dataset_from_batches` must have iterable `batch_iter`, got: {type(batch_iter)}')
        # produce items
        def get_batch_fn(i, j):
            batch = next(batches)
            assert len(batch) == (j-i)
            if mutator:
                batch = mutator(batch)
            return np.array(batch)
        # copy into the dataset
        self.fill_dataset(
            name=name,
            get_batch_fn=get_batch_fn,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        return self

    def add_dataset_from_array(
        self,
        name: str,
        array: np.ndarray,
        chunk_shape: ChunksType = 'batch',
        compression_lvl: Optional[int] = 4,
        attrs: Optional[Dict[str, Any]] = None,
        batch_size: Union[int, Literal['auto']] = 'auto',
        show_progress: bool = False,
        # optional, discovered automatically from array otherwise
        mutator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        dtype: Optional[np.dtype] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ):
        self.add_dataset(
            name=name,
            shape=array.shape if (shape is None) else shape,
            dtype=array.dtype if (dtype is None) else dtype,
            chunk_shape=chunk_shape,
            compression_lvl=compression_lvl,
            attrs=attrs,
        )
        self.fill_dataset_from_array(
            name=name,
            array=array,
            batch_size=batch_size,
            show_progress=show_progress,
            mutator=mutator,
        )

    def add_dataset_from_gt_data(
        self,
        data: Union['DisentDataset', 'GroundTruthData'],
        mutator: Optional[Callable[[Any], np.ndarray]] = None,
        img_shape: Tuple[Optional[int], ...] = (None, None, None),  # None items are automatically found
        batch_size: int = 32,
        compression_lvl: Optional[int] = 9,
        num_workers: int = min(os.cpu_count(), 16),
        show_progress: bool = True,
        dtype: str = 'uint8',
        attrs: Optional[dict] = None
    ):
        from disent.dataset import DisentDataset
        from disent.dataset.data import GroundTruthData
        # get dataset
        if isinstance(data, DisentDataset): gt_data = data.gt_data
        elif isinstance(data, GroundTruthData): gt_data = data
        else: raise TypeError(f'invalid data type: {type(data)}, must be {DisentDataset} or {GroundTruthData}')
        # magic vars
        name = 'data'
        # process image shape
        H, W, C = img_shape
        if H is None: H = gt_data.img_shape[0]
        if W is None: W = gt_data.img_shape[1]
        if C is None: C = gt_data.img_shape[2]
        # make the empty dataset
        self.add_dataset(
            name=name,
            shape=(len(gt_data), H, W, C),
            dtype=dtype,
            chunk_shape='batch',
            compression_lvl=compression_lvl,
            # THESE ATTRIBUTES SHOULD MATCH: SelfContainedHdf5GroundTruthData
            attrs=dict(
                dataset_name=gt_data.name,
                dataset_cls_name=gt_data.__class__.__name__,
                factor_sizes=np.array(gt_data.factor_sizes, dtype='uint'),
                factor_names=np.array(gt_data.factor_names, dtype='S'),
                # extra attrs -- we can't overwrite the above
                **(attrs if (attrs is not None) else {}),
            )
        )
        # fill the dataset!
        self.fill_dataset_from_batches(
            name=name,
            batch_iter=DataLoader(gt_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False),
            batch_size=batch_size,
            show_progress=show_progress,
            mutator=mutator,
        )

#     def resave_dataset(self,
#         name: str,
#         inp: Union[str, Path, h5py.File, h5py.Dataset, np.ndarray],
#         # h5 re-save settings
#         chunk_shape: ChunksType = 'batch',
#         compression_lvl: Optional[int] = 4,
#         attrs: Optional[Dict[str, Any]] = None,
#         batch_size: Union[int, Literal['auto']] = 'auto',
#         show_progress: bool = False,
#         # optional, discovered automatically from array otherwise
#         mutator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
#         dtype: Optional[np.dtype] = None,
#         obs_shape: Optional[Tuple[int, ...]] = None,
#     ):
#         # TODO: should this be more general and be able to handle add_dataset_from_gt_data too?
#         # TODO: this is very similar to save dataset below!
#         with _get_array_context(inp, name) as arr:
#             self.add_dataset_from_array(
#                 name=name,
#                 array=arr,
#                 chunk_shape=chunk_shape,
#                 compression_lvl=compression_lvl,
#                 attrs=attrs,
#                 batch_size=batch_size,
#                 show_progress=show_progress,
#                 mutator=mutator,
#                 dtype=dtype,
#                 shape=(len(arr), *obs_shape) if obs_shape else None,
#             )
#
#
# @contextlib.contextmanager
# def _get_array_context(
#     inp: Union[str, Path, h5py.File, h5py.Dataset, np.ndarray],
#     dataset_name: str = None,
# ) -> Union[h5py.Dataset, np.ndarray]:
#     # check the inputs
#     if not isinstance(inp, (str, Path, h5py.File, h5py.Dataset, np.ndarray)):
#         raise TypeError(f'unsupported input type: {type(inp)}')
#     # handle loading files
#     if isinstance(inp, str):
#         _, ext = os.path.splitext(inp)
#         if ext in ('.h5', '.hdf5'):
#             inp_context = h5py.File(inp, 'r')
#         else:
#             raise ValueError(f'unsupported extension: {repr(ext)} for path: {repr(inp)}')
#     else:
#         import contextlib
#         inp_context = contextlib.nullcontext(inp)
#     # re-save datasets
#     with inp_context as inp_data:
#         # get input dataset from h5 file
#         if isinstance(inp_data, h5py.File):
#             if dataset_name is None:
#                 raise ValueError('dataset_name must be specified if the input is an h5py.File so we can retrieve a h5py.Dataset')
#             inp_data = inp_data[dataset_name]
#         # return the data
#         yield inp_data


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
            assert batch.shape[1:] == obs_shape, f'obs shape: {tuple(batch.shape[1:])} from processed input data does not match required obs shape: {tuple(obs_shape)}, try changing the `obs_shape` or resizing the batch in the `out_mutator`.'
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
