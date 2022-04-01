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

#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#
import inspect
import os
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from disent.dataset import DisentDataset
from disent.dataset.data import Cars3d64Data
from disent.dataset.data import DSpritesData
from disent.dataset.data import DSpritesImagenetData
from disent.dataset.data import Mpi3dData
from disent.dataset.data import SmallNorb64Data
from disent.dataset.data import XColumnsData
from disent.dataset.data import XYObjectShadedData
from disent.dataset.data import GroundTruthData
from disent.dataset.data import Shapes3dData
from disent.dataset.data import XYSquaresData
from disent.dataset.data import XYObjectData
from disent.dataset.sampling import BaseDisentSampler
from disent.dataset.sampling import GroundTruthSingleSampler
from disent.dataset.transform import ToImgTensorF32
from disent.dataset.transform import ToImgTensorU8
from docs.examples.extend_experiment.code.groundtruth__xyblocks import XYBlocksData

# disent exports to make life easy
from disent.util.visualize.plot import plt_imshow
from disent.util.visualize.plot import plt_subplots
from disent.util.visualize.plot import plt_subplots_imshow
from disent.util.visualize.plot import plt_hide_axis
from disent.util.visualize.plot import visualize_dataset_traversal


# ========================================================================= #
# dataset                                                                   #
# ========================================================================= #


TransformTypeHint = Union[Literal['uint8'], Literal['float32'], Literal['none']]


def make_transform(mode: Optional[str]) -> Optional[callable]:
    if mode == 'uint8':
        return ToImgTensorU8()
    elif mode == 'float32':
        return ToImgTensorF32()
    elif mode in ('none', None):
        return None
    else:
        raise KeyError(f'invalid transform mode: {repr(mode)}')


# TODO: replace this with the disent registry!
def make_data(
    name: str = 'xysquares',
    factors: bool = False,
    data_root: str = 'data/dataset',
    try_in_memory: bool = False,
    transform_mode: TransformTypeHint = 'float32'
) -> GroundTruthData:
    # make the transform
    transform = make_transform(mode=transform_mode)
    # make data
    if   name == 'xysquares':      data = XYSquaresData(transform=transform)  # equivalent: [xysquares, xysquares_8x8, xysquares_8x8_s8]
    elif name == 'xysquares_1x1':  data = XYSquaresData(square_size=1, transform=transform)
    elif name == 'xysquares_2x2':  data = XYSquaresData(square_size=2, transform=transform)
    elif name == 'xysquares_4x4':  data = XYSquaresData(square_size=4, transform=transform)
    elif name == 'xysquares_8x8':  data = XYSquaresData(square_size=8, transform=transform)  # 8x8x8x8x8x8 = 262144  # equivalent: [xysquares, xysquares_8x8, xysquares_8x8_s8]
    elif name == 'xysquares_8x8_mini':  data = XYSquaresData(square_size=8, grid_spacing=14, transform=transform)  # 5x5x5x5x5x5 = 15625
    # TOY DATASETS
    elif name == 'xysquares_8x8_toy':     data = XYSquaresData(square_size=8, grid_spacing=8, rgb=False, num_squares=1, transform=transform)  # 8x8 = ?
    elif name == 'xysquares_8x8_toy_s1':  data = XYSquaresData(square_size=8, grid_spacing=1, rgb=False, num_squares=1, transform=transform)  # ?x? = ?
    elif name == 'xysquares_8x8_toy_s2':  data = XYSquaresData(square_size=8, grid_spacing=2, rgb=False, num_squares=1, transform=transform)  # ?x? = ?
    elif name == 'xysquares_8x8_toy_s4':  data = XYSquaresData(square_size=8, grid_spacing=4, rgb=False, num_squares=1, transform=transform)  # ?x? = ?
    elif name == 'xysquares_8x8_toy_s8':  data = XYSquaresData(square_size=8, grid_spacing=8, rgb=False, num_squares=1, transform=transform)  # 8x8 = ?
    # TOY DATASETS ALT
    elif name == 'xcolumns_8x_toy':     data = XColumnsData(square_size=8, grid_spacing=8, rgb=False, num_squares=1, transform=transform)  # 8 = ?
    elif name == 'xcolumns_8x_toy_s1':  data = XColumnsData(square_size=8, grid_spacing=1, rgb=False, num_squares=1, transform=transform)  # ? = ?
    elif name == 'xcolumns_8x_toy_s2':  data = XColumnsData(square_size=8, grid_spacing=2, rgb=False, num_squares=1, transform=transform)  # ? = ?
    elif name == 'xcolumns_8x_toy_s4':  data = XColumnsData(square_size=8, grid_spacing=4, rgb=False, num_squares=1, transform=transform)  # ? = ?
    elif name == 'xcolumns_8x_toy_s8':  data = XColumnsData(square_size=8, grid_spacing=8, rgb=False, num_squares=1, transform=transform)  # 8 = ?
    # OVERLAPPING DATASETS
    elif name == 'xysquares_8x8_s1':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=1, transform=transform)  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s2':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=2, transform=transform)  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s3':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=3, transform=transform)  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s4':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=4, transform=transform)  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s5':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=5, transform=transform)  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s6':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=6, transform=transform)  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s7':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=7, transform=transform)  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s8':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=8, transform=transform)  # 8x8x8x8x8x8 = 262144  # equivalent: [xysquares, xysquares_8x8, xysquares_8x8_s8]
    # OTHER SYNTHETIC DATASETS
    elif name == 'xyobject':         data = XYObjectData(transform=transform)
    elif name == 'xyobject_shaded':  data = XYObjectShadedData(transform=transform)
    elif name == 'xyblocks':         data = XYBlocksData(transform=transform)
    # NORMAL DATASETS
    elif name == 'cars3d':         data = Cars3d64Data(data_root=data_root,    prepare=True, transform=transform)
    elif name == 'smallnorb':      data = SmallNorb64Data(data_root=data_root, prepare=True, transform=transform)
    elif name == 'shapes3d':       data = Shapes3dData(data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites':       data = DSpritesData(data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'mpi3d_toy':      data = Mpi3dData(data_root=data_root,  prepare=True, subset='toy',       transform=transform, in_memory=try_in_memory)
    elif name == 'mpi3d_realistic':data = Mpi3dData(data_root=data_root,  prepare=True, subset='realistic', transform=transform, in_memory=try_in_memory)
    elif name == 'mpi3d_real':     data = Mpi3dData(data_root=data_root,  prepare=True, subset='real',      transform=transform, in_memory=try_in_memory)
    # CUSTOM DATASETS
    elif name == 'dsprites_imagenet_bg_100': data = DSpritesImagenetData(visibility=100, mode='bg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_bg_80':  data = DSpritesImagenetData(visibility=80, mode='bg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_bg_60':  data = DSpritesImagenetData(visibility=60, mode='bg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_bg_40':  data = DSpritesImagenetData(visibility=40, mode='bg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_bg_20':  data = DSpritesImagenetData(visibility=20, mode='bg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    #
    elif name == 'dsprites_imagenet_bg_75':  data = DSpritesImagenetData(visibility=75, mode='bg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_bg_50':  data = DSpritesImagenetData(visibility=50, mode='bg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_bg_25':  data = DSpritesImagenetData(visibility=25, mode='bg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_bg_0':   data = DSpritesImagenetData(visibility=0,  mode='bg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)  # same as normal dsprites, but with 3 channels
    # --- #
    elif name == 'dsprites_imagenet_fg_100': data = DSpritesImagenetData(visibility=100, mode='fg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_fg_80':  data = DSpritesImagenetData(visibility=80, mode='fg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_fg_60':  data = DSpritesImagenetData(visibility=60, mode='fg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_fg_40':  data = DSpritesImagenetData(visibility=40, mode='fg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_fg_20':  data = DSpritesImagenetData(visibility=20, mode='fg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    #
    elif name == 'dsprites_imagenet_fg_75':  data = DSpritesImagenetData(visibility=75, mode='fg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_fg_50':  data = DSpritesImagenetData(visibility=50, mode='fg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_fg_25':  data = DSpritesImagenetData(visibility=25, mode='fg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_fg_0':   data = DSpritesImagenetData(visibility=0,  mode='fg', data_root=data_root,  prepare=True, transform=transform, in_memory=try_in_memory)  # same as normal dsprites, but with 3 channels
    # DONE
    else: raise KeyError(f'invalid data name: {repr(name)}')
    # make dataset
    if factors:
        raise NotImplementedError('factor returning is not yet implemented in the rewrite! this needs to be fixed!')  # TODO!
    return data


def make_dataset(
    name: str = 'xysquares',
    factors: bool = False,
    data_root: str = 'data/dataset',
    try_in_memory: bool = False,
    transform_mode: TransformTypeHint = 'float32',
    sampler: BaseDisentSampler = None,
) -> DisentDataset:
    # make data
    data = make_data(
        name=name,
        data_root=data_root,
        try_in_memory=try_in_memory,
        transform_mode='none',  # we move the transform over to the DisentDataset instead!
    )
    return DisentDataset(
        data,
        sampler=GroundTruthSingleSampler() if (sampler is None) else sampler,
        return_indices=True,
        return_factors=factors,
        transform=make_transform(mode=transform_mode),
    )


# ========================================================================= #
# pair samplers                                                             #
# ========================================================================= #


def pair_indices_random(max_idx: int, approx_batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates pairs of indices in corresponding arrays,
    returning random permutations
    - considers [0, 1] and [1, 0] to be different  # TODO: consider them to be the same
    - never returns pairs with the same values, eg. [1, 1]
    - (default) number of returned values is: `max_idx * sqrt(max_idx) / 2`  -- arbitrarily chosen to scale slower than number of combinations
    """
    # defaults
    if approx_batch_size is None:
        approx_batch_size = int(max_idx * (max_idx ** 0.5) / 2)
    # sample values
    idx_a, idx_b = np.random.randint(0, max_idx, size=(2, approx_batch_size))
    # remove similar
    different = (idx_a != idx_b)
    idx_a = idx_a[different]
    idx_b = idx_b[different]
    # return values
    return idx_a, idx_b


def pair_indices_combinations(max_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates pairs of indices in corresponding arrays,
    returning all combinations
    - considers [0, 1] and [1, 0] to be the same, only returns one of them
    - never returns pairs with the same values, eg. [1, 1]
    - number of returned values is: `max_idx * (max_idx-1) / 2`
    """
    # upper triangle excluding diagonal
    # - similar to: `list(itertools.combinations(np.arange(len(t_idxs)), 2))`
    idxs_a, idxs_b = np.triu_indices(max_idx, k=1)
    return idxs_a, idxs_b


def pair_indices_nearby(max_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates pairs of indices in corresponding arrays,
    returning nearby combinations
    - considers [0, 1] and [1, 0] to be the same, only returns one of them
    - never returns pairs with the same values, eg. [1, 1]
    - number of returned values is: `max_idx`
    """
    idxs_a = np.arange(max_idx)                # eg. [0 1 2 3 4 5]
    idxs_b = np.roll(idxs_a, shift=1, axis=0)  # eg. [1 2 3 4 5 0]
    return idxs_a, idxs_b


_PAIR_INDICES_FNS = {
    'random': pair_indices_random,
    'combinations': pair_indices_combinations,
    'nearby': pair_indices_nearby,
}


def pair_indices(max_idx: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        fn = _PAIR_INDICES_FNS[mode]
    except:
        raise KeyError(f'invalid mode: {repr(mode)}')
    return fn(max_idx=max_idx)



# ========================================================================= #
# Files                                                                     #
# ========================================================================= #


def _make_rel_path(*path_segments, is_file=True, _calldepth=0):
    assert not os.path.isabs(os.path.join(*path_segments)), 'path must be relative'
    # get source
    stack = inspect.stack()
    module = inspect.getmodule(stack[_calldepth+1].frame)
    reldir = os.path.dirname(module.__file__)
    # make everything
    path = os.path.join(reldir, *path_segments)
    folder_path = os.path.dirname(path) if is_file else path
    os.makedirs(folder_path, exist_ok=True)
    return path


def _make_rel_path_add_ext(*path_segments, ext='.png', _calldepth=0):
    # make path
    path = _make_rel_path(*path_segments, is_file=True, _calldepth=_calldepth+1)
    if not os.path.splitext(path)[1]:
        path = f'{path}{ext}'
    return path


def make_rel_path(*path_segments, is_file=True):
    return _make_rel_path(*path_segments, is_file=is_file, _calldepth=1)


def make_rel_path_add_ext(*path_segments, ext='.png'):
    return _make_rel_path_add_ext(*path_segments, ext=ext, _calldepth=1)


def plt_rel_path_savefig(rel_path: Optional[str], save: bool = True, show: bool = True, ext='.png', dpi: Optional[int] = None, **kwargs):
    import matplotlib.pyplot as plt
    if save and (rel_path is not None):
        path = _make_rel_path_add_ext(rel_path, ext=ext, _calldepth=2)
        plt.savefig(path, dpi=dpi, **kwargs)
        print(f'saved: {repr(path)}')
    if show:
        plt.show(**kwargs)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
