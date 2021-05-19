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

import inspect
import os
from collections import Sized
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import torch_optimizer
from matplotlib import pyplot as plt
from torch.utils.data import BatchSampler
from torch.utils.data import Sampler

from disent.data.groundtruth import Cars3dData
from disent.data.groundtruth import GroundTruthData
from disent.data.groundtruth import Shapes3dData
from disent.data.groundtruth import XYSquaresData
from disent.dataset.groundtruth import GroundTruthDataset
from disent.dataset.groundtruth import GroundTruthDatasetAndFactors
from disent.frameworks.helper.reductions import batch_loss_reduction
from disent.transform import ToStandardisedTensor

from disent.util import TempNumpySeed
from disent.visualize.visualize_util import make_animated_image_grid
from disent.visualize.visualize_util import make_image_grid

# ========================================================================= #
# optimizer                                                                 #
# ========================================================================= #


def make_optimizer(model: torch.nn.Module, name: str = 'sgd', lr=1e-3, weight_decay: float = 0):
    if isinstance(model, torch.nn.Module):
        params = model.parameters()
    elif isinstance(model, torch.Tensor):
        assert model.requires_grad
        params = [model]
    else:
        raise TypeError(f'cannot optimize type: {type(model)}')
    # make optimizer
    if   name == 'sgd':   return torch.optim.SGD(params,       lr=lr, weight_decay=weight_decay)
    elif name == 'sgd_m': return torch.optim.SGD(params,       lr=lr, weight_decay=weight_decay, momentum=0.1)
    elif name == 'adam':  return torch.optim.Adam(params,      lr=lr, weight_decay=weight_decay)
    elif name == 'radam': return torch_optimizer.RAdam(params, lr=lr, weight_decay=weight_decay)
    else: raise KeyError(f'invalid optimizer name: {repr(name)}')


def step_optimizer(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ========================================================================= #
# dataset                                                                   #
# ========================================================================= #


def make_dataset(name: str = 'xysquares', factors: bool = False, data_dir='data/dataset'):
    Sampler = GroundTruthDatasetAndFactors if factors else GroundTruthDataset
    # make dataset
    if   name == 'xysquares':      dataset = Sampler(XYSquaresData(),              transform=ToStandardisedTensor())
    elif name == 'xysquares_1x1':  dataset = Sampler(XYSquaresData(square_size=1), transform=ToStandardisedTensor())
    elif name == 'xysquares_2x2':  dataset = Sampler(XYSquaresData(square_size=2), transform=ToStandardisedTensor())
    elif name == 'xysquares_4x4':  dataset = Sampler(XYSquaresData(square_size=4), transform=ToStandardisedTensor())
    elif name == 'xysquares_8x8':  dataset = Sampler(XYSquaresData(square_size=8), transform=ToStandardisedTensor())
    elif name == 'cars3d':         dataset = Sampler(Cars3dData(data_dir=os.path.join(data_dir, 'cars3d')),   transform=ToStandardisedTensor(size=64))
    elif name == 'shapes3d':       dataset = Sampler(Shapes3dData(data_dir=os.path.join(data_dir, '3dshapes')), transform=ToStandardisedTensor())
    else: raise KeyError(f'invalid data name: {repr(name)}')
    return dataset


def get_single_batch(dataloader, cuda=True):
    for batch in dataloader:
        (x_targ,) = batch['x_targ']
        break
    if cuda:
        x_targ = x_targ.cuda()
    return x_targ


# ========================================================================= #
# images                                                                    #
# ========================================================================= #


def to_img(x: torch.Tensor, scale=False):
    assert x.dtype in {torch.float16, torch.float32, torch.float64, torch.complex32, torch.complex64}, f'unsupported dtype: {x.dtype}'
    x = x.detach().cpu()
    x = torch.abs(x)
    if scale:
        m, M = torch.min(x), torch.max(x)
        x = (x - m) / (M - m)
    x = torch.moveaxis(x, 0, -1)
    x = torch.clamp(x, 0, 1)
    x = (x * 255).to(torch.uint8)
    return x


def show_img(x: torch.Tensor, scale=False, i=None, step=None, show=True):
    if show:
        if (i is None) or (step is None) or (i % step == 0):
            plt.imshow(to_img(x, scale=scale))
            plt.axis('off')
            plt.tight_layout()
            plt.show()


def show_imgs(xs: Sequence[torch.Tensor], scale=False, i=None, step=None, show=True):
    if show:
        if (i is None) or (step is None) or (i % step == 0):
            w = int(np.ceil(np.sqrt(len(xs))))
            h = (len(xs) + w - 1) // w
            fig, axs = plt.subplots(h, w)
            for ax, im in zip(np.array(axs).flatten(), xs):
                ax.imshow(to_img(im, scale=scale))
                ax.set_axis_off()
            plt.tight_layout()
            plt.show()


def grid_img(xs: torch.Tensor, scale=False, pad=8, border=True, bg_color=0.5, num_cols=None) -> np.ndarray:
    assert xs.ndim == 4, 'channels must be: (I, C, H, W)'
    imgs = [to_img(x, scale=scale).cpu().numpy() for x in xs]
    # create grid
    return make_image_grid(imgs, pad=pad, border=border, bg_color=bg_color, num_cols=num_cols)


def grid_animation(xs: torch.Tensor, scale=False, pad=8, border=True, bg_color=0.5, num_cols=None) -> np.ndarray:
    assert xs.ndim == 5, 'channels must be: (I, F, C, H, W)'
    frames = []
    for f in range(xs.shape[1]):
        frames.append(grid_img(xs[:, f, :, :, :], scale=scale, pad=pad, border=border, bg_color=bg_color, num_cols=num_cols))
    return np.array(frames)


# ========================================================================= #
# LOSS                                                                      #
# ========================================================================= #


def _unreduced_mse_loss(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, targ, reduction='none')


def _unreduced_mae_loss(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - targ)


def _unreduced_msae_loss(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - targ) + F.mse_loss(pred, targ, reduction='none')


def unreduced_loss(pred: torch.Tensor, targ: torch.Tensor, mode='mse') -> torch.Tensor:
    return _LOSS_FNS[mode](pred, targ)


_LOSS_FNS = {
    'mse': _unreduced_mse_loss,
    'mae': _unreduced_mae_loss,
    'msae': _unreduced_msae_loss,
}


def pairwise_loss(pred: torch.Tensor, targ: torch.Tensor, mode='mse', mean_dtype=None) -> torch.Tensor:
    # check input
    assert pred.shape == targ.shape
    # mean over final dims
    loss = unreduced_loss(pred=pred, targ=targ, mode=mode)
    loss = batch_loss_reduction(loss, reduction_dtype=mean_dtype, reduction='mean')
    # check result
    assert loss.shape == pred.shape[:1]
    # done
    return loss


# ========================================================================= #
# LOSS                                                                      #
# ========================================================================= #


def unreduced_overlap(pred: torch.Tensor, targ: torch.Tensor, mode='mse') -> torch.Tensor:
    # -ve loss
    return - unreduced_loss(pred=pred, targ=targ, mode=mode)


def pairwise_overlap(pred: torch.Tensor, targ: torch.Tensor, mode='mse', mean_dtype=None) -> torch.Tensor:
    # -ve loss
    return - pairwise_loss(pred=pred, targ=targ, mode=mode, mean_dtype=mean_dtype)


# ========================================================================= #
# sampling helper                                                           #
# ========================================================================= #


def normalise_factor_idx(dataset, factor: Union[int, str]) -> int:
    if isinstance(factor, str):
        try:
            f_idx = dataset.factor_names.index(factor)
        except:
            raise KeyError(f'{repr(factor)} is not one of: {dataset.factor_names}')
    else:
        f_idx = factor
    assert isinstance(f_idx, int)
    assert 0 <= f_idx < dataset.num_factors
    return f_idx


def normalise_factor_idxs(dataset: GroundTruthDataset, factors: Union[Sequence[Union[int, str]], Union[int, str]]) -> np.ndarray:
    if isinstance(factors, (int, str)):
        factors = [factors]
    factors = np.array([normalise_factor_idx(dataset, factor) for factor in factors])
    assert len(set(factors)) == len(factors)
    return factors


def get_factor_idxs(dataset: GroundTruthDataset, factors: Optional[Union[Sequence[Union[int, str]], Union[int, str]]] = None):
    if factors is None:
        return np.arange(dataset.num_factors)
    return normalise_factor_idxs(dataset, factors)


def sample_factors(dataset, num_obs: int = 1024, factor_mode: str = 'sample_random', factor: Union[int, str] = None):
    # sample multiple random factor traversals
    if factor_mode == 'sample_traversals':
        assert factor is not None, f'factor cannot be None when factor_mode=={repr(factor_mode)}'
        # get traversal
        f_idx = normalise_factor_idx(dataset, factor)
        # generate traversals
        factors = []
        for i in range((num_obs + dataset.factor_sizes[f_idx] - 1) // dataset.factor_sizes[f_idx]):
            factors.append(dataset.sample_random_traversal_factors(f_idx=f_idx))
        factors = np.concatenate(factors, axis=0)
    elif factor_mode == 'sample_random':
        factors = dataset.sample_factors(num_obs)
    else:
        raise KeyError
    return factors


def sample_batch_and_factors(dataset, num_samples: int, factor_mode: str = 'sample_random', factor: Union[int, str] = None, device=None):
    factors = sample_factors(dataset, num_obs=num_samples, factor_mode=factor_mode, factor=factor)
    batch = dataset.dataset_batch_from_factors(factors, mode='target').to(device=device)
    factors = torch.from_numpy(factors).to(dtype=torch.float32, device=device)
    return batch, factors


# ========================================================================= #
# mask helper                                                               #
# ========================================================================= #


def make_changed_mask(batch, masked=True):
    if masked:
        mask = torch.zeros_like(batch[0], dtype=torch.bool)
        for i in range(len(batch)):
            mask |= (batch[0] != batch[i])
    else:
        mask = torch.ones_like(batch[0], dtype=torch.bool)
    return mask


# ========================================================================= #
# fn args helper                                                            #
# ========================================================================= #


def _get_fn_from_stack(fn_name: str, stack):
    # -- do we actually need all of this?
    fn = None
    for s in stack:
        if fn_name in s.frame.f_locals:
            fn = s.frame.f_locals[fn_name]
            break
    if fn is None:
        raise RuntimeError(f'could not retrieve function: {repr(fn_name)} from call stack.')
    return fn


def get_caller_params(sort: bool = False, exclude: Sequence[str] = None) -> dict:
    stack = inspect.stack()
    fn_name = stack[1].function
    fn_locals = stack[1].frame.f_locals
    # get function and params
    fn = _get_fn_from_stack(fn_name, stack)
    fn_params = inspect.getfullargspec(fn).args
    # check excluded
    exclude = set() if (exclude is None) else set(exclude)
    fn_params = [p for p in fn_params if (p not in exclude)]
    # sort values
    if sort:
        fn_params = sorted(fn_params)
    # return dict
    return {
        k: fn_locals[k] for k in fn_params
    }


def params_as_string(params: dict, sep: str = '_', names: bool = False):
    # get strings
    if names:
        return sep.join(f"{k}={v}" for k, v in params.items())
    else:
        return sep.join(f"{v}" for k, v in params.items())


# ========================================================================= #
# dataset indices                                                           #
# ========================================================================= #


def sample_unique_batch_indices(num_obs, num_samples) -> np.ndarray:
    assert num_obs >= num_samples, 'not enough values to sample'
    assert (num_obs - num_samples) / num_obs > 0.5, 'this method might be inefficient'
    # get random sample
    indices = set()
    while len(indices) < num_samples:
        indices.update(np.random.randint(low=0, high=num_obs, size=num_samples - len(indices)))
    # make sure indices are randomly ordered
    indices = np.fromiter(indices, dtype=int)
    # indices = np.array(list(indices), dtype=int)
    np.random.shuffle(indices)
    # return values
    return indices


def generate_epoch_batch_idxs(num_obs: int, num_batches: int, mode: str = 'shuffle') -> List[np.ndarray]:
    """
    Generate `num_batches` batches of indices.
    - Each index is in the range [0, num_obs).
    - If num_obs is not divisible by num_batches, then batches may not all be the same size.

    eg. [0, 1, 2, 3, 4] -> [[0, 1], [2, 3], [4]] -- num_obs=5, num_batches=3, sample_mode='range'
    eg. [0, 1, 2, 3, 4] -> [[1, 4], [2, 0], [3]] -- num_obs=5, num_batches=3, sample_mode='shuffle'
    eg. [0, 1, 0, 3, 2] -> [[0, 1], [0, 3], [2]] -- num_obs=5, num_batches=3, sample_mode='random'
    """
    # generate indices
    if mode == 'range':
        idxs = np.arange(num_obs)
    elif mode == 'shuffle':
        idxs = np.arange(num_obs)
        np.random.shuffle(idxs)
    elif mode == 'random':
        idxs = np.random.randint(0, num_obs, size=(num_obs,))
    else:
        raise KeyError(f'invalid mode={repr(mode)}')
    # return batches
    return np.array_split(idxs, num_batches)


def generate_epochs_batch_idxs(num_obs: int, num_epochs: int, num_epoch_batches: int, mode: str = 'shuffle') -> List[np.ndarray]:
    """
    Like generate_epoch_batch_idxs, but concatenate the batches of calling the function `num_epochs` times.
    - The total number of batches returned is: `num_epochs * num_epoch_batches`
    """
    batches = []
    for i in range(num_epochs):
        batches.extend(generate_epoch_batch_idxs(num_obs=num_obs, num_batches=num_epoch_batches, mode=mode))
    return batches


# ========================================================================= #
# Dataloader Sampler Utilities                                              #
# ========================================================================= #


class StochasticSampler(Sampler):
    """
    Sample random batches, not guaranteed to be unique or cover the entire dataset in one epoch!
    """

    def __init__(self, data_source: Union[Sized, int], batch_size: int = 128):
        super().__init__(data_source)
        if isinstance(data_source, int):
            self._len = data_source
        else:
            self._len = len(data_source)
        self._batch_size = batch_size
        assert isinstance(self._len, int)
        assert self._len > 0
        assert isinstance(self._batch_size, int)
        assert self._batch_size > 0

    def __iter__(self):
        while True:
            yield from np.random.randint(0, self._len, size=self._batch_size)


def yield_dataloader(dataloader, steps: int):
    i = 0
    while True:
        for it in dataloader:
            yield it
            i += 1
            if i >= steps:
                return


def StochasticBatchSampler(data_source: Union[Sized, int], batch_size: int):
    return BatchSampler(
        sampler=StochasticSampler(data_source=data_source, batch_size=batch_size),
        batch_size=batch_size,
        drop_last=True
    )


# ========================================================================= #
# Dataset Visualisation / Traversals                                        #
# ========================================================================= #


def dataset_make_traversals(
    gt_data: Union[GroundTruthData, GroundTruthDataset],
    f_idxs=None, factors=None, num_cols=8,
    seed=777,
    transpose_grid=False,
    cyclic=True,
):
    with TempNumpySeed(seed):
        # get defaults
        if not isinstance(gt_data, GroundTruthDataset):
            gt_data = GroundTruthDataset(gt_data)
        # get f_idxs
        f_idxs = get_factor_idxs(gt_data, f_idxs)
        # sample traversals
        images = []
        for f_idx in f_idxs:
            fs = gt_data.sample_random_cycle_factors(f_idx, factors=factors, num=num_cols)
            # TODO: replace with cycle_factor from visualise_util
            if cyclic:
                fs = np.concatenate([fs, fs[1:-1][::-1]], axis=0)
            images.append(gt_data.dataset_batch_from_factors(fs, mode='raw') / 255.0)  # TODO: we shouldn't divide here?
        images = np.stack(images)
    # return grid
    if transpose_grid:
        return np.swapaxes(images, 0, 1)
    return images  # (F, N, H, W, C)


def dataset_make_image_grid(
    gt_data: Union[GroundTruthData, GroundTruthDataset],
    f_idxs=None, factors=None, num_cols=None,
    pad=8, bg_color=1.0, border=False,
    seed=777,
    return_traversals=False,
    transpose_grid=False,
):
    images = dataset_make_traversals(gt_data, f_idxs=f_idxs, factors=factors, num_cols=num_cols, seed=seed, transpose_grid=transpose_grid)
    image = make_image_grid(images.reshape(np.prod(images.shape[:2]), *images.shape[2:]), pad=pad, bg_color=bg_color, border=border, num_cols=images.shape[1])
    if return_traversals:
        return image, images
    return image


def dataset_make_animated_image_grid(
    gt_data: Union[GroundTruthData, GroundTruthDataset],
    f_idxs=None, factors=None, num_cols=None,
    pad=8, bg_color=1.0, border=False,
    seed=777,
    return_traversals=False,
    transpose_grid=False,
    cyclic=True,
):
    images = dataset_make_traversals(gt_data, f_idxs=f_idxs, factors=factors, num_cols=num_cols, seed=seed, transpose_grid=transpose_grid, cyclic=cyclic)
    image = make_animated_image_grid(images, pad=pad, bg_color=bg_color, border=border, num_cols=None)
    if return_traversals:
        return image, images
    return image


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
