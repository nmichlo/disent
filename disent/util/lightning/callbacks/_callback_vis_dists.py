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
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch

# TODO: wandb and matplotlib are not in requirements
import matplotlib.pyplot as plt
import wandb

import disent.util.strings.colors as c
from disent.dataset import DisentDataset
from disent.dataset.data import GroundTruthData
from disent.frameworks.ae import Ae
from disent.frameworks.vae import Vae
from disent.util.function import wrapped_partial
from disent.util.iters import chunked
from disent.util.lightning.callbacks._callbacks_base import BaseCallbackPeriodic
from disent.util.lightning.callbacks._helper import _get_dataset_and_ae_like
from disent.util.lightning.logger_util import wb_log_metrics
from disent.util.profiling import Timer
from disent.util.seeds import TempNumpySeed
from disent.util.visualize.plot import plt_subplots_imshow


log = logging.getLogger(__name__)


# ========================================================================= #
# Helper Functions                                                          #
# ========================================================================= #


# helper
def _to_dmat(
    size: int,
    i_a: np.ndarray,
    i_b: np.ndarray,
    dists: Union[torch.Tensor, np.ndarray],
) -> np.ndarray:
    if isinstance(dists, torch.Tensor):
        dists = dists.detach().cpu().numpy()
    # checks
    assert i_a.ndim == 1
    assert i_a.shape == i_b.shape
    assert i_a.shape == dists.shape
    # compute
    dmat = np.zeros([size, size], dtype='float32')
    dmat[i_a, i_b] = dists
    dmat[i_b, i_a] = dists
    return dmat


_AE_DIST_NAMES = ('x', 'z', 'x_recon')
_VAE_DIST_NAMES = ('x', 'z', 'kl', 'x_recon')


@torch.no_grad()
def _get_dists_ae(ae: Ae, x_a: torch.Tensor, x_b: torch.Tensor):
    # feed forware
    z_a, z_b = ae.encode(x_a), ae.encode(x_b)
    r_a, r_b = ae.decode(z_a), ae.decode(z_b)
    # distances
    return [
        ae.recon_handler.compute_pairwise_loss(x_a, x_b),
        torch.norm(z_a - z_b, p=2, dim=-1),  # l2 dist
        ae.recon_handler.compute_pairwise_loss(r_a, r_b),
    ]


@torch.no_grad()
def _get_dists_vae(vae: Vae, x_a: torch.Tensor, x_b: torch.Tensor):
    from torch.distributions import kl_divergence
    # feed forward
    (z_post_a, z_prior_a), (z_post_b, z_prior_b) = vae.encode_dists(x_a), vae.encode_dists(x_b)
    z_a, z_b = z_post_a.mean, z_post_b.mean
    r_a, r_b = vae.decode(z_a), vae.decode(z_b)
    # dists
    kl_ab = 0.5 * kl_divergence(z_post_a, z_post_b) + 0.5 * kl_divergence(z_post_b, z_post_a)
    # distances
    return [
        vae.recon_handler.compute_pairwise_loss(x_a, x_b),
        torch.norm(z_a - z_b, p=2, dim=-1),  # l2 dist
        vae.recon_handler._pairwise_reduce(kl_ab),
        vae.recon_handler.compute_pairwise_loss(r_a, r_b),
    ]


def _get_dists_fn(model: Ae) -> Tuple[Optional[Tuple[str, ...]], Optional[Callable[[object, object], Sequence[Sequence[float]]]]]:
    # get aggregate function
    if isinstance(model, Vae):
        dists_names, dists_fn = _VAE_DIST_NAMES, wrapped_partial(_get_dists_vae, model)
    elif isinstance(model, Ae):
        dists_names, dists_fn = _AE_DIST_NAMES, wrapped_partial(_get_dists_ae, model)
    else:
        dists_names, dists_fn = None, None
    return dists_names, dists_fn


@torch.no_grad()
def _collect_dists_subbatches(dists_fn: Callable[[object, object], Sequence[Sequence[float]]], batch: torch.Tensor, i_a: np.ndarray, i_b: np.ndarray, batch_size: int = 64):
    # feed forward
    results = []
    for idxs in chunked(np.stack([i_a, i_b], axis=-1), chunk_size=batch_size):
        ia, ib = idxs.T
        x_a, x_b = batch[ia], batch[ib]
        # feed forward
        data = dists_fn(x_a, x_b)
        results.append(data)
    return [torch.cat(r, dim=0) for r in zip(*results)]


def _compute_and_collect_dists(
    dataset: DisentDataset,
    dists_fn,
    dists_names: Sequence[str],
    traversal_repeats: int = 100,
    batch_size: int = 32,
    include_gt_factor_dists: bool = True,
    transform_batch: Callable[[object], object] = None,
    data_mode: str = 'input',
) -> Tuple[Tuple[str, ...], List[List[np.ndarray]]]:
    assert traversal_repeats > 0
    gt_data = dataset.gt_data
    # generate
    f_grid = []
    # generate
    for f_idx, f_size in enumerate(gt_data.factor_sizes):
        # save for the current factor (traversal_repeats, len(names), len(i_a))
        f_dists = []
        # upper triangle excluding diagonal
        i_a, i_b = np.triu_indices(f_size, k=1)
        # repeat over random traversals
        for i in range(traversal_repeats):
            # get random factor traversal
            factors = gt_data.sample_random_factor_traversal(f_idx=f_idx)
            indices = gt_data.pos_to_idx(factors)
            # load data
            batch = dataset.dataset_batch_from_indices(indices, data_mode)
            if transform_batch is not None:
                batch = transform_batch(batch)
            # feed forward & compute dists -- (len(names), len(i_a))
            dists = _collect_dists_subbatches(dists_fn=dists_fn, batch=batch, i_a=i_a, i_b=i_b, batch_size=batch_size)
            assert len(dists) == len(dists_names)
            # distances
            f_dists.append(dists)
        # aggregate all dists into distances matrices for current factor
        f_dmats = [
            _to_dmat(size=f_size, i_a=i_a, i_b=i_b, dists=torch.stack(dists, dim=0).mean(dim=0))
            for dists in zip(*f_dists)
        ]
        # handle factors
        if include_gt_factor_dists:
            i_dmat = _to_dmat(size=f_size, i_a=i_a, i_b=i_b, dists=np.abs(factors[i_a] - factors[i_b]).sum(axis=-1))
            f_dmats = [i_dmat, *f_dmats]
        # append data
        f_grid.append(f_dmats)
    # handle factors
    if include_gt_factor_dists:
        dists_names = ('factors', *dists_names)
    # done
    return tuple(dists_names), f_grid


def compute_factor_distances(
    dataset: DisentDataset,
    dists_fn,
    dists_names: Sequence[str],
    traversal_repeats: int = 100,
    batch_size: int = 32,
    include_gt_factor_dists: bool = True,
    transform_batch: Callable[[object], object] = None,
    seed: Optional[int] = 777,
    data_mode: str = 'input',
) -> Tuple[Tuple[str, ...], List[List[np.ndarray]]]:
    # log this callback
    gt_data = dataset.gt_data
    log.info(f'| {gt_data.name} - computing factor distances...')
    # compute various distances matrices for each factor
    with Timer() as timer, TempNumpySeed(seed):
        dists_names, f_grid = _compute_and_collect_dists(
            dataset=dataset,
            dists_fn=dists_fn,
            dists_names=dists_names,
            traversal_repeats=traversal_repeats,
            batch_size=batch_size,
            include_gt_factor_dists=include_gt_factor_dists,
            transform_batch=transform_batch,
            data_mode=data_mode,
        )
    # log this callback!
    log.info(f'| {gt_data.name} - computed factor distances! time{c.GRY}={c.lYLW}{timer.pretty:<9}{c.RST}')
    return dists_names, f_grid


def plt_factor_distances(
    gt_data: GroundTruthData,
    f_grid: List[List[np.ndarray]],
    dists_names: Sequence[str],
    title: str,
    plt_block_size: float = 1.25,
    plt_transpose: bool = False,
    plt_cmap='Blues',
):
    # plot information
    imshow_kwargs = dict(cmap=plt_cmap)
    figsize       = (plt_block_size*len(f_grid[0]), plt_block_size * gt_data.num_factors)
    # plot!
    if not plt_transpose:
        fig, axs = plt_subplots_imshow(grid=f_grid,             col_labels=dists_names,          row_labels=gt_data.factor_names, figsize=figsize,       title=title, imshow_kwargs=imshow_kwargs)
    else:
        fig, axs = plt_subplots_imshow(grid=list(zip(*f_grid)), col_labels=gt_data.factor_names, row_labels=dists_names,          figsize=figsize[::-1], title=title, imshow_kwargs=imshow_kwargs)
    # done
    return fig, axs


# ========================================================================= #
# Data Dists Visualisation Callback                                         #
# ========================================================================= #


class VaeGtDistsLoggingCallback(BaseCallbackPeriodic):

    def __init__(
        self,
        seed: Optional[int] = 7777,
        every_n_steps: Optional[int] = None,
        traversal_repeats: int = 100,
        begin_first_step: bool = False,
        plt_block_size: float = 1.25,
        plt_show: bool = False,
        plt_transpose: bool = False,
        log_wandb: bool = True,  # TODO: detect this automatically?
        batch_size: int = 128,
        include_factor_dists: bool = True,
    ):
        assert traversal_repeats > 0
        self._traversal_repeats = traversal_repeats
        self._seed = seed
        self._plt_block_size = plt_block_size
        self._plt_show = plt_show
        self._log_wandb = log_wandb
        self._include_gt_factor_dists = include_factor_dists
        self._transpose_plot = plt_transpose
        self._batch_size = batch_size
        super().__init__(every_n_steps, begin_first_step)

    @torch.no_grad()
    def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # exit early
        if not (self._plt_show or self._log_wandb):
            log.warning(f'skipping {self.__class__.__name__} neither `plt_show` or `log_wandb` is `True`!')
            return
        # get dataset and vae framework from trainer and module
        dataset, vae = _get_dataset_and_ae_like(trainer, pl_module, unwrap_groundtruth=True)
        # exit early
        if not dataset.is_ground_truth:
            log.warning(f'cannot run {self.__class__.__name__} over non-ground-truth data, skipping!')
            return
        # get aggregate function
        dists_names, dists_fn = _get_dists_fn(vae)
        if (dists_names is None) or (dists_fn is None):
            log.warning(f'cannot run {self.__class__.__name__}, unsupported model type: {type(vae)}, must be {Ae.__name__} or {Vae.__name__}')
            return
        # compute various distances matrices for each factor
        dists_names, f_grid = compute_factor_distances(
            dataset=dataset,
            dists_fn=dists_fn,
            dists_names=dists_names,
            traversal_repeats=self._traversal_repeats,
            batch_size=self._batch_size,
            include_gt_factor_dists=self._include_gt_factor_dists,
            transform_batch=lambda batch: batch.to(vae.device),
            seed=self._seed,
            data_mode='input',
        )
        # plot these results
        fig, axs = plt_factor_distances(
            gt_data=dataset.gt_data,
            f_grid=f_grid,
            dists_names=dists_names,
            title=f'{vae.__class__.__name__}: {dataset.gt_data.name.capitalize()} Distances',
            plt_block_size=self._plt_block_size,
            plt_transpose=self._transpose_plot,
            plt_cmap='Blues',
        )
        # show the plot
        if self._plt_show:
            plt.show()
        # log the plot to wandb
        if self._log_wandb:
            wb_log_metrics(trainer.logger, {
                'factor_distances': wandb.Image(fig)
            })


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
