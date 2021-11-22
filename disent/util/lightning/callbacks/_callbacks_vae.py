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
import warnings
from typing import Callable
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch

from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

import disent.metrics
import disent.util.strings.colors as c
from disent.dataset import DisentDataset
from disent.dataset.data import GroundTruthData
from disent.frameworks.ae import Ae
from disent.frameworks.helper.reconstructions import make_reconstruction_loss
from disent.frameworks.helper.reconstructions import ReconLossHandler
from disent.frameworks.vae import Vae
from disent.util.function import wrapped_partial
from disent.util.iters import chunked
from disent.util.lightning.callbacks._callbacks_base import BaseCallbackPeriodic
from disent.util.lightning.logger_util import log_metrics
from disent.util.lightning.logger_util import wb_log_metrics
from disent.util.lightning.logger_util import wb_log_reduced_summaries
from disent.util.profiling import Timer
from disent.util.seeds import TempNumpySeed
from disent.util.visualize.plot import plt_subplots_imshow
from disent.util.visualize.vis_model import latent_cycle_grid_animation
from disent.util.visualize.vis_util import make_image_grid

# TODO: wandb and matplotlib are not in requirements
import matplotlib.pyplot as plt
import wandb


log = logging.getLogger(__name__)


# ========================================================================= #
# Helper Functions                                                          #
# ========================================================================= #


def _get_dataset_and_vae(trainer: pl.Trainer, pl_module: pl.LightningModule, unwrap_groundtruth: bool = False) -> (DisentDataset, Ae):
    # TODO: improve handling!
    assert isinstance(pl_module, Ae), f'{pl_module.__class__} is not an instance of {Ae}'
    # get dataset
    if hasattr(trainer, 'datamodule') and (trainer.datamodule is not None):
        assert hasattr(trainer.datamodule, 'dataset_train_noaug')  # TODO: this is for experiments, another way of handling this should be added
        dataset = trainer.datamodule.dataset_train_noaug
    elif hasattr(trainer, 'train_dataloader') and (trainer.train_dataloader is not None):
        if isinstance(trainer.train_dataloader, CombinedLoader):
            dataset = trainer.train_dataloader.loaders.dataset
        else:
            raise RuntimeError(f'invalid trainer.train_dataloader: {trainer.train_dataloader}')
    else:
        raise RuntimeError('could not retrieve dataset! please report this...')
    # check dataset
    assert isinstance(dataset, DisentDataset), f'retrieved dataset is not an {DisentDataset.__name__}'
    # unwarp dataset
    if unwrap_groundtruth:
        if dataset.is_wrapped_gt_data:
            old_dataset, dataset = dataset, dataset.unwrapped_disent_dataset()
            warnings.warn(f'Unwrapped ground truth dataset returned! {type(old_dataset.data).__name__} -> {type(dataset.data).__name__}')
    # done checks
    return dataset, pl_module


# ========================================================================= #
# Vae Framework Callbacks                                                   #
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
        log_wandb: bool = True,
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
        # get dataset and vae framework from trainer and module
        dataset, vae = _get_dataset_and_vae(trainer, pl_module, unwrap_groundtruth=True)
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


def _normalize_min_max_mean_std_to_min_max(recon_min, recon_max, recon_mean, recon_std) -> Union[Tuple[None, None], Tuple[np.ndarray, np.ndarray]]:
    # check recon_min and recon_max
    if (recon_min is not None) or (recon_max is not None):
        if (recon_mean is not None) or (recon_std is not None):
            raise ValueError('must choose either recon_min & recon_max OR recon_mean & recon_std, cannot specify both')
        if (recon_min is None) or (recon_max is None):
            raise ValueError('both recon_min & recon_max must be specified')
        # check strings
        if isinstance(recon_min, str) or isinstance(recon_max, str):
            if not (isinstance(recon_min, str) and isinstance(recon_max, str)):
                raise ValueError('both recon_min & recon_max must be "auto" if one is "auto"')
            return None, None
    # check recon_mean and recon_std
    elif (recon_mean is not None) or (recon_std is not None):
        if (recon_min is not None) or (recon_max is not None):
            raise ValueError('must choose either recon_min & recon_max OR recon_mean & recon_std, cannot specify both')
        if (recon_mean is None) or (recon_std is None):
            raise ValueError('both recon_mean & recon_std must be specified')
        # set values:
        #  | ORIG: [0, 1]
        #  | TRANSFORM: (x - mean) / std         ->  [(0-mean)/std, (1-mean)/std]
        #  | REVERT:    (x - min) / (max - min)  ->  [0, 1]
        #  |            min=(0-mean)/std, max=(1-mean)/std
        recon_mean, recon_std = np.array(recon_mean, dtype='float32'), np.array(recon_std, dtype='float32')
        recon_min = np.divide(0 - recon_mean, recon_std)
        recon_max = np.divide(1 - recon_mean, recon_std)
    # set defaults
    if recon_min is None: recon_min = 0.0
    if recon_max is None: recon_max = 0.0
    # change type
    recon_min = np.array(recon_min)
    recon_max = np.array(recon_max)
    assert recon_min.ndim in (0, 1)
    assert recon_max.ndim in (0, 1)
    # checks
    assert np.all(recon_min < np.all(recon_max)), f'recon_min={recon_min} must be less than recon_max={recon_max}'
    return recon_min, recon_max


class VaeLatentCycleLoggingCallback(BaseCallbackPeriodic):

    def __init__(
        self,
        seed: Optional[int] = 7777,
        every_n_steps: Optional[int] = None,
        begin_first_step: bool = False,
        num_frames: int = 17,
        mode: str = 'fitted_gaussian_cycle',
        wandb_mode: str = 'both',
        wandb_fps: int = 4,
        plt_show: bool = False,
        plt_block_size: float = 1.0,
        # recon_min & recon_max
        recon_min: Optional[Union[int, Literal['auto']]] = None,       # scale data in this range [min, max] to [0, 1]
        recon_max: Optional[Union[int, Literal['auto']]] = None,       # scale data in this range [min, max] to [0, 1]
        recon_mean: Optional[Union[Tuple[float, ...], float]] = None,  # automatically converted to min & max [(0-mean)/std, (1-mean)/std], assuming original range of values is [0, 1]
        recon_std: Optional[Union[Tuple[float, ...], float]] = None,   # automatically converted to min & max [(0-mean)/std, (1-mean)/std], assuming original range of values is [0, 1]
    ):
        super().__init__(every_n_steps, begin_first_step)
        self.seed = seed
        self.mode = mode
        self.plt_show = plt_show
        self.plt_block_size = plt_block_size
        self._wandb_mode = wandb_mode
        self._num_frames = num_frames
        self._fps = wandb_fps
        # checks
        assert wandb_mode in {'none', 'img', 'vid', 'both'}, f'invalid wandb_mode={repr(wandb_mode)}, must be one of: ("none", "img", "vid", "both")'
        # normalize
        self._recon_min, self._recon_max = _normalize_min_max_mean_std_to_min_max(
            recon_min=recon_min,
            recon_max=recon_max,
            recon_mean=recon_mean,
            recon_std=recon_std,
        )


    def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # get dataset and vae framework from trainer and module
        dataset, vae = _get_dataset_and_vae(trainer, pl_module, unwrap_groundtruth=True)

        # TODO: should this not use `visualize_dataset_traversal`?

        with torch.no_grad():
            # get random sample of z_means and z_logvars for computing the range of values for the latent_cycle
            with TempNumpySeed(self.seed):
                batch = dataset.dataset_sample_batch(64, mode='input').to(vae.device)

            # get representations
            if isinstance(vae, Vae):
                # variational auto-encoder
                ds_posterior, ds_prior = vae.encode_dists(batch)
                zs_mean, zs_logvar = ds_posterior.mean, torch.log(ds_posterior.variance)
            elif isinstance(vae, Ae):
                # auto-encoder
                zs_mean = vae.encode(batch)
                zs_logvar = torch.ones_like(zs_mean)
            else:
                log.warning(f'cannot run {self.__class__.__name__}, unsupported type: {type(vae)}, must be {Ae.__name__} or {Vae.__name__}')
                return

            # get min and max if auto
            if (self._recon_min is None) or (self._recon_max is None):
                if self._recon_min is None: self._recon_min = float(torch.min(batch).cpu())
                if self._recon_max is None: self._recon_max = float(torch.max(batch).cpu())
                log.info(f'auto visualisation min: {self._recon_min} and max: {self._recon_max} obtained from {len(batch)} samples')

            # produce latent cycle grid animation
            # TODO: this needs to be fixed to not use logvar, but rather the representations or distributions themselves
            animation, stills = latent_cycle_grid_animation(
                vae.decode, zs_mean, zs_logvar,
                mode=self.mode, num_frames=self._num_frames, decoder_device=vae.device, tensor_style_channels=False, return_stills=True,
                to_uint8=True, recon_min=self._recon_min, recon_max=self._recon_max,
            )
            image = make_image_grid(stills.reshape(-1, *stills.shape[2:]), num_cols=stills.shape[1], pad=4)

        # log video -- none, img, vid, both
        wandb_items = {}
        if self._wandb_mode in ('img', 'both'): wandb_items[f'{self.mode}_img'] = wandb.Image(image)
        if self._wandb_mode in ('vid', 'both'): wandb_items[f'{self.mode}_vid'] = wandb.Video(np.transpose(animation, [0, 3, 1, 2]), fps=self._fps, format='mp4'),
        wb_log_metrics(trainer.logger, wandb_items)

        # log locally
        if self.plt_show:
            fig, ax = plt.subplots(1, 1, figsize=(self.plt_block_size*stills.shape[1], self.plt_block_size*stills.shape[0]))
            ax.imshow(image)
            ax.axis('off')
            fig.tight_layout()
            plt.show()


def _normalized_numeric_metrics(items: dict):
    results = {}
    for k, v in items.items():
        if isinstance(v, (float, int)):
            results[k] = v
        else:
            try:
                results[k] = float(v)
            except:
                log.warning(f'SKIPPED: metric with key: {repr(k)}, result has invalid type: {type(v)} with value: {repr(v)}')
    return results


class VaeMetricLoggingCallback(BaseCallbackPeriodic):

    def __init__(
        self,
        step_end_metrics: Optional[Sequence[str]] = None,
        train_end_metrics: Optional[Sequence[str]] = None,
        every_n_steps: Optional[int] = None,
        begin_first_step: bool = False,
    ):
        super().__init__(every_n_steps, begin_first_step)
        self.step_end_metrics = step_end_metrics if step_end_metrics else []
        self.train_end_metrics = train_end_metrics if train_end_metrics else []
        assert isinstance(self.step_end_metrics, list)
        assert isinstance(self.train_end_metrics, list)
        assert self.step_end_metrics or self.train_end_metrics, 'No metrics given to step_end_metrics or train_end_metrics'

    def _compute_metrics_and_log(self, trainer: pl.Trainer, pl_module: pl.LightningModule, metrics: list, is_final=False):
        # get dataset and vae framework from trainer and module
        dataset, vae = _get_dataset_and_vae(trainer, pl_module, unwrap_groundtruth=True)
        # check if we need to skip
        # TODO: dataset needs to be able to handle wrapped datasets!
        if not dataset.is_ground_truth:
            warnings.warn(f'{dataset.__class__.__name__} is not an instance of {GroundTruthData.__name__}. Skipping callback: {self.__class__.__name__}!')
            return
        # compute all metrics
        for metric in metrics:
            pad = max(7+len(k) for k in disent.metrics.DEFAULT_METRICS)  # I know this is a magic variable... im just OCD
            if is_final:
                log.info(f'| {metric.__name__:<{pad}} - computing...')
            with Timer() as timer:
                scores = metric(dataset, lambda x: vae.encode(x.to(vae.device)))
            metric_results = ' '.join(f'{k}{c.GRY}={c.lMGT}{v:.3f}{c.RST}' for k, v in scores.items())
            log.info(f'| {metric.__name__:<{pad}} - time{c.GRY}={c.lYLW}{timer.pretty:<9}{c.RST} - {metric_results}')

            # log to trainer
            prefix = 'final_metric' if is_final else 'epoch_metric'
            prefixed_scores = {f'{prefix}/{k}': v for k, v in scores.items()}
            log_metrics(trainer.logger, _normalized_numeric_metrics(prefixed_scores))

            # log summary for WANDB
            # this is kinda hacky... the above should work for parallel coordinate plots
            wb_log_reduced_summaries(trainer.logger, prefixed_scores, reduction='max')

    def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.step_end_metrics:
            log.debug('Computing Epoch Metrics:')
            with Timer() as timer:
                self._compute_metrics_and_log(trainer, pl_module, metrics=self.step_end_metrics, is_final=False)
            log.debug(f'Computed Epoch Metrics! {timer.pretty}')

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.train_end_metrics:
            log.debug('Computing Final Metrics...')
            with Timer() as timer:
                self._compute_metrics_and_log(trainer, pl_module, metrics=self.train_end_metrics, is_final=True)
            log.debug(f'Computed Final Metrics! {timer.pretty}')


# class VaeLatentCorrelationLoggingCallback(BaseCallbackPeriodic):
#
#     def __init__(self, repeats_per_factor=10, every_n_steps=None, begin_first_step=False):
#         super().__init__(every_n_steps=every_n_steps, begin_first_step=begin_first_step)
#         self._repeats_per_factor = repeats_per_factor
#
#     def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
#         # get dataset and vae framework from trainer and module
#         dataset, vae = _get_dataset_and_vae(trainer, pl_module)
#         # check if we need to skip
#         if not dataset.is_ground_truth:
#             warnings.warn(f'{dataset.__class__.__name__} is not an instance of {GroundTruthData.__name__}. Skipping callback: {self.__class__.__name__}!')
#             return
#         # TODO: CONVERT THIS TO A METRIC!
#         # log the correspondence between factors and the latent space.
#         num_samples = np.sum(dataset.ground_truth_data.factor_sizes) * self._repeats_per_factor
#         factors = dataset.ground_truth_data.sample_factors(num_samples)
#         # encode observations of factors
#         zs = np.concatenate([
#             to_numpy(vae.encode(dataset.dataset_batch_from_factors(factor_batch, mode='input').to(vae.device)))
#             for factor_batch in iter_chunks(factors, 256)
#         ])
#         z_size = zs.shape[-1]
#
#         # calculate correlation matrix
#         f_and_z = np.concatenate([factors.T, zs.T])
#         f_and_z_corr = np.corrcoef(f_and_z)
#         # get correlation submatricies
#         f_corr = f_and_z_corr[:z_size, :z_size]   # upper left
#         z_corr = f_and_z_corr[z_size:, z_size:]   # bottom right
#         fz_corr = f_and_z_corr[z_size:, :z_size]  # upper right | y: z, x: f
#         # get maximum z correlations per factor
#         z_to_f_corr_maxs = np.max(np.abs(fz_corr), axis=0)
#         f_to_z_corr_maxs = np.max(np.abs(fz_corr), axis=1)
#         assert len(z_to_f_corr_maxs) == z_size
#         assert len(f_to_z_corr_maxs) == dataset.ground_truth_data.num_factors
#         # average correlation
#         ave_f_to_z_corr = f_to_z_corr_maxs.mean()
#         ave_z_to_f_corr = z_to_f_corr_maxs.mean()
#
#         # print
#         log.info(f'ave latent correlation: {ave_z_to_f_corr}')
#         log.info(f'ave factor correlation: {ave_f_to_z_corr}')
#         # log everything
#         log_metrics(trainer.logger, {
#             'metric.ave_latent_correlation': ave_z_to_f_corr,
#             'metric.ave_factor_correlation': ave_f_to_z_corr,
#         })
#         # make sure we only log the heatmap to WandB
#         wb_log_metrics(trainer.logger, {
#             'metric.correlation_heatmap': wandb.plots.HeatMap(
#                 x_labels=[f'z{i}' for i in range(z_size)],
#                 y_labels=list(dataset.ground_truth_data.factor_names),
#                 matrix_values=fz_corr, show_text=False
#             ),
#         })
#
#         NUM = 1
#         # generate traversal value graphs
#         for i in range(z_size):
#             correlation = np.abs(f_corr[i, :])
#             correlation[i] = 0
#             for j in np.argsort(correlation)[::-1][:NUM]:
#                 if i == j:
#                     continue
#                 ix, iy = (i, j)  # if i < j else (j, i)
#                 plt.scatter(zs[:, ix], zs[:, iy])
#                 plt.title(f'z{ix}-vs-z{iy}')
#                 plt.xlabel(f'z{ix}')
#                 plt.ylabel(f'z{iy}')
#
#                 # wandb.log({f"chart.correlation.z{ix}-vs-z{iy}": plt})
#                 # make sure we only log to WANDB
#                 wb_log_metrics(trainer.logger, {f"chart.correlation.z{ix}-vs-max-corr": plt})


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
