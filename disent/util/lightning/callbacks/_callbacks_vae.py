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
from typing import Literal
from typing import Optional
from typing import Sequence
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
from disent.util.iters import chunked
from disent.util.lightning.callbacks._callbacks_base import BaseCallbackPeriodic
from disent.util.lightning.logger_util import log_metrics
from disent.util.lightning.logger_util import wb_log_metrics
from disent.util.lightning.logger_util import wb_log_reduced_summaries
from disent.util.profiling import Timer
from disent.util.seeds import TempNumpySeed
from disent.util.visualize.vis_model import latent_cycle_grid_animation
from disent.util.visualize.vis_util import make_image_grid

# TODO: wandb and matplotlib are not in requirements
import matplotlib.pyplot as plt
import wandb

from research.util import plt_hide_axis
from research.util import plt_subplots_imshow


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


# _AE_DIST_NAMES = ('x', 'z_l1', 'z_l2', 'x_recon', 'z_d1', 'z_d2')
# _VAE_DIST_NAMES = ('x', 'z_l1', 'z_l2', 'kl', 'x_recon', 'z_d1', 'z_d2', 'kl_center_d1', 'kl_center_d2')

_AE_DIST_NAMES = ('x', 'z_l1', 'x_recon')
_VAE_DIST_NAMES = ('x', 'z_l1', 'kl', 'x_recon')


@torch.no_grad()
def _get_dists_ae(ae: Ae, x_a: torch.Tensor, x_b: torch.Tensor, recon_loss: ReconLossHandler):
    # feed forware
    z_a, z_b = ae.encode(x_a), ae.encode(x_b)
    r_a, r_b = ae.decode(z_a), ae.decode(z_b)
    # distances
    return _AE_DIST_NAMES, [
        recon_loss.compute_pairwise_loss(x_a, x_b),
        torch.norm(z_a - z_b, p=1, dim=-1),  # l1 dist
        recon_loss.compute_pairwise_loss(r_a, r_b),
    ]


@torch.no_grad()
def _get_dists_vae(vae: Vae, x_a: torch.Tensor, x_b: torch.Tensor, recon_loss: ReconLossHandler):
    from torch.distributions import kl_divergence
    # feed forward
    (z_post_a, z_prior_a), (z_post_b, z_prior_b) = vae.encode_dists(x_a), vae.encode_dists(x_b)
    z_a, z_b = z_post_a.mean, z_post_b.mean
    r_a, r_b = vae.decode(z_a), vae.decode(z_b)
    # dists
    kl_ab = 0.5 * kl_divergence(z_post_a, z_post_b) + 0.5 * kl_divergence(z_post_b, z_post_a)
    # distances
    return _VAE_DIST_NAMES, [
        recon_loss.compute_pairwise_loss(x_a, x_b),
        torch.norm(z_a - z_b, p=1, dim=-1),  # l1 dist
        recon_loss._pairwise_reduce(kl_ab),
        recon_loss.compute_pairwise_loss(r_a, r_b),
    ]


@torch.no_grad()
def _collect_dists_subbatches(model, dists_fn, obs: torch.Tensor, i_a: np.ndarray, i_b: np.ndarray, recon_loss: ReconLossHandler, batch_size: int = 64):
    # feed forward
    results = []
    for idxs in chunked(np.stack([i_a, i_b], axis=-1), chunk_size=batch_size):
        ia, ib = idxs.T
        x_a, x_b = obs[ia], obs[ib]
        # feed forward
        name, data = dists_fn(model, x_a=x_a, x_b=x_b, recon_loss=recon_loss)
        results.append(data)
    return name, [torch.cat(r, dim=0) for r in zip(*results)]


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
        self._recon_loss = make_reconstruction_loss('mse', 'mean')
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
        if isinstance(vae, Vae): dists_fn = _get_dists_vae
        elif isinstance(vae, Ae): dists_fn = _get_dists_ae
        else:
            log.warning(f'cannot run {self.__class__.__name__}, unsupported model type: {type(vae)}, must be {Ae.__name__} or {Vae.__name__}')
            return
        # get gt data
        gt_data = dataset.gt_data

        # log this callback
        log.info(f'| {gt_data.name} - computing factor distances...')

        # this can be moved into a helper method!
        with Timer() as timer, TempNumpySeed(self._seed):
            f_data = []
            for f_idx, f_size in enumerate(gt_data.factor_sizes):
                # save for the current factor
                f_dists = []
                # upper triangle excluding diagonal
                i_a, i_b = np.triu_indices(f_size, k=1)
                # repeat over random traversals
                for i in range(self._traversal_repeats):
                    # get random factor traversal
                    factors = gt_data.sample_random_factor_traversal(f_idx=f_idx)
                    indices = gt_data.pos_to_idx(factors)
                    # load data
                    obs = dataset.dataset_batch_from_indices(indices, 'input')
                    obs = obs.to(vae.device)
                    # feed forward
                    names, dists = _collect_dists_subbatches(vae, dists_fn=dists_fn, obs=obs, i_a=i_a, i_b=i_b, recon_loss=self._recon_loss, batch_size=self._batch_size)
                    # distances
                    f_dists.append(dists)
                # aggregate all dists into distances matrices for current factor
                f_dmats = [
                    _to_dmat(size=f_size, i_a=i_a, i_b=i_b, dists=torch.stack(dists, dim=0).mean(dim=0))
                    for dists in zip(*f_dists)
                ]
                # handle factors or not
                if self._include_gt_factor_dists:
                    i_dmat = _to_dmat(size=f_size, i_a=i_a, i_b=i_b, dists=np.abs(factors[i_a] - factors[i_b]).sum(axis=-1))
                    names = ('factors', *names)
                    f_dmats = [i_dmat, *f_dmats]
                # append data
                f_data.append(f_dmats)

        # log this callback!
        log.info(f'| {gt_data.name} - computed factor distances! time{c.GRY}={c.lYLW}{timer.pretty:<9}{c.RST}')

        # plot!
        title         = f'{vae.__class__.__name__}: {gt_data.name.capitalize()} Distances'
        imshow_kwargs = dict(cmap='Blues')
        figsize       = (self._plt_block_size*len(f_data[0]), self._plt_block_size*gt_data.num_factors)

        if not self._transpose_plot:
            fig, axs = plt_subplots_imshow(
                grid=f_data,
                col_labels=names,
                row_labels=gt_data.factor_names,
                figsize=figsize,
                title=title,
                imshow_kwargs=imshow_kwargs,
            )
        else:
            fig, axs = plt_subplots_imshow(
                grid=list(zip(*f_data)),
                col_labels=gt_data.factor_names,
                row_labels=names,
                figsize=figsize[::-1],
                title=title,
                imshow_kwargs=imshow_kwargs,
            )

        if self._plt_show:
            plt.show()

        if self._log_wandb:
            wb_log_metrics(trainer.logger, {
                'factor_distances': wandb.Image(fig)
            })


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
        recon_min: Union[int, Literal['auto']] = 0.,
        recon_max: Union[int, Literal['auto']] = 1.,
    ):
        super().__init__(every_n_steps, begin_first_step)
        self.seed = seed
        self.mode = mode
        self.plt_show = plt_show
        self.plt_block_size = plt_block_size
        self._wandb_mode = wandb_mode
        self._recon_min = recon_min
        self._recon_max = recon_max
        self._num_frames = num_frames
        self._fps = wandb_fps
        # checks
        assert wandb_mode in {'none', 'img', 'vid', 'both'}, f'invalid wandb_mode={repr(wandb_mode)}, must be one of: ("none", "img", "vid", "both")'

    def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # get dataset and vae framework from trainer and module
        dataset, vae = _get_dataset_and_vae(trainer, pl_module, unwrap_groundtruth=True)

        # TODO: should this not use `visualize_dataset_traversal`?

        with torch.no_grad():
            # get random sample of z_means and z_logvars for computing the range of values for the latent_cycle
            with TempNumpySeed(self.seed):
                obs = dataset.dataset_sample_batch(64, mode='input').to(vae.device)

            # get representations
            if isinstance(vae, Vae):
                # variational auto-encoder
                ds_posterior, ds_prior = vae.encode_dists(obs)
                zs_mean, zs_logvar = ds_posterior.mean, torch.log(ds_posterior.variance)
            elif isinstance(vae, Ae):
                # auto-encoder
                zs_mean = vae.encode(obs)
                zs_logvar = torch.ones_like(zs_mean)
            else:
                log.warning(f'cannot run {self.__class__.__name__}, unsupported type: {type(vae)}, must be {Ae.__name__} or {Vae.__name__}')
                return

            # get min and max if auto
            if (self._recon_min == 'auto') or (self._recon_max == 'auto'):
                if self._recon_min == 'auto': self._recon_min = float(torch.min(obs).cpu())
                if self._recon_max == 'auto': self._recon_max = float(torch.max(obs).cpu())
                log.info(f'auto visualisation min: {self._recon_min} and max: {self._recon_max} obtained from {len(obs)} samples')

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
            log_metrics(trainer.logger, prefixed_scores)

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
