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
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch

from pytorch_lightning.trainer.supporters import CombinedLoader

import disent.metrics
import disent.util.strings.colors as c
from disent.dataset import DisentDataset
from disent.dataset.data import GroundTruthData
from disent.frameworks.ae import Ae
from disent.frameworks.vae import Vae
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


class VaeLatentCycleLoggingCallback(BaseCallbackPeriodic):

    def __init__(self, seed=7777, every_n_steps=None, begin_first_step=False, mode='fitted_gaussian_cycle', plt_show=False, plt_block_size=1.0, recon_min: Union[int, Literal['auto']] = 0., recon_max: Union[int, Literal['auto']] = 1.):
        super().__init__(every_n_steps, begin_first_step)
        self.seed = seed
        self.mode = mode
        self.plt_show = plt_show
        self.plt_block_size = plt_block_size
        self._recon_min = recon_min
        self._recon_max = recon_max

    def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # get dataset and vae framework from trainer and module
        dataset, vae = _get_dataset_and_vae(trainer, pl_module, unwrap_groundtruth=True)

        with torch.no_grad():
            # get random sample of z_means and z_logvars for computing the range of values for the latent_cycle
            with TempNumpySeed(self.seed):
                obs = dataset.dataset_sample_batch(64, mode='input').to(vae.device)

            # get representations
            if isinstance(vae, Vae):
                # variational auto-encoder
                ds_posterior, ds_prior = vae.encode_dists(obs)
                zs_mean, zs_logvar = ds_posterior.mean, torch.log(ds_posterior.variance)
            else:
                # auto-encoder
                zs_mean = vae.encode(obs)
                zs_logvar = torch.ones_like(zs_mean)

            # get min and max if auto
            if (self._recon_min == 'auto') or (self._recon_max == 'auto'):
                if self._recon_min == 'auto': self._recon_min = float(torch.min(obs).cpu())
                if self._recon_max == 'auto': self._recon_max = float(torch.max(obs).cpu())
                log.info(f'auto visualisation min: {self._recon_min} and max: {self._recon_max} obtained from {len(obs)} samples')

            # produce latent cycle grid animation
            # TODO: this needs to be fixed to not use logvar, but rather the representations or distributions themselves
            frames, stills = latent_cycle_grid_animation(
                vae.decode, zs_mean, zs_logvar,
                mode=self.mode, num_frames=21, decoder_device=vae.device, tensor_style_channels=False, return_stills=True,
                to_uint8=True, recon_min=self._recon_min, recon_max=self._recon_max,
            )

        # log video
        wb_log_metrics(trainer.logger, {
            self.mode: wandb.Video(np.transpose(frames, [0, 3, 1, 2]), fps=4, format='mp4'),
        })

        if self.plt_show:
            grid = make_image_grid(np.reshape(stills, (-1, *stills.shape[2:])), num_cols=stills.shape[1], pad=4)
            fig, ax = plt.subplots(1, 1, figsize=(self.plt_block_size*stills.shape[1], self.plt_block_size*stills.shape[0]))
            ax.imshow(grid)
            ax.axis('off')
            fig.tight_layout()
            plt.show()


class VaeDisentanglementLoggingCallback(BaseCallbackPeriodic):

    def __init__(self, step_end_metrics=None, train_end_metrics=None, every_n_steps=None, begin_first_step=False):
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
