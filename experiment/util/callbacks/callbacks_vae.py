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

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.trainer.supporters import CombinedLoader

import disent.metrics
import disent.util.colors as c
from disent.dataset._augment_util import AugmentableDataset
from disent.dataset.groundtruth import GroundTruthDataset
from disent.frameworks.ae import AE
from disent.frameworks.vae import Vae
from disent.util import iter_chunks
from disent.util import TempNumpySeed
from disent.util import Timer
from disent.util import to_numpy
from disent.visualize.visualize_model import latent_cycle_grid_animation
from disent.visualize.visualize_util import make_image_grid
from experiment.util.callbacks.callbacks_base import _PeriodicCallback
from experiment.util.hydra_data import HydraDataModule
from experiment.util.logger_util import log_metrics
from experiment.util.logger_util import wb_log_metrics
from experiment.util.logger_util import wb_log_reduced_summaries


log = logging.getLogger(__name__)


# ========================================================================= #
# Helper Functions                                                          #
# ========================================================================= #


def _get_dataset_and_vae(trainer: pl.Trainer, pl_module: pl.LightningModule) -> (AugmentableDataset, AE):
    assert isinstance(pl_module, AE), f'{pl_module.__class__} is not an instance of {AE}'
    # get dataset
    if hasattr(trainer, 'datamodule') and (trainer.datamodule is not None):
        assert isinstance(trainer.datamodule, HydraDataModule)
        dataset = trainer.datamodule.dataset_train_noaug
    elif hasattr(trainer, 'train_dataloader') and (trainer.train_dataloader is not None):
        if isinstance(trainer.train_dataloader, CombinedLoader):
            dataset = trainer.train_dataloader.loaders.dataset
        else:
            raise RuntimeError(f'invalid trainer.train_dataloader: {trainer.train_dataloader}')
    else:
        raise RuntimeError('could not retrieve dataset! please report this...')
    # check dataset
    assert isinstance(dataset, AugmentableDataset), f'retrieved dataset is not an {AugmentableDataset.__name__}'
    # done checks
    return dataset, pl_module


def _should_skip_groundtruth_callback(callback, dataset):
    # TODO: this might be a hack and should be implemented some other way?
    if not isinstance(dataset, GroundTruthDataset):
        warnings.warn(f'{dataset.__class__.__name__} is not an instance of {GroundTruthDataset.__name__}. Skipping callback: {callback.__class__.__name__}!')
        return True
    return False


# ========================================================================= #
# Vae Framework Callbacks                                                   #
# ========================================================================= #


class VaeLatentCycleLoggingCallback(_PeriodicCallback):

    def __init__(self, seed=7777, every_n_steps=None, begin_first_step=False, mode='fitted_gaussian_cycle', plt_show=False):
        super().__init__(every_n_steps, begin_first_step)
        self.seed = seed
        self.mode = mode
        self.plt_show = plt_show

    def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # get dataset and vae framework from trainer and module
        dataset, vae = _get_dataset_and_vae(trainer, pl_module)

        # get random sample of z_means and z_logvars for computing the range of values for the latent_cycle
        with TempNumpySeed(self.seed):
            obs = dataset.dataset_sample_batch(64, mode='input').to(vae.device)

        if isinstance(vae, Vae):
            z_means, z_logvars = vae.encode_params(obs)
        else:
            z_means = vae.encode_params(obs)
            z_logvars = torch.ones_like(z_means)

        # produce latent cycle grid animation
        frames, stills = latent_cycle_grid_animation(vae.decode, z_means, z_logvars, mode=self.mode, num_frames=21, decoder_device=vae.device, tensor_style_channels=False, return_stills=True, to_uint8=True)

        # log video
        wb_log_metrics(trainer.logger, {
            self.mode: wandb.Video(np.transpose(frames, [0, 3, 1, 2]), fps=5, format='mp4'),
        })

        if self.plt_show:
            grid = make_image_grid(np.reshape(stills, (-1, *stills.shape[2:])), num_cols=stills.shape[1])
            plt.imshow(grid)
            plt.tight_layout()
            plt.axis('off')
            plt.show()

class VaeDisentanglementLoggingCallback(_PeriodicCallback):

    def __init__(self, step_end_metrics=None, train_end_metrics=None, every_n_steps=None, begin_first_step=False):
        super().__init__(every_n_steps, begin_first_step)
        self.step_end_metrics = step_end_metrics if step_end_metrics else []
        self.train_end_metrics = train_end_metrics if train_end_metrics else []
        assert isinstance(self.step_end_metrics, list)
        assert isinstance(self.train_end_metrics, list)
        assert self.step_end_metrics or self.train_end_metrics, 'No metrics given to step_end_metrics or train_end_metrics'

    def _compute_metrics_and_log(self, trainer: pl.Trainer, pl_module: pl.LightningModule, metrics: list, is_final=False):
        # get dataset and vae framework from trainer and module
        dataset, vae = _get_dataset_and_vae(trainer, pl_module)
        # check if we need to skip
        if _should_skip_groundtruth_callback(self, dataset):
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


class VaeLatentCorrelationLoggingCallback(_PeriodicCallback):
    
    def __init__(self, repeats_per_factor=10, every_n_steps=None, begin_first_step=False):
        super().__init__(every_n_steps=every_n_steps, begin_first_step=begin_first_step)
        self._repeats_per_factor = repeats_per_factor

    def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # get dataset and vae framework from trainer and module
        dataset, vae = _get_dataset_and_vae(trainer, pl_module)
        # check if we need to skip
        if _should_skip_groundtruth_callback(self, dataset):
            return
        # TODO: CONVERT THIS TO A METRIC!
        # log the correspondence between factors and the latent space.
        num_samples = np.sum(dataset.factor_sizes) * self._repeats_per_factor
        factors = dataset.sample_factors(num_samples)
        # encode observations of factors
        zs = np.concatenate([
            to_numpy(vae.encode(dataset.dataset_batch_from_factors(factor_batch, mode='input').to(vae.device)))
            for factor_batch in iter_chunks(factors, 256)
        ])
        z_size = zs.shape[-1]

        # calculate correlation matrix
        f_and_z = np.concatenate([factors.T, zs.T])
        f_and_z_corr = np.corrcoef(f_and_z)
        # get correlation submatricies
        f_corr = f_and_z_corr[:z_size, :z_size]   # upper left
        z_corr = f_and_z_corr[z_size:, z_size:]   # bottom right
        fz_corr = f_and_z_corr[z_size:, :z_size]  # upper right | y: z, x: f
        # get maximum z correlations per factor
        z_to_f_corr_maxs = np.max(np.abs(fz_corr), axis=0)
        f_to_z_corr_maxs = np.max(np.abs(fz_corr), axis=1)
        assert len(z_to_f_corr_maxs) == z_size
        assert len(f_to_z_corr_maxs) == dataset.num_factors
        # average correlation
        ave_f_to_z_corr = f_to_z_corr_maxs.mean()
        ave_z_to_f_corr = z_to_f_corr_maxs.mean()

        # print
        log.info(f'ave latent correlation: {ave_z_to_f_corr}')
        log.info(f'ave factor correlation: {ave_f_to_z_corr}')
        # log everything
        log_metrics(trainer.logger, {
            'metric.ave_latent_correlation': ave_z_to_f_corr,
            'metric.ave_factor_correlation': ave_f_to_z_corr,
        })
        # make sure we only log the heatmap to WandB
        wb_log_metrics(trainer.logger, {
            'metric.correlation_heatmap': wandb.plots.HeatMap(
                x_labels=[f'z{i}' for i in range(z_size)],
                y_labels=list(dataset.factor_names),
                matrix_values=fz_corr, show_text=False
            ),
        })

        NUM = 1
        # generate traversal value graphs
        for i in range(z_size):
            correlation = np.abs(f_corr[i, :])
            correlation[i] = 0
            for j in np.argsort(correlation)[::-1][:NUM]:
                if i == j:
                    continue
                ix, iy = (i, j)  # if i < j else (j, i)
                plt.scatter(zs[:, ix], zs[:, iy])
                plt.title(f'z{ix}-vs-z{iy}')
                plt.xlabel(f'z{ix}')
                plt.ylabel(f'z{iy}')
                # wandb.log({f"chart.correlation.z{ix}-vs-z{iy}": plt})
                wandb.log({f"chart.correlation.z{ix}-vs-max-corr": plt})

# ========================================================================= #
# END                                                                       #
# ========================================================================= #

#
# # get random sample of z_means and z_logvars for computing the range of values for the latent_cycle
# from torchvision.transforms import ToTensor
#
# from disent.dataset import GroundTruthDataset
# from disent.data.groundtruth._xygrid import XYGridData
# from disent.model.ae import DecoderConv64, EncoderConv64, AutoEncoder
# from disent.util import TempNumpySeed, chunked, to_numpy
# import numpy as np
#
# dataset = GroundTruthDataset(XYGridData(), transform=ToTensor())
#
# # =========================== #
# # Options
# # =========================== #
# repeats_per_factor = 10
# num_generated_samples = np.sum(dataset.factor_sizes) * repeats_per_factor
# # =========================== #
# # Frist Mode - one factor is repeated, but then overwrite single factor to have entire range
# # =========================== #
# with TempNumpySeed(777):
#     factors = dataset.sample_factors(repeats_per_factor)
# factor_traversals = []
# for i, factor_size in enumerate(dataset.factor_sizes):
#     traversal = factors[:, None, :].repeat(factor_size, axis=1)
#     traversal[:, :, i] = np.arange(factor_size)
#     traversal = traversal.reshape(-1, dataset.num_factors)
#     factor_traversals.append(traversal)
# assert num_generated_samples == sum(len(traversal) for traversal in factor_traversals)
# # =========================== #
# # Second Mode - resample each time, but then overwrite single factor to have entire range
# # =========================== #
# factor_traversals = []
# with TempNumpySeed(777):
#     for i, factor_size in enumerate(dataset.factor_sizes):
#         traversal = dataset.sample_factors(repeats_per_factor * factor_size)
#         traversal = traversal.reshape(repeats_per_factor, factor_size, dataset.num_factors)
#         traversal[:, :, i] = np.arange(factor_size)
#         traversal = traversal.reshape(-1, dataset.num_factors)
#         factor_traversals.append(traversal)
# assert num_generated_samples == sum(len(traversal) for traversal in factor_traversals)
# # =========================== #
# # Third mode, randomly sample everything!
# # =========================== #
# factors = dataset.sample_factors(num_generated_samples)
# assert num_generated_samples == len(factors)
#
#
# # =========================== #
# # PROCESS!
# # =========================== #
#
# # z_size = 6
# # model = AutoEncoder(
# #     encoder=EncoderConv64(x_shape=(3, 64, 64), z_size=z_size, z_multiplier=2),
# #     decoder=DecoderConv64(x_shape=(3, 64, 64), z_size=z_size),
# # )
# #
# # zs = []
# # for factor_batch in chunked(factors, 256):
# #     observations = dataset.sample_observations_from_factors(factor_batch)
# #     z_mean, z_logvar = model.encode_gaussian(observations)
# #     zs.append(to_numpy(z_mean))
# # zs = np.concatenate(zs)
# #
# # f_and_z = np.concatenate([factors.T, zs.T])
# #
# # corr = np.corrcoef(f_and_z)
