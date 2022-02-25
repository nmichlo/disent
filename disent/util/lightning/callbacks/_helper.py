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

import warnings
from typing import Union

import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

from disent.dataset import DisentDataset
from disent.frameworks.ae import Ae
from disent.frameworks.vae import Vae


# ========================================================================= #
# Helper Functions                                                          #
# ========================================================================= #


def _get_dataset_and_ae_like(trainer_or_dataset: pl.Trainer, pl_module: pl.LightningModule, unwrap_groundtruth: bool = False) -> (DisentDataset, Union[Ae, Vae]):
    assert isinstance(pl_module, (Ae, Vae)), f'{pl_module.__class__} is not an instance of {Ae} or {Vae}'
    # get dataset
    if isinstance(trainer_or_dataset, pl.Trainer):
        trainer = trainer_or_dataset
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
    else:
        dataset = trainer_or_dataset
    # check dataset
    assert isinstance(dataset, DisentDataset), f'retrieved dataset is not an {DisentDataset.__name__}'
    # unwrap dataset
    if unwrap_groundtruth:
        if dataset.is_wrapped_gt_data:
            old_dataset, dataset = dataset, dataset.unwrapped_shallow_copy()
            warnings.warn(f'Unwrapped ground truth dataset returned! {type(old_dataset.data).__name__} -> {type(dataset.data).__name__}')
    # done checks
    return dataset, pl_module


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


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
