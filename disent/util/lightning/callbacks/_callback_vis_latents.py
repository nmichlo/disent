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

import logging
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from matplotlib import pyplot as plt

from disent.frameworks.ae import Ae
from disent.frameworks.vae import Vae
from disent.util.lightning.callbacks._callback_vis_dists import log
from disent.util.lightning.callbacks._callbacks_base import BaseCallbackPeriodic
from disent.util.lightning.callbacks._helper import _get_dataset_and_ae_like
from disent.util.lightning.logger_util import wb_log_metrics
from disent.util.seeds import TempNumpySeed
from disent.util.visualize.vis_model import latent_cycle_grid_animation
from disent.util.visualize.vis_util import make_image_grid


log = logging.getLogger(__name__)


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


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


# ========================================================================= #
# Latent Visualisation Callback                                             #
# ========================================================================= #


class VaeLatentCycleLoggingCallback(BaseCallbackPeriodic):

    def __init__(
        self,
        seed: Optional[int] = 7777,
        every_n_steps: Optional[int] = None,
        begin_first_step: bool = False,
        num_frames: int = 17,
        mode: str = 'fitted_gaussian_cycle',
        log_wandb: bool = True,  # TODO: detect this automatically?
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
        self._seed = seed
        self._mode = mode
        self._plt_show = plt_show
        self._plt_block_size = plt_block_size
        self._log_wandb = log_wandb
        self._wandb_mode = wandb_mode
        self._num_frames = num_frames
        self._fps = wandb_fps
        # checks
        assert wandb_mode in {'img', 'vid', 'both'}, f'invalid wandb_mode={repr(wandb_mode)}, must be one of: ("img", "vid", "both")'
        # normalize
        self._recon_min, self._recon_max = _normalize_min_max_mean_std_to_min_max(
            recon_min=recon_min,
            recon_max=recon_max,
            recon_mean=recon_mean,
            recon_std=recon_std,
        )


    @torch.no_grad()
    def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # exit early
        if not (self._plt_show or self._log_wandb):
            log.warning(f'skipping {self.__class__.__name__} neither `plt_show` or `log_wandb` is `True`!')
            return

        # get dataset and vae framework from trainer and module
        dataset, vae = _get_dataset_and_ae_like(trainer, pl_module, unwrap_groundtruth=True)

        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # generate traversal
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # get random sample of z_means and z_logvars for computing the range of values for the latent_cycle
        with TempNumpySeed(self._seed):
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
            if self._recon_min is None: self._recon_min = float(torch.amin(batch).cpu())
            if self._recon_max is None: self._recon_max = float(torch.amax(batch).cpu())
            log.info(f'auto visualisation min: {self._recon_min} and max: {self._recon_max} obtained from {len(batch)} samples')

        # produce latent cycle grid animation
        # TODO: this needs to be fixed to not use logvar, but rather the representations or distributions themselves
        animation, stills = latent_cycle_grid_animation(
            vae.decode, zs_mean, zs_logvar,
            mode=self._mode, num_frames=self._num_frames, decoder_device=vae.device, tensor_style_channels=False, return_stills=True,
            to_uint8=True, recon_min=self._recon_min, recon_max=self._recon_max,
        )

        # TODO: should this not use `visualize_dataset_traversal`?
        image = make_image_grid(stills.reshape(-1, *stills.shape[2:]), num_cols=stills.shape[1], pad=4)

        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # log traversal
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # log video -- none, img, vid, both
        if self._log_wandb:
            wandb_items = {}
            if self._wandb_mode in ('img', 'both'): wandb_items[f'{self._mode}_img'] = wandb.Image(image)
            if self._wandb_mode in ('vid', 'both'): wandb_items[f'{self._mode}_vid'] = wandb.Video(np.transpose(animation, [0, 3, 1, 2]), fps=self._fps, format='mp4'),
            wb_log_metrics(trainer.logger, wandb_items)

        # log locally
        if self._plt_show:
            fig, ax = plt.subplots(1, 1, figsize=(self._plt_block_size*stills.shape[1], self._plt_block_size*stills.shape[0]))
            ax.imshow(image)
            ax.axis('off')
            fig.tight_layout()
            plt.show()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
