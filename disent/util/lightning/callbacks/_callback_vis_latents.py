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

from disent.dataset import DisentDataset
from disent.frameworks.ae import Ae
from disent.frameworks.vae import Vae
from disent.util.lightning.callbacks._callbacks_base import BaseCallbackPeriodic
from disent.util.lightning.callbacks._helper import _get_dataset_and_ae_like
from disent.util.lightning.logger_util import wb_log_metrics
from disent.util.seeds import TempNumpySeed
from disent.util.visualize.vis_img import torch_to_images
from disent.util.visualize.vis_latents import make_decoded_latent_cycles
from disent.util.visualize.vis_util import make_animated_image_grid
from disent.util.visualize.vis_util import make_image_grid


log = logging.getLogger(__name__)


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


MinMaxHint = Optional[Union[int, Literal['auto']]]
MeanStdHint = Optional[Union[Tuple[float, ...], float]]


def get_vis_min_max(
    recon_min: MinMaxHint = None,
    recon_max: MinMaxHint = None,
    recon_mean: MeanStdHint = None,
    recon_std:  MeanStdHint = None,
) -> Union[Tuple[None, None], Tuple[np.ndarray, np.ndarray]]:
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
            return None, None  # "auto" -> None
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
    if recon_max is None: recon_max = 1.0
    # change type
    recon_min = np.array(recon_min)
    recon_max = np.array(recon_max)
    assert recon_min.ndim in (0, 1)
    assert recon_max.ndim in (0, 1)
    # checks
    assert np.all(recon_min < recon_max), f'recon_min={recon_min} must be less than recon_max={recon_max}'
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
        mode: str = 'minmax_interval_cycle',
        num_stats_samples: int = 64,
        log_wandb: bool = True,  # TODO: detect this automatically?
        wandb_mode: str = 'both',
        wandb_fps: int = 4,
        plt_show: bool = False,
        plt_block_size: float = 1.0,
        # recon_min & recon_max
        recon_min: MinMaxHint = None,    # scale data in this range [min, max] to [0, 1]
        recon_max: MinMaxHint = None,    # scale data in this range [min, max] to [0, 1]
        recon_mean: MeanStdHint = None,  # automatically converted to min & max [(0-mean)/std, (1-mean)/std], assuming original range of values is [0, 1]
        recon_std: MeanStdHint = None,   # automatically converted to min & max [(0-mean)/std, (1-mean)/std], assuming original range of values is [0, 1]
    ):
        super().__init__(every_n_steps, begin_first_step)
        self._seed = seed
        self._mode = mode
        self._num_stats_samples = num_stats_samples
        self._plt_show = plt_show
        self._plt_block_size = plt_block_size
        self._log_wandb = log_wandb
        self._wandb_mode = wandb_mode
        self._num_frames = num_frames
        self._fps = wandb_fps
        # checks
        assert wandb_mode in {'img', 'vid', 'both'}, f'invalid wandb_mode={repr(wandb_mode)}, must be one of: ("img", "vid", "both")'
        # normalize
        self._recon_min, self._recon_max = get_vis_min_max(
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

        # feed forward and visualise everything!
        stills, animation, image = self.get_visualisations(trainer, pl_module)

        # log video -- none, img, vid, both
        # TODO: there might be a memory leak in making the video below? Or there could be one in the actual DSPRITES dataset? memory usage seems to be very high and increase on this dataset.
        if self._log_wandb:
            import wandb
            wandb_items = {}
            if self._wandb_mode in ('img', 'both'): wandb_items[f'{self._mode}_img'] = wandb.Image(image)
            if self._wandb_mode in ('vid', 'both'): wandb_items[f'{self._mode}_vid'] = wandb.Video(np.transpose(animation, [0, 3, 1, 2]), fps=self._fps, format='mp4'),
            wb_log_metrics(trainer.logger, wandb_items)

        # log locally
        if self._plt_show:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(self._plt_block_size*stills.shape[1], self._plt_block_size*stills.shape[0]))
            ax.imshow(image)
            ax.axis('off')
            fig.tight_layout()
            plt.show()

    def get_visualisations(
        self,
        trainer_or_dataset: Union[pl.Trainer, DisentDataset],
        pl_module: pl.LightningModule,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]]:
        return self.generate_visualisations(
            trainer_or_dataset,
            pl_module,
            seed=self._seed,
            num_frames=self._num_frames,
            mode=self._mode,
            num_stats_samples=self._num_stats_samples,
            recon_min=self._recon_min,
            recon_max=self._recon_max,
            recon_mean=None,
            recon_std=None,
        )

    @classmethod
    def generate_visualisations(
        cls,
        trainer_or_dataset: Union[pl.Trainer, DisentDataset],
        pl_module: pl.LightningModule,
        seed: Optional[int] = 7777,
        num_frames: int = 17,
        mode: str = 'fitted_gaussian_cycle',
        num_stats_samples: int = 64,
        # recon_min & recon_max
        recon_min: MinMaxHint = None,
        recon_max: MinMaxHint = None,
        recon_mean: MeanStdHint = None,
        recon_std: MeanStdHint = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]]:
        # normalize
        recon_min, recon_max = get_vis_min_max(
            recon_min=recon_min,
            recon_max=recon_max,
            recon_mean=recon_mean,
            recon_std=recon_std,
        )

        # get dataset and vae framework from trainer and module
        dataset, vae = _get_dataset_and_ae_like(trainer_or_dataset, pl_module, unwrap_groundtruth=True)

        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # generate traversal
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # get random sample of z_means and z_logvars for computing the range of values for the latent_cycle
        with TempNumpySeed(seed):
            batch, indices = dataset.dataset_sample_batch(num_stats_samples, mode='input', replace=True, return_indices=True)  # replace just in case the dataset it tiny
            batch = batch.to(vae.device)

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
            log.warning(f'cannot run {cls.__name__}, unsupported type: {type(vae)}, must be {Ae.__name__} or {Vae.__name__}')
            return

        # get min and max if auto
        if (recon_min is None) or (recon_max is None):
            if recon_min is None: recon_min = float(torch.amin(batch).cpu())
            if recon_max is None: recon_max = float(torch.amax(batch).cpu())
            log.info(f'auto visualisation min: {recon_min} and max: {recon_max} obtained from {len(batch)} samples')

        # produce latent cycle still images & convert them to images
        stills = make_decoded_latent_cycles(vae.decode, zs_mean, zs_logvar, mode=mode, num_animations=1, num_frames=num_frames, decoder_device=vae.device)[0]
        stills = torch_to_images(stills, in_dims='CHW', out_dims='HWC', in_min=recon_min, in_max=recon_max, always_rgb=True, to_numpy=True)

        # generate the video frames and image grid from the stills
        # - TODO: this needs to be fixed to not use logvar, but rather the representations or distributions themselves
        # - TODO: should this not use `visualize_dataset_traversal`?
        frames = make_animated_image_grid(stills, pad=4, border=True, bg_color=None)
        image = make_image_grid(stills.reshape(-1, *stills.shape[2:]), num_cols=stills.shape[1], pad=4, border=True, bg_color=None)

        # done
        return stills, frames, image



# TODO: FIX THIS ERROR:
#    [2022-03-12 11:44:59,661][experiment.util.run_utils][ERROR] - exiting: experiment error | [Errno 12] Cannot allocate memory
#    Traceback (most recent call last):
#      File "/home-mscluster/nmichlo/workspace/research/disent/experiment/run.py", line 462, in hydra_main
#        run_action(cfg)
#      File "/home-mscluster/nmichlo/workspace/research/disent/experiment/run.py", line 378, in run_action
#        action(cfg)
#      File "/home-mscluster/nmichlo/workspace/research/disent/experiment/run.py", line 351, in action_train
#        trainer.fit(framework, datamodule=datamodule)
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 735, in fit
#        self._call_and_handle_interrupt(
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 682, in _call_and_handle_interrupt
#        return trainer_fn(*args, **kwargs)
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 770, in _fit_impl
#        self._run(model, ckpt_path=ckpt_path)
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1193, in _run
#        self._dispatch()
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1272, in _dispatch
#        self.training_type_plugin.start_training(self)
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py", line 202, in start_training
#        self._results = trainer.run_stage()
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1282, in run_stage
#        return self._run_train()
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1312, in _run_train
#        self.fit_loop.run()
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/loops/base.py", line 145, in run
#        self.advance(*args, **kwargs)
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py", line 234, in advance
#        self.epoch_loop.run(data_fetcher)
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/loops/base.py", line 145, in run
#        self.advance(*args, **kwargs)
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/loops/epoch/training_epoch_loop.py", line 221, in advance
#        self.trainer.call_hook("on_batch_end")
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1477, in call_hook
#        callback_fx(*args, **kwargs)
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/pytorch_lightning/trainer/callback_hook.py", line 163, in on_batch_end
#        callback.on_batch_end(self, self.lightning_module)
#      File "/home-mscluster/nmichlo/workspace/research/disent/disent/util/lightning/callbacks/_callbacks_base.py", line 57, in on_batch_end
#        self.do_step(trainer, pl_module)
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/torch/autograd/grad_mode.py", line 28, in decorate_context
#        return func(*args, **kwargs)
#      File "/home-mscluster/nmichlo/workspace/research/disent/disent/util/lightning/callbacks/_callback_vis_latents.py", line 165, in do_step
#        if self._wandb_mode in ('vid', 'both'): wandb_items[f'{self._mode}_vid'] = wandb.Video(np.transpose(animation, [0, 3, 1, 2]), fps=self._fps, format='mp4'),
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/wandb/sdk/data_types.py", line 1232, in __init__
#        self.encode()
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/wandb/sdk/data_types.py", line 1255, in encode
#        clip.write_videofile(filename, **kwargs)
#      File "<decorator-gen-55>", line 2, in write_videofile
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/moviepy/decorators.py", line 54, in requires_duration
#        return f(clip, *a, **k)
#      File "<decorator-gen-54>", line 2, in write_videofile
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/moviepy/decorators.py", line 135, in use_clip_fps_by_default
#        return f(clip, *new_a, **new_kw)
#      File "<decorator-gen-53>", line 2, in write_videofile
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/moviepy/decorators.py", line 22, in convert_masks_to_RGB
#        return f(clip, *a, **k)
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/moviepy/video/VideoClip.py", line 300, in write_videofile
#        ffmpeg_write_video(self, filename, fps, codec,
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/moviepy/video/io/ffmpeg_writer.py", line 213, in ffmpeg_write_video
#        with FFMPEG_VideoWriter(filename, clip.size, fps, codec = codec,
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/site-packages/moviepy/video/io/ffmpeg_writer.py", line 129, in __init__
#        self.proc = sp.Popen(cmd, **popen_params)
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/subprocess.py", line 951, in __init__
#        self._execute_child(args, executable, preexec_fn, close_fds,
#      File "/home-mscluster/nmichlo/installed/pyenv/versions/miniconda3-latest/envs/disent-conda/lib/python3.9/subprocess.py", line 1754, in _execute_child
#        self.pid = _posixsubprocess.fork_exec(
#    OSError: [Errno 12] Cannot allocate memory


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
