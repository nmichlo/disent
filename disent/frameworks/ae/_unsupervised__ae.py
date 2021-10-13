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
from dataclasses import dataclass
from numbers import Number
from typing import Any
from typing import Dict
from typing import final
from typing import Sequence
from typing import Tuple
from typing import Union

import torch

from disent.frameworks import DisentFramework
from disent.frameworks.helper.reconstructions import make_reconstruction_loss
from disent.frameworks.helper.reconstructions import ReconLossHandler
from disent.frameworks.helper.util import detach_all
from disent.model import AutoEncoder
from disent.util.iters import map_all


log = logging.getLogger(__name__)


# ========================================================================= #
# framework_vae                                                             #
# ========================================================================= #


class Ae(DisentFramework):
    """
    Basic Auto Encoder
    ------------------

    See the docs for the VAE, while the AE is more simple, the VAE docs
    cover the concepts needed to get started with writing Auto-Encoder
    sub-classes.

    Like the VAE, the AE is also written such that you can change the
    number of required input observations that should be fed through the
    network in parallel with `REQUIRED_OBS`. Various hooks are also made
    available to add functionality and access the internal data.

    - HOOKS:
        * `hook_ae_intercept_zs`
        * `hook_ae_compute_ave_aug_loss` (NB: not the same as `hook_compute_ave_aug_loss` from VAEs)

    - OVERRIDES:
        * `compute_ave_recon_loss`
    """

    REQUIRED_Z_MULTIPLIER = 1
    REQUIRED_OBS = 1

    @dataclass
    class cfg(DisentFramework.cfg):
        recon_loss: str = 'mse'
        # multiple reduction modes exist for the various loss components.
        # - 'sum': sum over the entire batch
        # - 'mean': mean over the entire batch
        # - 'mean_sum': sum each observation, returning the mean sum over the batch
        loss_reduction: str = 'mean'
        # disable various components
        disable_decoder: bool = False
        disable_rec_loss: bool = False
        disable_aug_loss: bool = False

    def __init__(self, model: AutoEncoder, cfg: cfg = None, batch_augment=None):
        super().__init__(cfg=cfg, batch_augment=batch_augment)
        # vae model
        self._model = model
        # check the model
        assert isinstance(self._model, AutoEncoder)
        assert self._model.z_multiplier == self.REQUIRED_Z_MULTIPLIER, f'model z_multiplier is {repr(self._model.z_multiplier)} but {self.__class__.__name__} requires that it is: {repr(self.REQUIRED_Z_MULTIPLIER)}'
        # recon loss & activation fn
        self.__recon_handler: ReconLossHandler = make_reconstruction_loss(self.cfg.recon_loss, reduction=self.cfg.loss_reduction)

    @final
    @property
    def recon_handler(self) -> ReconLossHandler:
        return self.__recon_handler

    # --------------------------------------------------------------------- #
    # AE Training Step -- Overridable                                       #
    # --------------------------------------------------------------------- #

    @final
    def _get_xs_and_targs(self, batch: Dict[str, Tuple[torch.Tensor, ...]], batch_idx) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        xs_targ = batch['x_targ']
        if 'x' not in batch:
            # TODO: re-enable this warning but only ever print once!
            # warnings.warn('dataset does not have input: x -> x_targ using target as input: x_targ -> x_targ')
            xs = xs_targ
        else:
            xs = batch['x']
        # check that we have the correct number of inputs
        if (len(xs) != self.REQUIRED_OBS) or (len(xs_targ) != self.REQUIRED_OBS):
            log.warning(f'batch len(xs)={len(xs)} and len(xs_targ)={len(xs_targ)} observation count mismatch, requires: {self.REQUIRED_OBS}')
        # done
        return xs, xs_targ

    @final
    def do_training_step(self, batch, batch_idx):
        xs, xs_targ = self._get_xs_and_targs(batch, batch_idx)

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent variables
        zs = map_all(self.encode, xs)
        # intercept latent variables
        zs, logs_intercept_zs = self.hook_ae_intercept_zs(zs)
        # reconstruct without the final activation
        xs_partial_recon = map_all(self.decode_partial, detach_all(zs, if_=self.cfg.disable_decoder))
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # compute all the recon losses
        recon_loss, logs_recon = self.compute_ave_recon_loss(xs_partial_recon, xs_targ)
        # [HOOK] augment loss
        aug_loss, logs_aug = self.hook_ae_compute_ave_aug_loss(zs=zs, xs_partial_recon=xs_partial_recon, xs_targ=xs_targ)
        # compute combined loss
        loss = 0
        if not self.cfg.disable_rec_loss: loss += recon_loss
        if not self.cfg.disable_aug_loss: loss += aug_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # return values
        return loss, {
            **logs_intercept_zs,
            **logs_recon,
            **logs_aug,
            'recon_loss': recon_loss,
            'aug_loss': aug_loss,
        }

    # --------------------------------------------------------------------- #
    # Overrideable                                                          #
    # --------------------------------------------------------------------- #

    def hook_ae_intercept_zs(self, zs: Sequence[torch.Tensor]) -> Tuple[Sequence[torch.Tensor], Dict[str, Any]]:
        return zs, {}

    def hook_ae_compute_ave_aug_loss(self, zs: Sequence[torch.Tensor], xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]) -> Tuple[Union[torch.Tensor, Number], Dict[str, Any]]:
        return 0, {}

    def compute_ave_recon_loss(self, xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]) -> Tuple[Union[torch.Tensor, Number], Dict[str, Any]]:
        # compute reconstruction loss
        pixel_loss = self.recon_handler.compute_ave_loss_from_partial(xs_partial_recon, xs_targ)
        # return logs
        return pixel_loss, {
            'pixel_loss': pixel_loss
        }

    # --------------------------------------------------------------------- #
    # AE Model Utility Functions (Visualisation)                            #
    # --------------------------------------------------------------------- #

    @final
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get the deterministic latent representation (useful for visualisation)"""
        return self._model.encode(x)

    @final
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z into reconstruction x_recon (useful for visualisation)"""
        return self.recon_handler.activate(self._model.decode(z))

    @final
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Feed through the full deterministic model (useful for visualisation)"""
        return self.decode(self.encode(batch))

    # --------------------------------------------------------------------- #
    # AE Model Utility Functions (Training)                                 #
    # --------------------------------------------------------------------- #

    @final
    def decode_partial(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z into partial reconstructions that exclude the final activation if there is one."""
        return self._model.decode(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
