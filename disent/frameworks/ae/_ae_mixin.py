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
from dataclasses import dataclass
from typing import Dict
from typing import final
from typing import Tuple

import torch

from disent.frameworks import DisentFramework
from disent.frameworks.helper.reconstructions import make_reconstruction_loss
from disent.frameworks.helper.reconstructions import ReconLossHandler
from disent.model import AutoEncoder


log = logging.getLogger(__name__)


# ========================================================================= #
# framework_vae                                                             #
# ========================================================================= #


class _AeAndVaeMixin(DisentFramework):
    """
    Base class containing common logic for both the Ae and Vae classes.
    -- This is private because handling of both classes needs to be conducted differently.
       An instance of a Vae should not be an instance of an Ae and vice-versa
    """

    @dataclass
    class cfg(DisentFramework.cfg):
        recon_loss: str = 'mse'
        # multiple reduction modes exist for the various loss components.
        # - 'sum': sum over the entire batch
        # - 'mean': mean over the entire batch
        # - 'mean_sum': sum each observation, returning the mean sum over the batch
        loss_reduction: str = 'mean'
        # disable various components
        detach_decoder: bool = False
        disable_rec_loss: bool = False
        disable_aug_loss: bool = False

    # --------------------------------------------------------------------- #
    # AE/VAE Attributes                                                     #
    # --------------------------------------------------------------------- #

    @property
    def REQUIRED_Z_MULTIPLIER(self) -> int:
        raise NotImplementedError

    @property
    def REQUIRED_OBS(self) -> int:
        raise NotImplementedError

    @final
    @property
    def recon_handler(self) -> ReconLossHandler:
        return self.__recon_handler

    # --------------------------------------------------------------------- #
    # AE/VAE Init                                                           #
    # --------------------------------------------------------------------- #

    # attributes provided by this class and initialised in _init_ae_mixin
    _model: AutoEncoder
    __recon_handler: ReconLossHandler

    def _init_ae_mixin(self, model: AutoEncoder):
        # vae model
        self._model = model
        # check the model
        assert isinstance(self._model, AutoEncoder), f'model must be an instance of {AutoEncoder.__name__}, got: {type(model)}'
        assert self._model.z_multiplier == self.REQUIRED_Z_MULTIPLIER, f'model z_multiplier is {repr(self._model.z_multiplier)} but {self.__class__.__name__} requires that it is: {repr(self.REQUIRED_Z_MULTIPLIER)}'
        # recon loss & activation fn
        self.__recon_handler: ReconLossHandler = make_reconstruction_loss(self.cfg.recon_loss, reduction=self.cfg.loss_reduction)

    # --------------------------------------------------------------------- #
    # AE/VAE Training Step Helper                                           #
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

    # --------------------------------------------------------------------- #
    # AE/VAE Model Utility Functions (Visualisation)                        #
    # --------------------------------------------------------------------- #

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get the deterministic latent representation (useful for visualisation)"""
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z into reconstruction x_recon (useful for visualisation)"""
        raise NotImplementedError

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Feed through the full deterministic model (useful for visualisation)"""
        raise NotImplementedError

    # --------------------------------------------------------------------- #
    # AE/VAE Model Utility Functions (Training)                             #
    # --------------------------------------------------------------------- #

    def decode_partial(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z into partial reconstructions that exclude the final activation if there is one."""
        raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
