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

from dataclasses import dataclass
from typing import Dict
from typing import final
from typing import Sequence
from typing import Tuple

import torch

from disent.frameworks.helper.reconstructions import ReconLossHandler, make_reconstruction_loss
from disent.model.ae.base import AutoEncoder
from disent.frameworks.framework import BaseFramework
from disent.util import aggregate_dict
from disent.util import collect_dicts

from disent.util import map_all


# ========================================================================= #
# framework_vae                                                             #
# ========================================================================= #


class AE(BaseFramework):
    """
    Basic Auto Encoder
    """

    REQUIRED_Z_MULTIPLIER = 1

    @dataclass
    class cfg(BaseFramework.cfg):
        recon_loss: str = 'mse'
        # multiple reduction modes exist for the various loss components.
        # - 'sum': sum over the entire batch
        # - 'mean': mean over the entire batch
        # - 'mean_sum': sum each observation, returning the mean sum over the batch
        loss_reduction: str = 'mean'

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, batch_augment=batch_augment, cfg=cfg)
        # vae model
        assert callable(make_model_fn)
        self._model: AutoEncoder = make_model_fn()
        # check the model
        assert isinstance(self._model, AutoEncoder)
        assert self._model.z_multiplier == self.REQUIRED_Z_MULTIPLIER, f'model z_multiplier is {repr(self._model.z_multiplier)} but {self.__class__.__name__} requires that it is: {repr(self.REQUIRED_Z_MULTIPLIER)}'
        # recon loss & activation fn
        self._recon_handler: ReconLossHandler = make_reconstruction_loss(self.cfg.recon_loss)

    # --------------------------------------------------------------------- #
    # AE Training Step -- Overridable                                       #
    # --------------------------------------------------------------------- #

    @final
    def do_training_step(self, batch, batch_idx):
        xs, xs_targ, = batch['x'], batch['x_targ']

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent variables
        zs = map_all(self.encode_params, xs)
        # intercept latent variables
        zs, logs_intercept = self.training_intercept_zs(zs)
        # reconstruct without the final activation
        xs_partial_recon = map_all(self.decode_partial, zs)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # compute all the recon losses
        recon_loss, logs_recon = map_all(self.compute_reconstruction_loss, xs_partial_recon, xs_targ, collect_returned=True)
        recon_loss = sum(recon_loss) / len(recon_loss)
        # compute combined loss
        loss = recon_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # return values
        return loss, {
            **logs_intercept,
            **aggregate_dict(collect_dicts(logs_recon)),
            'recon_loss': recon_loss,
        }

    # --------------------------------------------------------------------- #
    # Overrideable                                                          #
    # --------------------------------------------------------------------- #

    def training_intercept_zs(self, zs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        return zs

    def compute_reconstruction_loss(self, x_partial_recon: torch.Tensor, x_targ: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # compute reconstruction loss
        pixel_loss = self.recon_loss(x_partial_recon, x_targ)
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
        return self._recon_handler.activate(self._model.decode(z))

    @final
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Feed through the full deterministic model (useful for visualisation)"""
        return self.decode(self.encode(batch))

    # --------------------------------------------------------------------- #
    # AE Model Utility Functions (Training)                                 #
    # --------------------------------------------------------------------- #

    @final
    def encode_params(self, x: torch.Tensor) -> torch.Tensor:
        """Get parametrisations of the latent distributions, which are sampled from during training."""
        return self._model.encode(x)

    @final
    def decode_partial(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z into partial reconstructions that exclude the final activation if there is one."""
        return self._model.decode(z)

    @final
    def recon_loss(self, x_partial_recon: torch.Tensor, x_targ: torch.Tensor) -> torch.Tensor:
        return self._recon_handler.training_compute_loss(
            x_partial_recon, x_targ, reduction=self.cfg.loss_reduction,
        )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
