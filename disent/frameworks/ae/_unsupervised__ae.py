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
from numbers import Number
from typing import Any
from typing import Dict
from typing import final
from typing import Sequence
from typing import Tuple
from typing import Union

import torch

from disent.frameworks._ae_mixin import _AeAndVaeMixin
from disent.frameworks.helper.util import detach_all
from disent.model import AutoEncoder
from disent.util.iters import map_all


# ========================================================================= #
# framework_vae                                                             #
# ========================================================================= #


class Ae(_AeAndVaeMixin):
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

    # override
    REQUIRED_Z_MULTIPLIER = 1
    REQUIRED_OBS = 1

    @dataclass
    class cfg(_AeAndVaeMixin.cfg):
        pass

    def __init__(self, model: AutoEncoder, cfg: cfg = None, batch_augment=None):
        super().__init__(cfg=cfg, batch_augment=batch_augment)
        # initialise the auto-encoder mixin (recon handler, model, enc, dec, etc.)
        self._init_ae_mixin(model=model)

    # --------------------------------------------------------------------- #
    # AE Training Step -- Overridable                                       #
    # --------------------------------------------------------------------- #

    @final
    def do_training_step(self, batch, batch_idx):
        xs, xs_targ = self._get_xs_and_targs(batch, batch_idx)

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent variables
        zs = map_all(self.encode, xs)
        # [HOOK] intercept latent variables
        zs, logs_intercept_zs = self.hook_ae_intercept_zs(zs)
        # reconstruct without the final activation
        xs_partial_recon = map_all(self.decode_partial, detach_all(zs, if_=self.cfg.detach_decoder))
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

        # log general
        self.log_dict({
            **logs_intercept_zs,
            **logs_recon,
            **logs_aug,
        })

        # log progress bar
        self.log_dict({
            'recon_loss': float(recon_loss),
            'aug_loss': float(aug_loss),
        }, prog_bar=True)

        # return values
        return loss

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
