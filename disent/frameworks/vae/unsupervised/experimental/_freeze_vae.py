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
from typing import final
import torch
from disent.frameworks.vae.unsupervised import BetaVae
from disent.util import DisentModule
import logging

log = logging.getLogger(__name__)


class Noop(DisentModule):
    def forward(self, x):
        return x


# ========================================================================= #
# Beta-VAE Loss                                                             #
# ========================================================================= #


class FreezeVae(BetaVae):

    @dataclass
    class cfg(BetaVae.cfg):
        beta: float = 0
        freeze_after_steps: int = 100
        freeze_beta: float = 0.03
        freeze_hidden_size: int = 256

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # NOOP, replaced after time
        self._encoder_mu_swapper = Noop()
        self._encoder_logvar_swapper = Noop()
        self._decoder_mu_swapper = Noop()
        # model
        self._NEW_encoder_mu_swapper = torch.nn.Sequential(
            torch.nn.Linear(in_features=self._model.z_size, out_features=self.cfg.freeze_hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=self.cfg.freeze_hidden_size, out_features=self._model.z_size),
        )
        self._NEW_encoder_logvar_swapper = torch.nn.Sequential(
            torch.nn.Linear(in_features=self._model.z_size, out_features=self.cfg.freeze_hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=self.cfg.freeze_hidden_size, out_features=self._model.z_size),
        )
        # model
        self._NEW_decoder_mu_swapper = torch.nn.Sequential(
            torch.nn.Linear(in_features=self._model.z_size, out_features=self.cfg.freeze_hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=self.cfg.freeze_hidden_size, out_features=self._model.z_size),
        )
        # steps
        self._steps = 0

    def training_regularize_kl(self, kl_loss):
        # do stuff
        if self._steps == self.cfg.freeze_after_steps:
            log.warning('Freezing weights and adding swap layers!')
            # replace beta
            self.cfg.beta = self.cfg.freeze_beta
            # replace models
            self._encoder_mu_swapper = self._NEW_encoder_mu_swapper
            self._encoder_logvar_swapper = self._NEW_encoder_logvar_swapper
            self._decoder_mu_swapper = self._NEW_decoder_mu_swapper
            # freeze weights
            for params in self._model.parameters():
                params.requires_grad = False
        # update
        self._steps += 1
        # BETA VAE
        return self.cfg.beta * kl_loss

    @final
    def training_encode_params(self, x: torch.Tensor) -> 'Params':
        """Get parametrisations of the latent distributions, which are sampled from during training."""
        z_mean, z_logvar = self._model.encode(x)
        # swap
        z_mean_swap = self._encoder_mu_swapper(z_mean)
        z_logvar_swap = self._encoder_logvar_swapper(z_mean)  # THIS IS Z_MEAN ON PURPOSE ... TODO: test alternative...
        # get params
        z_params = self._distributions.encoding_to_params((z_mean_swap, z_logvar_swap))
        return z_params

    @final
    def training_decode_partial(self, z_swap: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z into partial reconstructions that exclude the final activation if there is one."""
        z = self._decoder_mu_swapper(z_swap)
        return self._model.decode(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
