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
from typing import Sequence

import torch
from torch.distributions import Normal

from disent.frameworks.helper.util import compute_ave_loss_and_logs
from disent.frameworks.vae._unsupervised__betavae import BetaVae
from disent.nn.functional import torch_cov_matrix


# ========================================================================= #
# Dfc Vae                                                                   #
# ========================================================================= #


class DipVae(BetaVae):
    """
    Disentangled Inferred Prior Variational Auto-Encoder (DIP-VAE)
    https://arxiv.org/pdf/1711.00848.pdf

    Reference implementation is from: https://github.com/AntixK/PyTorch-VAE
    """

    REQUIRED_OBS = 1

    @dataclass
    class cfg(BetaVae.cfg):
        dip_mode: str = 'ii'
        dip_beta: float = 1.0
        lambda_d: float = 10.
        lambda_od: float = 5.

    def __init__(self, model: 'AutoEncoder', cfg: cfg = None, batch_augment=None):
        super().__init__(model=model, cfg=cfg, batch_augment=batch_augment)
        # checks
        assert self.cfg.dip_mode in {'i', 'ii'}, f'unsupported dip_mode={repr(self.cfg.dip_mode)} for {self.__class__.__name__}. Must be one of: {{"i", "ii"}}'
        assert self.cfg.dip_beta >= 0, 'dip_beta must be >= 0'
        assert self.cfg.lambda_d >= 0, 'lambda_d must be >= 0'
        assert self.cfg.lambda_od >= 0, 'lambda_od must be >= 0'

    # --------------------------------------------------------------------- #
    # Overrides                                                             #
    # --------------------------------------------------------------------- #

    def compute_ave_reg_loss(self, ds_posterior: Sequence[Normal], ds_prior: Sequence[Normal], zs_sampled):
        # compute kl loss
        kl_reg_loss, logs_kl_reg = super().compute_ave_reg_loss(ds_posterior, ds_prior, zs_sampled)
        # compute dip loss
        dip_reg_loss, logs_dip_reg = compute_ave_loss_and_logs(self._dip_compute_loss, ds_posterior)
        # combine
        combined_loss = kl_reg_loss + dip_reg_loss
        # return logs
        return combined_loss, {
            **logs_kl_reg,   # kl_loss, kl_reg_loss
            **logs_dip_reg,  # dip_loss_d, dip_loss_od, dip_loss, dip_reg_loss,
        }

    # --------------------------------------------------------------------- #
    # Helper                                                                #
    # --------------------------------------------------------------------- #

    def _dip_compute_loss(self, d_posterior: Normal):
        cov_matrix = self._dip_estimate_cov_matrix(d_posterior)
        return self._dip_compute_regulariser(cov_matrix)

    def _dip_compute_regulariser(self, cov_matrix):
        """
        Compute DIP regularises for diagonal & off diagonal components
        - covariance matrix should match the identity matrix
        - diagonal and off diagonals are weighted differently
        """
        # covariance diagonal & off diagonal components
        cov_diag = torch.diagonal(cov_matrix)
        cov_off_diag = cov_matrix - torch.diag(cov_diag)
        # regulariser, encourage covariance to match identity matrix
        # TODO: this is all summed, we should rather calculate the means of each so that the scale stays the same after changing z_size
        dip_loss_d = torch.sum((cov_diag - 1)**2)  # / num_diag
        dip_loss_od = torch.sum(cov_off_diag**2)   # / num_off_diag
        dip_loss = (self.cfg.lambda_d * dip_loss_d) + (self.cfg.lambda_od * dip_loss_od)
        # scale dip loss - like beta for beta vae
        dip_reg_loss = self.cfg.dip_beta * dip_loss
        # return logs
        return dip_reg_loss, {
            'dip_loss_d': dip_loss_d,
            'dip_loss_od': dip_loss_od,
            'dip_loss': dip_loss,
            'dip_reg_loss': dip_reg_loss,
        }

    def _dip_estimate_cov_matrix(self, d_posterior: Normal):
        z_mean, z_var = d_posterior.mean, d_posterior.variance
        # compute covariance over batch
        cov_z_mean = torch_cov_matrix(z_mean)
        # compute covariance matrix based on mode
        if self.cfg.dip_mode == "i":
            cov_matrix = cov_z_mean
        elif self.cfg.dip_mode == "ii":
            # E[var]
            E_var = torch.mean(torch.diag(z_var), dim=0)
            cov_matrix = cov_z_mean + E_var
        else:  # pragma: no cover
            raise KeyError(f'Unknown DipVAE mode: {self.cfg.dip_mode}')
        # shape: (Z, Z)
        return cov_matrix


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
