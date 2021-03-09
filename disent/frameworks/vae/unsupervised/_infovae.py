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
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions import Normal

from disent.frameworks.vae.unsupervised import Vae


# ========================================================================= #
# Dfc Vae                                                                   #
# ========================================================================= #


class InfoVae(Vae):
    """
    InfoVAE: Balancing Learning and Inference in Variational Autoencoders
    https://arxiv.org/pdf/1706.02262.pdf

    TODO: this needs to be verified, things have changed
    Reference implementation is from: https://github.com/AntixK/PyTorch-VAE
    """

    REQUIRED_OBS = 1

    @dataclass
    class cfg(Vae.cfg):
        info_alpha: float = -0.5
        info_lambda: float = 5.0
        info_kernel: str = 'imq'
        # what is this? I don't think this should be configurable
        z_var: float = 2.

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # checks
        assert self.cfg.info_alpha <= 0, f'cfg.info_alpha must be <= zero, current value is: {self.cfg.info_alpha}'

    # --------------------------------------------------------------------- #
    # Overrides                                                             #
    # --------------------------------------------------------------------- #

    def compute_ave_reg_loss(self, ds_posterior: Sequence[Normal], ds_prior: Sequence[Normal], zs_sampled):
        # only supports one input observation at the moment
        (d_posterior,), (d_prior,), (z_sampled,) = ds_posterior, ds_prior, zs_sampled

        # compute kl divergence
        kl_loss = self.latents_handler.compute_ave_kl_loss(ds_posterior, ds_prior, zs_sampled)
        # compute maximum-mean discrepancy
        mmd_loss = self._compute_mmd(d_posterior, d_prior, z_sampled)

        # get weights TODO: do we really need these?
        kld_weight = 1.0  # account for minibatch samples from dataset, orig value was 0.005
        mmd_weight = 1.0 / (len(z_sampled) * (len(z_sampled) - 1))  # bias correction

        # weight the loss terms
        kl_reg_loss  = kld_weight * (1 - self.cfg.info_alpha) * kl_loss
        mmd_reg_loss = mmd_weight * (self.cfg.info_alpha + self.cfg.info_lambda - 1) * mmd_loss

        # compute combined loss
        combined_loss = kl_reg_loss + mmd_reg_loss

        # return logs
        return combined_loss, {
            'kl_loss': kl_loss,
            'kl_reg_loss': kl_reg_loss,
            'mmd_loss': mmd_loss,
            'mmd_reg_loss': mmd_reg_loss,
        }

    def _compute_mmd(self, d_posterior: Distribution, d_prior: Distribution, z_sampled) -> Tensor:
        # Sample from prior (Gaussian) distribution
        z_posterior = z_sampled  # TODO: should this rather be d_posterior.sample()?
        z_prior = d_prior.sample()
        # check sizes
        assert z_posterior.shape == z_prior.shape == z_sampled.shape
        # compute kernels
        kernel_pz_pz = self._compute_kernel(z_prior, z_prior)
        kernel_pz_qz = self._compute_kernel(z_prior, z_posterior)
        kernel_qz_qz = self._compute_kernel(z_posterior, z_posterior)
        # maximum-mean discrepancy
        mmd = kernel_pz_pz.mean() - 2*kernel_pz_qz.mean() + kernel_qz_qz.mean()
        return mmd

    def _compute_kernel(self, z0: Tensor, z1: Tensor) -> Tensor:
        batch_size, z_size = z0.shape
        # convert tensors
        z0 = z0.unsqueeze(-2)  # convert to column tensor
        z1 = z1.unsqueeze(-3)  # convert to row tensor
        # in our case this is not required, however it is useful
        # if z0 and z1 have different sizes along the 0th dimension.
        z0 = z0.expand(batch_size, batch_size, z_size)
        z1 = z1.expand(batch_size, batch_size, z_size)
        # compute correct kernel
        if self.cfg.info_kernel == 'rbf':
            return self._compute_kernel_rbf(z0, z1)
        elif self.cfg.info_kernel == 'imq':
            return self._compute_kernel_imq(z0, z1)
        else:
            raise KeyError(f'invalid cfg.info_kernel: {self.cfg.info_kernel}')

    def _compute_kernel_rbf(self, z0: Tensor, z1: Tensor) -> Tensor:
        """RBF Kernel"""
        z_size = z0.shape[-1]
        sigma = 2. * z_size * self.cfg.z_var
        result = torch.exp(-((z0 - z1).pow(2).mean(-1) / sigma))
        return result

    def _compute_kernel_imq(self, z0: Tensor, z1: Tensor, eps: float = 1e-7) -> Tensor:
        """Inverse Multi-Quadratics Kernel"""
        z_size = z0.shape[-1]
        sigma = 2 * z_size * self.cfg.z_var
        kernel = sigma / (eps + sigma + (z0 - z1).pow(2).sum(dim=-1))
        # remove diagonal values
        result = kernel.sum() - kernel.diag().sum()
        return result


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
