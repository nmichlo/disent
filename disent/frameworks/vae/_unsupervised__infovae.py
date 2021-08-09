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

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal

from disent.frameworks.vae._unsupervised__vae import Vae


# ========================================================================= #
# InfoVae                                                                   #
# ========================================================================= #


class InfoVae(Vae):
    """
    InfoVAE: Balancing Learning and Inference in Variational Autoencoders
    https://arxiv.org/pdf/1706.02262.pdf

    TODO: this is not verified
    Reference implementation is from: https://github.com/AntixK/PyTorch-VAE
    Changes:
        1. kernels are computed weirdly in this implementation
        2. uses unbiased MMD estimates from https://arxiv.org/pdf/1505.03906.pdf
        3. computes means, not sums
    """

    REQUIRED_OBS = 1

    @dataclass
    class cfg(Vae.cfg):
        info_alpha: float = -0.5
        info_lambda: float = 5.0
        info_kernel: str = 'rbf'
        # what is this? I don't think this should be configurable
        z_var: float = 2.
        # this is optional
        maintain_reg_ratio: bool = True

    def __init__(self, model: 'AutoEncoder', cfg: cfg = None, batch_augment=None):
        super().__init__(model=model, cfg=cfg, batch_augment=batch_augment)
        # checks
        assert self.cfg.info_alpha <= 0, f'cfg.info_alpha must be <= zero, current value is: {self.cfg.info_alpha}'
        assert self.cfg.loss_reduction == 'mean', 'InfoVAE only supports cfg.loss_reduction == "mean"'

    # --------------------------------------------------------------------- #
    # Overrides                                                             #
    # --------------------------------------------------------------------- #

    def compute_ave_reg_loss(self, ds_posterior: Sequence[Normal], ds_prior: Sequence[Normal], zs_sampled):
        """
        TODO: This could be wrong?
        """

        # only supports one input observation at the moment
        (d_posterior,), (d_prior,), (z_sampled,) = ds_posterior, ds_prior, zs_sampled

        # compute kl divergence
        # compute maximum-mean discrepancy
        kl_loss = self.latents_handler.compute_ave_kl_loss(ds_posterior, ds_prior, zs_sampled)
        mmd_loss = self._compute_mmd(z_posterior_samples=z_sampled, z_prior_samples=d_prior.rsample())

        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # original loss Sums Everything, we use the mean and scale everything to keep the ratios the same
        # OLD: (C*W*H) * recon_mean + (Z)         * kl_mean + (Z)         * mmd_mean
        # NEW:           recon_mean + (Z)/(C*W*H) * kl_mean + (Z)/(C*W*H) * mmd_mean
        # compute the weight
        # TODO: maybe this should be standardised to something like Z=9, W=64, H=64, C=3
        # TODO: this could be moved into other models
        reg_weight = (self._model.z_size / np.prod(self._model.x_shape)) if self.cfg.maintain_reg_ratio else 1.0
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # weight the loss terms
        kl_reg_loss = reg_weight * (1 - self.cfg.info_alpha) * kl_loss
        mmd_reg_loss = reg_weight * (self.cfg.info_alpha + self.cfg.info_lambda - 1) * mmd_loss
        # compute combined loss
        combined_loss = kl_reg_loss + mmd_reg_loss

        # return logs
        return combined_loss, {
            'kl_loss': kl_loss,
            'kl_reg_loss': kl_reg_loss,
            'mmd_loss': mmd_loss,
            'mmd_reg_loss': mmd_reg_loss,
        }

    def _compute_mmd(self, z_posterior_samples: Tensor, z_prior_samples: Tensor) -> Tensor:
        """
        (✓) visual inspection against:
            https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Kernel_two-sample_test
        """
        # check sizes - these conditions can be relaxed in practice, just for debugging
        assert z_posterior_samples.ndim == 2
        assert z_posterior_samples.shape == z_prior_samples.shape
        # compute kernels: (B, Z) -> (,)
        mean_pz_pz = self._compute_unbiased_mean(self._compute_kernel(z_prior_samples, z_prior_samples), unbaised=True)
        mean_pz_qz = self._compute_unbiased_mean(self._compute_kernel(z_prior_samples, z_posterior_samples), unbaised=False)
        mean_qz_qz = self._compute_unbiased_mean(self._compute_kernel(z_posterior_samples, z_posterior_samples), unbaised=True)
        # maximum-mean discrepancy
        mmd = mean_pz_pz - 2*mean_pz_qz + mean_qz_qz
        return mmd

    def _compute_unbiased_mean(self, kernel: Tensor, unbaised: bool) -> Tensor:
        """
        (✓) visual inspection against equation (8) of
            Training generative neural networks via Maximum Mean Discrepancy optimization
            https://arxiv.org/pdf/1505.03906.pdf
        """
        # (B, B) == (N, M) ie. N=B and M=B
        N, M = kernel.shape
        assert N == M
        # compute mean along first and second dims
        if unbaised:
            # diagonal stacks values along last dimension ie. (B, B, Z) -> (Z, B) or (B, B) -> (B,)
            sum_kernel = kernel.sum(dim=(0, 1)) - torch.diagonal(kernel, dim1=0, dim2=1).sum(dim=-1)  # (B, B,) -> (,)
            # compute unbiased mean
            mean_kernel = sum_kernel / (N*(N-1))
        else:
            mean_kernel = kernel.mean(dim=(0, 1))  # (B, B,) -> (,)
        # check size again
        assert mean_kernel.ndim == 0
        return mean_kernel

    def _compute_kernel(self, z0: Tensor, z1: Tensor) -> Tensor:
        """
        (✓) visual inspection against:
            https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Kernel_two-sample_test
        """
        batch_size, z_size = z0.shape
        # convert tensors
        z0 = z0.unsqueeze(-2)  # convert to column tensor  # [B, Z] -> [B, 1, Z]
        z1 = z1.unsqueeze(-3)  # convert to row tensor     # [B, Z] -> [1, B, Z]
        # in our case this is not required, however it is useful
        # if z0 and z1 have different sizes along the 0th dimension.
        z0 = z0.expand(batch_size, batch_size, z_size)     # [B, 1, Z] -> [B, B, Z]
        z1 = z1.expand(batch_size, batch_size, z_size)     # [1, B, Z] -> [B, B, Z]
        # compute correct kernel
        if self.cfg.info_kernel == 'rbf':
            kernel = self._kernel_rbf(z0, z1)
        # elif self.cfg.info_kernel == 'imq':
        #     kernel = self._kernel_imq(z0, z1)
        else:  # pragma: no cover
            raise KeyError(f'invalid cfg.info_kernel: {self.cfg.info_kernel}')
        # check result size
        assert kernel.shape == (batch_size, batch_size)
        return kernel

    def _kernel_rbf(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Radial Basis Function (RBF) Kernel a.k.a. Gaussian Kernel
        k(x, y) = exp(- ||x - y||^2 / (2*sigma^2))

        (✓) visual inspection against:
            https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space#Radial_basis_function_kernels

        TODO: how do we arrive at the value for sigma?
              - multiplying sigma by z_size is that same as computing .mean(dim=-1)
                instead of the current sum
        TODO: do we treat each latent variable separately? or as vectors like now due to the .sum?
        """
        z_size = x.shape[-1]
        sigma = 2 * self.cfg.z_var * z_size
        kernel = torch.exp(-((x - y).pow(2).sum(dim=-1) / sigma))
        return kernel

    # def _kernel_imq(self, x: Tensor, y: Tensor, eps: float = 1e-7) -> Tensor:
    #     """
    #     Inverse Multi-Quadratics Kernel
    #     k(x, y) = (c^2 + ||x - y||^2)^b
    #         c ∈ R
    #         b < 0 but better if b ∈ (0, 1)
    #
    #     TODO: This could be wrong?
    #     # TODO: how do we arrive at the value for c
    #     """
    #     z_size = x.shape[-1]
    #     c = 2 * self.cfg.z_var * z_size
    #     kernel = c / (eps + c + (x - y).pow(2).sum(-1))
    #     return kernel


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
