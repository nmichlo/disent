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

# ========================================================================= #
# Beta-TC-VAE                                                               #
# ========================================================================= #


class BetaTcVae(BetaVae):
    """
    β-TCVAE - Isolating Sources of Disentanglement in VAEs
    https://arxiv.org/abs/1802.04942

    TODO: verify
    TODO: loss scales are not correct!
    TODO: simplify, some code is duplicated!
    Reference implementation is from: https://github.com/amir-abdi/disentanglement-pytorch
    """

    REQUIRED_OBS = 1

    @dataclass
    class cfg(BetaVae.cfg):
        """
        Equation (4) with alpha=gamma=1 can be written as ELBO+(1-beta)*TC
        """

    # --------------------------------------------------------------------- #
    # Overrides                                                             #
    # --------------------------------------------------------------------- #

    def compute_ave_reg_loss(self, ds_posterior: Sequence[Normal], ds_prior: Sequence[Normal], zs_sampled):
        # compute kl loss
        # TODO: this should be KL instead? not KL Reg?
        kl_reg_loss, logs_kl_reg = super().compute_ave_reg_loss(ds_posterior, ds_prior, zs_sampled)
        # compute dip loss
        tc_reg_loss, logs_tc_reg = compute_ave_loss_and_logs(self._betatc_compute_loss, ds_posterior, zs_sampled)
        # combine
        combined_loss = kl_reg_loss + tc_reg_loss
        # return logs
        return combined_loss, {
            **logs_kl_reg,   # kl_loss, kl_reg_loss
            **logs_tc_reg,   #
        }

    # --------------------------------------------------------------------- #
    # Helper                                                                #
    # --------------------------------------------------------------------- #

    def _betatc_compute_loss(self, d_posterior: Normal, z_sampled):
        tc_loss = BetaTcVae._betatc_compute_total_correlation(
            z_sampled=z_sampled,
            z_mean=d_posterior.mean,
            z_logvar=torch.log(d_posterior.variance),
        )
        tc_reg_loss = (self.cfg.beta - 1.) * tc_loss
        return tc_reg_loss, {
            'tc_loss': tc_loss,
            'tc_reg_loss': tc_reg_loss,
        }

    @staticmethod
    def _betatc_compute_total_correlation(z_sampled, z_mean, z_logvar):
        """
        Estimate total correlation over a batch.
        Reference implementation is from: https://github.com/amir-abdi/disentanglement-pytorch
        """
        # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
        # tensor of size [batch_size, batch_size, num_latents]. In the following
        # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
        log_qz_prob = BetaTcVae._betatc_compute_gaussian_log_density(z_sampled.unsqueeze(dim=1), z_mean.unsqueeze(dim=0), z_logvar.unsqueeze(dim=0))

        # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
        # + constant) for each sample in the batch, which is a vector of size
        # [batch_size,].
        log_qz_product = log_qz_prob.exp().sum(dim=1, keepdim=False).log().sum(dim=1, keepdim=False)

        # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
        # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
        log_qz = log_qz_prob.sum(dim=2, keepdim=False).exp().sum(dim=1, keepdim=False).log()

        return (log_qz - log_qz_product).mean()

    @staticmethod
    def _betatc_compute_gaussian_log_density(samples, mean, log_var):
        """
        Estimate the log density of a Gaussian distribution
        Reference implementation is from: https://github.com/amir-abdi/disentanglement-pytorch
        """
        # TODO: can this be replaced with some variant of Normal.log_prob?
        import math
        pi = torch.tensor(math.pi, requires_grad=False)
        normalization = torch.log(2 * pi)
        inv_sigma = torch.exp(-log_var)
        tmp = samples - mean
        return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)