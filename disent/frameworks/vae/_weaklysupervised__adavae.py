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

from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

import torch
from dataclasses import dataclass
from torch.distributions import Distribution
from torch.distributions import Normal

from disent.frameworks.vae._unsupervised__betavae import BetaVae


# ========================================================================= #
# Ada-GVAE                                                                  #
# ========================================================================= #


class AdaVae(BetaVae):
    """
    Weakly Supervised Disentanglement Learning Without Compromises: https://arxiv.org/abs/2002.02886
    - pretty much a beta-vae with averaging between decoder outputs to form weak supervision signal.
    - GAdaVAE:   Averaging from https://arxiv.org/abs/1809.02383
    - ML-AdaVAE: Averaging from https://arxiv.org/abs/1705.08841

    MODIFICATION:
    - Symmetric KL Calculation used by default, described in: https://arxiv.org/pdf/2010.14407.pdf
    - adjustable threshold value
    """

    REQUIRED_OBS = 2

    @dataclass
    class cfg(BetaVae.cfg):
        ada_average_mode: str = 'gvae'
        ada_thresh_mode: str = 'symmetric_kl'  # kl, symmetric_kl, dist, sampled_dist
        ada_thresh_ratio: float = 0.5

    def hook_intercept_ds(self, ds_posterior: Sequence[Distribution], ds_prior: Sequence[Distribution]) -> Tuple[Sequence[Distribution], Sequence[Distribution], Dict[str, Any]]:
        """
        Adaptive VAE Glue Method, putting the various components together
        1. find differences between deltas
        2. estimate a threshold for differences
        3. compute a shared mask from this threshold
        4. average together elements that should be considered shared

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (aggregate_argmax)
        """
        d0_posterior, d1_posterior = ds_posterior
        # shared elements that need to be averaged, computed per pair in the batch.
        share_mask = self.compute_posterior_shared_mask(d0_posterior, d1_posterior, thresh_mode=self.cfg.ada_thresh_mode, ratio=self.cfg.ada_thresh_ratio)
        # compute average posteriors
        new_ds_posterior = self.make_averaged_distributions(d0_posterior, d1_posterior, share_mask, average_mode=self.cfg.ada_average_mode)
        # return new args & generate logs
        return new_ds_posterior, ds_prior, {
            'shared': share_mask.sum(dim=1).float().mean()
        }

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # HELPER                                                                #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @classmethod
    def compute_posterior_deltas(cls, d0_posterior: Distribution, d1_posterior: Distribution, thresh_mode: str):
        """
        (✓) Visual inspection against reference implementation
        https://github.com/google-research/disentanglement_lib (compute_kl)
        - difference is that they don't multiply by 0.5 to get true kl, but that's not needed

        TODO: this might be numerically unstable with f32 passed to distributions
        """
        # shared elements that need to be averaged, computed per pair in the batch.
        # [𝛿_i ...]
        if thresh_mode == 'kl':
            # ORIGINAL
            deltas = torch.distributions.kl_divergence(d1_posterior, d0_posterior)
        elif thresh_mode == 'symmetric_kl':
            # FROM: https://openreview.net/pdf?id=8VXvj1QNRl1
            kl_deltas_d1_d0 = torch.distributions.kl_divergence(d1_posterior, d0_posterior)
            kl_deltas_d0_d1 = torch.distributions.kl_divergence(d0_posterior, d1_posterior)
            deltas = (0.5 * kl_deltas_d1_d0) + (0.5 * kl_deltas_d0_d1)
        elif thresh_mode == 'dist':
            deltas = cls.compute_z_deltas(d1_posterior.mean, d0_posterior.mean)
        elif thresh_mode == 'sampled_dist':
            deltas = cls.compute_z_deltas(d1_posterior.rsample(), d0_posterior.rsample())
        else:
            raise KeyError(f'invalid thresh_mode: {repr(thresh_mode)}')

        # return values
        return deltas

    @classmethod
    def compute_posterior_shared_mask(cls, d0_posterior: Distribution, d1_posterior: Distribution, thresh_mode: str, ratio=0.5):
        return cls.estimate_shared_mask(z_deltas=cls.compute_posterior_deltas(d0_posterior, d1_posterior, thresh_mode=thresh_mode), ratio=ratio)

    @classmethod
    def compute_z_deltas(cls, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        return torch.abs(z0 - z1)

    @classmethod
    def compute_z_shared_mask(cls, z0: torch.Tensor, z1: torch.Tensor, ratio: float = 0.5):
        return cls.estimate_shared_mask(z_deltas=cls.compute_z_deltas(z0, z1), ratio=ratio)

    @classmethod
    def estimate_shared_mask(cls, z_deltas: torch.Tensor, ratio: float = 0.5):
        """
        Core of the adaptive VAE algorithm, estimating which factors
        have changed (or in this case which are shared and should remained unchanged
        by being be averaged) between pairs of observations.

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (aggregate_argmax)
            - Implementation conversion is non-trivial, items are histogram binned.
              If we are in the second histogram bin, ie. 1, then kl_deltas <= kl_threshs
            - TODO: (aggregate_labels) An alternative mode exists where you can bind the
                    latent variables to any individual label, by one-hot encoding which
                    latent variable should not be shared: "enforce that each dimension
                    of the latent code learns one factor (dimension 1 learns factor 1)
                    and enforce that each factor of variation is encoded in a single
                    dimension."
        """
        # threshold τ
        z_threshs = cls.estimate_threshold(z_deltas, ratio=ratio)
        # true if 'unchanged' and should be average
        shared_mask = z_deltas < z_threshs
        # return
        return shared_mask

    @classmethod
    def estimate_threshold(cls, kl_deltas: torch.Tensor, keepdim: bool = True, ratio: float = 0.5):
        """
        Compute the threshold for each image pair in a batch of kl divergences of all elements of the latent distributions.
        It should be noted that for a perfectly trained model, this threshold is always correct.

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (aggregate_argmax)
        """
        maximums = kl_deltas.max(axis=1, keepdim=keepdim).values
        minimums = kl_deltas.min(axis=1, keepdim=keepdim).values
        return torch.lerp(minimums, maximums, weight=ratio)

    @classmethod
    def make_averaged_distributions(cls, d0_posterior: Normal, d1_posterior: Normal, share_mask: torch.Tensor, average_mode: str):
        # compute average posterior
        ave_posterior = compute_average_distribution(d0_posterior=d0_posterior, d1_posterior=d1_posterior, average_mode=average_mode)
        # select averages
        ave_z0_posterior = ave_posterior.__class__(
            loc=torch.where(share_mask, ave_posterior.loc, d0_posterior.loc),
            scale=torch.where(share_mask, ave_posterior.scale, d0_posterior.scale),
        )
        ave_z1_posterior = ave_posterior.__class__(
            loc=torch.where(share_mask, ave_posterior.loc, d1_posterior.loc),
            scale=torch.where(share_mask, ave_posterior.scale, d1_posterior.scale),
        )
        # return values
        return ave_z0_posterior, ave_z1_posterior

    @classmethod
    def make_averaged_zs(cls, z0: torch.Tensor, z1: torch.Tensor, share_mask: torch.Tensor):
        ave = 0.5 * z0 + 0.5 * z1
        ave_z0 = torch.where(share_mask, ave, z0)
        ave_z1 = torch.where(share_mask, ave, z1)
        return ave_z0, ave_z1


# ========================================================================= #
# Averaging Functions                                                       #
# ========================================================================= #


def compute_average_gvae(z0_mean: torch.Tensor, z0_var: torch.Tensor, z1_mean: torch.Tensor, z1_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the arithmetic mean of the encoder distributions.
    - Ada-GVAE Averaging function

    (✓) Visual inspection against reference implementation:
        https://github.com/google-research/disentanglement_lib (GroupVAEBase.model_fn)
    """
    # TODO: would the mean of the std be better?
    # averages
    ave_var = 0.5 * (z0_var + z1_var)
    ave_mean = 0.5 * (z0_mean + z1_mean)
    # mean, logvar
    return ave_mean, ave_var  # natural log


def compute_average_ml_vae(z0_mean: torch.Tensor, z0_var: torch.Tensor, z1_mean: torch.Tensor, z1_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the product of the encoder distributions.
    - Ada-ML-VAE Averaging function

    (✓) Visual inspection against reference implementation:
        https://github.com/google-research/disentanglement_lib (MLVae.model_fn)

    # TODO: recheck
    """
    # Diagonal matrix inverse: E^-1 = 1 / E
    # https://proofwiki.org/wiki/Inverse_of_Diagonal_Matrix
    z0_invvar, z1_invvar = z0_var.reciprocal(), z1_var.reciprocal()
    # average var: E^-1 = E1^-1 + E2^-1
    # disentanglement_lib: ave_var = 2 * z0_var * z1_var / (z0_var + z1_var)
    ave_var = 2 * (z0_invvar + z1_invvar).reciprocal()
    # average mean: u^T = (u1^T E1^-1 + u2^T E2^-1) E
    # disentanglement_lib: ave_mean = (z0_mean/z0_var + z1_mean/z1_var) * ave_var * 0.5
    ave_mean = (z0_mean*z0_invvar + z1_mean*z1_invvar) * ave_var * 0.5
    # mean, logvar
    return ave_mean, ave_var  # natural log


COMPUTE_AVE_FNS = {
    'gvae': compute_average_gvae,
    'ml-vae': compute_average_ml_vae,
}


def compute_average(z0_mean: torch.Tensor, z0_var: torch.Tensor, z1_mean: torch.Tensor, z1_var: torch.Tensor, average_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    return COMPUTE_AVE_FNS[average_mode](z0_mean=z0_mean, z0_var=z0_var, z1_mean=z1_mean, z1_var=z1_var)


def compute_average_distribution(d0_posterior: Normal, d1_posterior: Normal, average_mode: str) -> Normal:
    assert isinstance(d0_posterior, Normal) and isinstance(d1_posterior, Normal)
    ave_mean, ave_var = compute_average(
        z0_mean=d0_posterior.mean, z0_var=d0_posterior.variance,
        z1_mean=d1_posterior.mean, z1_var=d1_posterior.variance,
        average_mode=average_mode,
    )
    return Normal(loc=ave_mean, scale=torch.sqrt(ave_var))


# def compute_average_params(z0_params: 'Params', z1_params: 'Params', average_mode: str) -> 'Params':
#     ave_mean, ave_logvar = compute_average(z0_params.mean, z0_params.logvar, z1_params.mean, z1_params.logvar, average_mode=average_mode)
#     return z0_params.__class__(ave_mean, ave_logvar)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
