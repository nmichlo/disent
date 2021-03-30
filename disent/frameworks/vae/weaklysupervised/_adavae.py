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

from disent.frameworks.vae.unsupervised._betavae import BetaVae


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
    - Symmetric KL Calculation used by default, described in: https://openreview.net/pdf?id=8VXvj1QNRl1
    """

    REQUIRED_OBS = 2

    @dataclass
    class cfg(BetaVae.cfg):
        # TODO: prefix all variables with "ada_"
        ada_average_mode: str = 'gvae'
        ada_thresh_mode: str = 'symmetric_kl'  # kl, symmetric_kl, dist, sampled_dist
        ada_thresh_ratio: float = 0.5

    def hook_intercept_zs(self, zs_params: Sequence['Params']) -> Tuple[Sequence['Params'], Dict[str, Any]]:
        """
        Adaptive VAE Glue Method, putting the various components together
        1. find differences between deltas
        2. estimate a threshold for differences
        3. compute a shared mask from this threshold
        4. average together elements that should be considered shared

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (aggregate_argmax)
        """
        z0_params, z1_params = zs_params
        d0_posterior, _ = self.params_to_dists(z0_params)
        d1_posterior, _ = self.params_to_dists(z1_params)
        # shared elements that need to be averaged, computed per pair in the batch.
        share_mask = self.compute_posterior_shared_mask(d0_posterior, d1_posterior, thresh_mode=self.cfg.ada_thresh_mode, ratio=self.cfg.ada_thresh_ratio)
        # compute average posteriors
        new_zs_params = self.make_averaged_params(z0_params, z1_params, share_mask, average_mode=self.cfg.ada_average_mode)
        # return new args & generate logs
        return new_zs_params, {
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
    def make_averaged_params(cls, z0_params, z1_params, share_mask, average_mode: str):
        # compute average posteriors
        ave_mean, ave_logvar = compute_average(
            z0_params.mean, z0_params.logvar,
            z1_params.mean, z1_params.logvar,
            average_mode=average_mode,
        )
        # select averages
        ave_z0_mean = torch.where(share_mask, ave_mean, z0_params.mean)
        ave_z1_mean = torch.where(share_mask, ave_mean, z1_params.mean)
        ave_z0_logvar = torch.where(share_mask, ave_logvar, z0_params.logvar)
        ave_z1_logvar = torch.where(share_mask, ave_logvar, z1_params.logvar)
        # return values
        return z0_params.__class__(ave_z0_mean, ave_z0_logvar), z1_params.__class__(ave_z1_mean, ave_z1_logvar)


# ========================================================================= #
# Averaging Functions                                                       #
# ========================================================================= #


def compute_average_gvae(z0_mean: torch.Tensor, z0_logvar: torch.Tensor, z1_mean: torch.Tensor, z1_logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the arithmetic mean of the encoder distributions.
    - Ada-GVAE Averaging function

    (✓) Visual inspection against reference implementation:
        https://github.com/google-research/disentanglement_lib (GroupVAEBase.model_fn)
    """
    # helper
    z0_var, z1_var = z0_logvar.exp(), z1_logvar.exp()
    # averages
    ave_var = (z0_var + z1_var) * 0.5
    ave_mean = (z0_mean + z1_mean) * 0.5
    # mean, logvar
    return ave_mean, ave_var.log()  # natural log


def compute_average_ml_vae(z0_mean: torch.Tensor, z0_logvar: torch.Tensor, z1_mean: torch.Tensor, z1_logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the product of the encoder distributions.
    - Ada-ML-VAE Averaging function

    (✓) Visual inspection against reference implementation:
        https://github.com/google-research/disentanglement_lib (MLVae.model_fn)

    # TODO: recheck
    """
    # helper
    z0_var, z1_var = z0_logvar.exp(), z1_logvar.exp()
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
    return ave_mean, ave_var.log()  # natural log


COMPUTE_AVE_FNS = {
    'gvae': compute_average_gvae,
    'ml-vae': compute_average_ml_vae,
}


def compute_average(z0_mean: torch.Tensor, z0_logvar: torch.Tensor, z1_mean: torch.Tensor, z1_logvar: torch.Tensor, average_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    return COMPUTE_AVE_FNS[average_mode](z0_mean, z0_logvar, z1_mean, z1_logvar)


def compute_average_params(z0_params: 'Params', z1_params: 'Params', average_mode: str) -> 'Params':
    ave_mean, ave_logvar = compute_average(z0_params.mean, z0_params.logvar, z1_params.mean, z1_params.logvar, average_mode=average_mode)
    return z0_params.__class__(ave_mean, ave_logvar)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
