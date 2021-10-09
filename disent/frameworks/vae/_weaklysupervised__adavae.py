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
from torch.distributions import kl_divergence
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

    * This class is a little over complicated because it has the added functionality
      listed above, and because we want to re-use features elsewhere. The code can
      be compressed down into about ~20 neat lines for `hook_intercept_ds` if we
      select and chose fixed `cfg` values.
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
        share_mask = self.compute_shared_mask_from_posteriors(d0_posterior, d1_posterior, thresh_mode=self.cfg.ada_thresh_mode, ratio=self.cfg.ada_thresh_ratio)
        # compute average posteriors
        new_ds_posterior = self.make_shared_posteriors(d0_posterior, d1_posterior, share_mask, average_mode=self.cfg.ada_average_mode)
        # return new args & generate logs
        return new_ds_posterior, ds_prior, {
            'shared': share_mask.sum(dim=1).float().mean()
        }

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # HELPER - POSTERIORS                                                   #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @classmethod
    def compute_deltas_from_posteriors(cls, d0_posterior: Distribution, d1_posterior: Distribution, thresh_mode: str):
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
            deltas = kl_divergence(d1_posterior, d0_posterior)
        elif thresh_mode == 'symmetric_kl':
            # FROM: https://openreview.net/pdf?id=8VXvj1QNRl1
            kl_deltas_d1_d0 = kl_divergence(d1_posterior, d0_posterior)
            kl_deltas_d0_d1 = kl_divergence(d0_posterior, d1_posterior)
            deltas = (0.5 * kl_deltas_d1_d0) + (0.5 * kl_deltas_d0_d1)
        elif thresh_mode == 'dist':
            deltas = cls.compute_deltas_from_zs(d1_posterior.mean, d0_posterior.mean)
        elif thresh_mode == 'sampled_dist':
            deltas = cls.compute_deltas_from_zs(d1_posterior.rsample(), d0_posterior.rsample())
        else:
            raise KeyError(f'invalid thresh_mode: {repr(thresh_mode)}')

        # return values
        return deltas

    @classmethod
    def compute_shared_mask_from_posteriors(cls, d0_posterior: Distribution, d1_posterior: Distribution, thresh_mode: str, ratio=0.5):
        return cls.estimate_shared_mask(z_deltas=cls.compute_deltas_from_posteriors(d0_posterior, d1_posterior, thresh_mode=thresh_mode), ratio=ratio)

    @classmethod
    def make_shared_posteriors(cls, d0_posterior: Normal, d1_posterior: Normal, share_mask: torch.Tensor, average_mode: str) -> Tuple[Normal, Normal]:
        # compute average posterior
        ave_posterior = AdaVae.compute_average_distribution(d0_posterior=d0_posterior, d1_posterior=d1_posterior, average_mode=average_mode)
        # select shared elements
        ave_d0_posterior = Normal(loc=torch.where(share_mask, ave_posterior.loc, d0_posterior.loc), scale=torch.where(share_mask, ave_posterior.scale, d0_posterior.scale))
        ave_d1_posterior = Normal(loc=torch.where(share_mask, ave_posterior.loc, d1_posterior.loc), scale=torch.where(share_mask, ave_posterior.scale, d1_posterior.scale))
        # return values
        return ave_d0_posterior, ave_d1_posterior

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # HELPER - MEAN/MU VALUES (same functionality as posterior versions)    #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @classmethod
    def compute_deltas_from_zs(cls, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        return torch.abs(z0 - z1)

    @classmethod
    def compute_shared_mask_from_zs(cls, z0: torch.Tensor, z1: torch.Tensor, ratio: float = 0.5):
        return cls.estimate_shared_mask(z_deltas=cls.compute_deltas_from_zs(z0, z1), ratio=ratio)

    @classmethod
    def make_shared_zs(cls, z0: torch.Tensor, z1: torch.Tensor, share_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute average values
        ave = 0.5 * z0 + 0.5 * z1
        # select shared elements
        ave_z0 = torch.where(share_mask, ave, z0)
        ave_z1 = torch.where(share_mask, ave, z1)
        return ave_z0, ave_z1

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # HELPER - COMMON                                                       #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @classmethod
    def estimate_shared_mask(cls, z_deltas: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Core of the adaptive VAE algorithm, estimating which factors
        have changed (or in this case which are shared and should remained unchanged
        by being be averaged) between pairs of observations.
        - custom ratio is an addition, when ratio==0.5 then
          this is equivalent to the original implementation.

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
        assert 0 <= ratio <= 1, f'ratio must be in the range: 0 <= ratio <= 1, got: {repr(ratio)}'
        # threshold τ
        maximums = z_deltas.max(axis=1, keepdim=True).values      # (B, 1)
        minimums = z_deltas.min(axis=1, keepdim=True).values      # (B, 1)
        z_threshs = torch.lerp(minimums, maximums, weight=ratio)  # (B, 1)
        # true if 'unchanged' and should be average
        shared_mask = z_deltas < z_threshs                        # broadcast (B, Z) and (B, 1) -> (B, Z)
        # return
        return shared_mask

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # HELPER - DISTRIBUTIONS                                                #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @classmethod
    def compute_average_distribution(cls, d0_posterior: Normal, d1_posterior: Normal, average_mode: str) -> Normal:
        return _COMPUTE_AVE_FNS[average_mode](
            d0_posterior=d0_posterior,
            d1_posterior=d1_posterior,
        )


# ========================================================================= #
# Averaging Functions                                                       #
# ========================================================================= #


def compute_average_gvae_std(d0_posterior: Normal, d1_posterior: Normal) -> Normal:
    """
    Compute the arithmetic mean of the encoder distributions.
    - This is a custom function based on the Ada-GVAE averaging,
      except over the standard deviation instead of the variance!

    *NB* this is un-official!
    """
    assert isinstance(d0_posterior, Normal), f'posterior distributions must be {Normal.__name__} distributions, got: {type(d0_posterior)}'
    assert isinstance(d1_posterior, Normal), f'posterior distributions must be {Normal.__name__} distributions, got: {type(d1_posterior)}'
    # averages
    ave_std = 0.5 * (d0_posterior.stddev + d1_posterior.stddev)
    ave_mean = 0.5 * (d1_posterior.mean + d1_posterior.mean)
    # done!
    return Normal(loc=ave_mean, scale=ave_std)


def compute_average_gvae(d0_posterior: Normal, d1_posterior: Normal) -> Normal:
    """
    Compute the arithmetic mean of the encoder distributions.
    - Ada-GVAE Averaging function

    (✓) Visual inspection against reference implementation:
        https://github.com/google-research/disentanglement_lib (GroupVAEBase.model_fn)
    """
    assert isinstance(d0_posterior, Normal), f'posterior distributions must be {Normal.__name__} distributions, got: {type(d0_posterior)}'
    assert isinstance(d1_posterior, Normal), f'posterior distributions must be {Normal.__name__} distributions, got: {type(d1_posterior)}'
    # averages
    ave_var = 0.5 * (d0_posterior.variance + d1_posterior.variance)
    ave_mean = 0.5 * (d1_posterior.mean + d1_posterior.mean)
    # done!
    return Normal(loc=ave_mean, scale=torch.sqrt(ave_var))


def compute_average_ml_vae(d0_posterior: Normal, d1_posterior: Normal) -> Normal:
    """
    Compute the product of the encoder distributions.
    - Ada-ML-VAE Averaging function

    (✓) Visual inspection against reference implementation:
        https://github.com/google-research/disentanglement_lib (MLVae.model_fn)

    # TODO: recheck
    """
    assert isinstance(d0_posterior, Normal), f'posterior distributions must be {Normal.__name__} distributions, got: {type(d0_posterior)}'
    assert isinstance(d1_posterior, Normal), f'posterior distributions must be {Normal.__name__} distributions, got: {type(d1_posterior)}'
    # Diagonal matrix inverse: E^-1 = 1 / E
    # https://proofwiki.org/wiki/Inverse_of_Diagonal_Matrix
    z0_invvar, z1_invvar = d0_posterior.variance.reciprocal(), d1_posterior.variance.reciprocal()
    # average var: E^-1 = E1^-1 + E2^-1
    # disentanglement_lib: ave_var = 2 * z0_var * z1_var / (z0_var + z1_var)
    ave_var = 2 * (z0_invvar + z1_invvar).reciprocal()
    # average mean: u^T = (u1^T E1^-1 + u2^T E2^-1) E
    # disentanglement_lib: ave_mean = (z0_mean/z0_var + z1_mean/z1_var) * ave_var * 0.5
    ave_mean = (d0_posterior.mean*z0_invvar + d1_posterior.mean*z1_invvar) * ave_var * 0.5
    # done!
    return Normal(loc=ave_mean, scale=torch.sqrt(ave_var))


_COMPUTE_AVE_FNS = {
    'gvae': compute_average_gvae,
    'ml-vae': compute_average_ml_vae,
    'gvae_std': compute_average_gvae_std,  # this is un-official!
}


# ========================================================================= #
# Ada-GVAE                                                                  #
# ========================================================================= #


class AdaGVaeMinimal(BetaVae):
    """
    This is a direct implementation of the Ada-GVAE,
    which should be equivalent to the AdaVae with config values:

    >>> AdaVae.cfg(
    >>>    ada_average_mode='gvae',
    >>>    ada_thresh_mode='symmetric_kl',
    >>>    ada_thresh_ratio=0.5,
    >>> )
    """

    REQUIRED_OBS = 2

    def hook_intercept_ds(self, ds_posterior: Sequence[Distribution], ds_prior: Sequence[Distribution]) -> Tuple[Sequence[Distribution], Sequence[Distribution], Dict[str, Any]]:
        """
        Adaptive VAE Method, putting the various components together
            1. compute differences between representations
            2. estimate a threshold for differences
            3. compute a shared mask from this threshold
            4. average together elements that are marked as shared

        (x) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (aggregate_argmax)
        """
        d0_posterior, d1_posterior = ds_posterior
        assert isinstance(d0_posterior, Normal), f'posterior distributions must be {Normal.__name__} distributions, got: {type(d0_posterior)}'
        assert isinstance(d1_posterior, Normal), f'posterior distributions must be {Normal.__name__} distributions, got: {type(d1_posterior)}'

        # [1] symmetric KL Divergence FROM: https://openreview.net/pdf?id=8VXvj1QNRl1
        z_deltas = 0.5 * kl_divergence(d1_posterior, d0_posterior) + 0.5 * kl_divergence(d0_posterior, d1_posterior)

        # [2] estimate threshold from deltas
        z_deltas_min = z_deltas.min(axis=1, keepdim=True).values  # (B, 1)
        z_deltas_max = z_deltas.max(axis=1, keepdim=True).values  # (B, 1)
        z_thresh     = (0.5 * z_deltas_min + 0.5 * z_deltas_max)  # (B, 1)

        # [3] shared elements that need to be averaged, computed per pair in the batch
        share_mask = z_deltas < z_thresh  # broadcast (B, Z) and (B, 1) to get (B, Z)

        # [4.a] compute average representations
        # - this is the only difference between the Ada-ML-VAE
        ave_mean = (0.5 * d0_posterior.mean     + 0.5 * d1_posterior.mean)
        ave_std  = (0.5 * d0_posterior.variance + 0.5 * d1_posterior.variance) ** 0.5

        # [4.b] select shared or original values based on mask
        z0_mean = torch.where(share_mask,  d0_posterior.loc,   ave_mean)
        z1_mean = torch.where(share_mask,  d1_posterior.loc,   ave_mean)
        z0_std  = torch.where(share_mask,  d0_posterior.scale, ave_std)
        z1_std  = torch.where(share_mask,  d1_posterior.scale, ave_std)

        # construct distributions
        ave_d0_posterior = Normal(loc=z0_mean, scale=z0_std)
        ave_d1_posterior = Normal(loc=z1_mean, scale=z1_std)
        new_ds_posterior = (ave_d0_posterior, ave_d1_posterior)

        # [done] return new args & generate logs
        return new_ds_posterior, ds_prior, {
            'shared': share_mask.sum(dim=1).float().mean()
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
