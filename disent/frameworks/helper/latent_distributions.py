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
from dataclasses import fields
from typing import Sequence
from typing import Tuple, final

import numpy as np
import torch
from torch.distributions import Normal, Distribution

from disent.frameworks.helper.reductions import loss_reduction
from disent.frameworks.helper.util import compute_ave_loss
from disent.util import TupleDataClass


# ========================================================================= #
# Helper Functions                                                          #
# ========================================================================= #


def short_dataclass_repr(self):
    vals = {
        k: v.shape if isinstance(v, (torch.Tensor, np.ndarray)) else v
        for k, v in ((f.name, getattr(self, f.name)) for f in fields(self))
    }
    return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in vals.items())})'


# ========================================================================= #
# Kl Loss                                                                   #
# ========================================================================= #


def kl_loss_direct_reverse(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # This is how the original VAE/BetaVAE papers do it.
    # - we compute the reverse kl divergence directly instead of approximating it
    # - kl(post|prior)
    # FORWARD vs. REVERSE kl (https://www.tuananhle.co.uk/notes/reverse-forward-kl.html)
    # - If we minimize the kl(post|prior) or the reverse/exclusive KL, the zero-forcing/mode-seeking behavior arises.
    # - If we minimize the kl(prior|post) or the forward/inclusive KL, the mass-covering/mean-seeking behavior arises.
    return torch.distributions.kl_divergence(posterior, prior)


def kl_loss_approx_reverse(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # This is how pytorch-lightning-bolts does it:
    # - kl(post|prior)
    # See issue: https://github.com/PyTorchLightning/pytorch-lightning-bolts/issues/565
    # - we approximate the reverse kl divergence instead of computing it analytically
    assert z_sampled is not None, 'to compute the approximate kl loss, z_sampled needs to be defined (cfg.kl_mode="approx")'
    return posterior.log_prob(z_sampled) - prior.log_prob(z_sampled)


def kl_loss_direct_forward(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # compute the forward kl
    # - kl(prior|post)
    return torch.distributions.kl_divergence(prior, posterior)


def kl_loss_approx_forward(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # compute the approximate forward kl
    # - kl(prior|post)
    assert z_sampled is not None, 'to compute the approximate kl loss, z_sampled needs to be defined (cfg.kl_mode="approx")'
    return prior.log_prob(z_sampled) - posterior.log_prob(z_sampled)


def kl_loss_direct_symmetric(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # compute the (scaled) symmetric kl
    # - 0.5 * kl(prior|post) + 0.5 * kl(prior|post)
    return 0.5 * kl_loss_direct_reverse(posterior, prior, z_sampled) + 0.5 * kl_loss_direct_forward(posterior, prior, z_sampled)


def kl_loss_approx_symmetric(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # compute the approximate (scaled) symmetric kl
    # - 0.5 * kl(prior|post) + 0.5 * kl(prior|post)
    return 0.5 * kl_loss_approx_reverse(posterior, prior, z_sampled) + 0.5 * kl_loss_approx_forward(posterior, prior, z_sampled)


_KL_LOSS_MODES = {
    # reverse kl -- how it should be done for VAEs
    'direct':         kl_loss_direct_reverse,  # alias for reverse modes
    'approx':         kl_loss_approx_reverse,  # alias for reverse modes
    'direct_reverse': kl_loss_direct_reverse,
    'approx_reverse': kl_loss_approx_reverse,
    # forward kl
    'direct_forward': kl_loss_direct_forward,
    'approx_forward': kl_loss_approx_forward,
    # symmetric kl
    'direct_symmetric': kl_loss_direct_symmetric,
    'approx_symmetric': kl_loss_approx_symmetric,
}


def kl_loss(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None, mode='direct'):
    return _KL_LOSS_MODES[mode](posterior, prior, z_sampled)


# ========================================================================= #
# Vae Distributions                                                         #
# ========================================================================= #


class LatentDistsHandler(object):

    def __init__(self,  kl_mode: str = 'direct', reduction='mean'):
        self._kl_mode = kl_mode
        self._reduction = reduction

    @dataclass
    class Params(TupleDataClass):
        """
        We use a params object so frameworks can check
        what kind of ops are supported, debug easier, and give type hints.
        - its a bit less efficient memory wise, but hardly...
        """
        __repr__ = short_dataclass_repr

    def encoding_to_params(self, z_raw):
        raise NotImplementedError

    def params_to_representation(self, z_params: Params) -> torch.Tensor:
        raise NotImplementedError

    def params_to_dists(self, z_params: Params) -> Tuple[Distribution, Distribution]:
        """
        make the posterior and prior distributions
        """
        raise NotImplementedError

    def dist_to_params(self, d_posterior: Distribution) -> Params:
        raise NotImplementedError

    @final
    def params_to_dists_and_sample(self, z_params: Params) -> Tuple[Distribution, Distribution, torch.Tensor]:
        """
        Return the parameterized prior and the approximate posterior distributions,
        as well as a sample from the approximate posterior using the 'reparameterization trick'.
        """
        posterior, prior = self.params_to_dists(z_params)
        # sample from posterior -- reparameterization trick!
        # ie. z ~ q(z|x)
        z_sampled = posterior.rsample()
        # return values
        return posterior, prior, z_sampled

    @final
    def compute_kl_loss(self, posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the kl divergence
        """
        kl = kl_loss(posterior, prior, z_sampled, mode=self._kl_mode)
        kl = loss_reduction(kl, reduction=self._reduction)
        return kl

    @final
    def compute_unreduced_kl_loss(self, posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None) -> torch.Tensor:
        return kl_loss(posterior, prior, z_sampled, mode=self._kl_mode)

    @final
    def compute_ave_kl_loss(self, ds_posterior: Sequence[Distribution], ds_prior: Sequence[Distribution], zs_sampled: Sequence[torch.Tensor]) -> torch.Tensor:
        return compute_ave_loss(self.compute_kl_loss, ds_posterior, ds_prior, zs_sampled)


# ========================================================================= #
# Normal Distribution                                                       #
# ========================================================================= #


class LatentDistsHandlerNormal(LatentDistsHandler):
    """
    Latent distributions with:
    - posterior: normal distribution with diagonal covariance
    - prior: unit normal distribution
    """

    @dataclass
    class Params(LatentDistsHandler.Params):
        mean: torch.Tensor = None
        logvar: torch.Tensor = None
        __repr__ = short_dataclass_repr

    @final
    def encoding_to_params(self, raw_z: Tuple[torch.Tensor, torch.Tensor]) -> Params:
        z_mean, z_logvar = raw_z
        return self.Params(z_mean, z_logvar)

    @final
    def params_to_representation(self, z_params: Params) -> torch.Tensor:
        return z_params.mean

    @final
    def params_to_dists(self, z_params: Params) -> Tuple[Normal, Normal]:
        """
        Return the parameterized prior and the approximate posterior distributions.
        - The standard VAE parameterizes the gaussian normal with diagonal covariance.
        - logvar is used to avoid negative values for the standard deviation
        - Gaussian Encoder Model Distribution: pg. 25 in Variational Auto Encoders

        (✓) Visual inspection against reference implementations:
            https://github.com/google-research/disentanglement_lib (sample_from_latent_distribution)
            https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
        """
        z_mean, z_logvar = z_params
        # compute required values
        z_std = torch.exp(0.5 * z_logvar)
        # q: approximate posterior distribution
        posterior = Normal(z_mean, z_std)
        # p: prior distribution
        prior = Normal(torch.zeros_like(z_mean), torch.ones_like(z_std))
        # return values
        return posterior, prior

    @final
    def dist_to_params(self, d_posterior: Normal) -> Params:
        return self.Params(
            mean=d_posterior.mean,
            logvar=d_posterior.variance.log()
        )

    @staticmethod
    def LEGACY_compute_kl_loss(mu, logvar, mode: str = 'direct', reduction='mean_sum'):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.
        FROM: https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (compute_gaussian_kl)
        """
        assert mode == 'direct', f'legacy reference implementation of KL loss only supports mode="direct", not {repr(mode)}'
        assert reduction == 'mean_sum', f'legacy reference implementation of KL loss only supports reduction="mean_sum", not {repr(reduction)}'
        # Calculate KL divergence
        kl_values = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # Sum KL divergence across latent vector for each sample
        kl_sums = torch.sum(kl_values, dim=1)
        # KL loss is mean of the KL divergence sums
        kl_loss = torch.mean(kl_sums)
        return kl_loss


# ========================================================================= #
# Factory                                                                   #
# ========================================================================= #


def make_latent_distribution(name: str, kl_mode: str, reduction: str) -> LatentDistsHandler:
    if name == 'normal':
        cls = LatentDistsHandlerNormal
    else:
        raise KeyError(f'unknown vae distribution name: {name}')
    # make instance
    return cls(kl_mode=kl_mode, reduction=reduction)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
