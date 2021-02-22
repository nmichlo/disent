from dataclasses import dataclass
from typing import Tuple, final

import torch
from torch.distributions import Normal, Distribution

from disent.frameworks.helper.reductions import loss_reduction
from disent.util import TupleDataClass


# ========================================================================= #
# Helper Functions                                                          #
# ========================================================================= #


def kl_loss_direct(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # This is how the original VAE/BetaVAE papers do it:s
    # - we compute the kl divergence directly instead of approximating it
    return torch.distributions.kl_divergence(posterior, prior)


def kl_loss_approx(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # This is how pytorch-lightning-bolts does it:
    # See issue: https://github.com/PyTorchLightning/pytorch-lightning-bolts/issues/565
    # - we approximate the kl divergence instead of computing it analytically
    assert z_sampled is not None, 'to compute the approximate kl loss, z_sampled needs to be defined (cfg.kl_mode="approx")'
    return posterior.log_prob(z_sampled) - prior.log_prob(z_sampled)


_KL_LOSS_MODES = {
    'direct': kl_loss_direct,
    'approx': kl_loss_approx,
}


def kl_loss(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None, mode='direct'):
    return _KL_LOSS_MODES[mode](posterior, prior, z_sampled)


# ========================================================================= #
# Vae Distributions                                                         #
# ========================================================================= #


class LatentDistribution(object):

    @dataclass
    class Params(TupleDataClass):
        """
        We use a params object so frameworks can check
        what kind of ops are supported, debug easier, and give type hints.
        - its a bit less efficient memory wise, but hardly...
        """

    def encoding_to_params(self, z_raw):
        raise NotImplementedError

    def params_to_representation(self, z_params: Params) -> torch.Tensor:
        raise NotImplementedError

    def params_to_distributions(self, z_params: Params) -> Tuple[Distribution, Distribution]:
        """
        make the posterior and prior distributions
        """
        raise NotImplementedError

    @final
    def params_to_distributions_and_sample(self, z_params: Params) -> Tuple[Tuple[Distribution, Distribution], torch.Tensor]:
        """
        Return the parameterized prior and the approximate posterior distributions,
        as well as a sample from the approximate posterior using the 'reparameterization trick'.
        """
        posterior, prior = self.params_to_distributions(z_params)
        # sample from posterior -- reparameterization trick!
        # ie. z ~ q(z|x)
        z_sampled = posterior.rsample()
        # return values
        return (posterior, prior), z_sampled

    @classmethod
    def compute_kl_loss(
            cls,
            posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None,
            mode: str = 'direct', reduction='batch_mean'
    ):
        """
        Compute the kl divergence
        """
        kl = kl_loss(posterior, prior, z_sampled, mode=mode)
        kl = loss_reduction(kl, reduction=reduction)
        return kl


# ========================================================================= #
# Normal Distribution                                                       #
# ========================================================================= #


class LatentDistributionNormal(LatentDistribution):
    """
    Latent distributions with:
    - posterior: normal distribution with diagonal covariance
    - prior: unit normal distribution
    """

    @dataclass
    class Params(LatentDistribution.Params):
        mean: torch.Tensor = None
        logvar: torch.Tensor = None

    @final
    def encoding_to_params(self, raw_z: Tuple[torch.Tensor, torch.Tensor]) -> Params:
        z_mean, z_logvar = raw_z
        return self.Params(z_mean, z_logvar)

    @final
    def params_to_representation(self, z_params: Params) -> torch.Tensor:
        return z_params.mean

    @final
    def params_to_distributions(self, z_params: Params) -> Tuple[Normal, Normal]:
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

    @staticmethod
    def LEGACY_compute_kl_loss(mu, logvar, mode: str = 'direct', reduction='batch_mean'):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.
        FROM: https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (compute_gaussian_kl)
        """
        assert mode == 'direct', f'legacy reference implementation of KL loss only supports mode="direct", not {repr(mode)}'
        assert reduction == 'batch_mean', f'legacy reference implementation of KL loss only supports reduction="batch_mean", not {repr(reduction)}'
        # Calculate KL divergence
        kl_values = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # Sum KL divergence across latent vector for each sample
        kl_sums = torch.sum(kl_values, dim=1)
        # KL loss is mean of the KL divergence sums
        kl_loss = torch.mean(kl_sums)
        return kl_loss

    # @classmethod
    # def compute_kl_loss(
    #         cls,
    #         posterior: Normal, prior: Normal, z_sampled: torch.Tensor = None,
    #         mode: str = 'direct', reduction='batch_mean'
    # ):
    #     # check dtypes
    #     dtype = posterior.loc.dtype
    #     assert dtype == torch.float32
    #     assert dtype == posterior.loc.dtype == posterior.scale.dtype == prior.loc.dtype == prior.scale.dtype
    #     # convert to float64
    #     posterior = Normal(posterior.loc.to(torch.float64), posterior.scale.to(torch.float64))
    #     prior = Normal(prior.loc.to(torch.float64), prior.scale.to(torch.float64))
    #     if z_sampled is not None:
    #         assert dtype == z_sampled.dtype
    #         z_sampled = z_sampled.to(torch.float64)
    #     # compute kl like normal
    #     kl = super().compute_kl_loss(posterior, prior, z_sampled, mode=mode, reduction=reduction)
    #     # convert back to float32
    #     return kl.to(torch.float32)




# ========================================================================= #
# Factory                                                                   #
# ========================================================================= #


def make_latent_distribution(name: str) -> LatentDistribution:
    if name == 'normal':
        return LatentDistributionNormal()
    else:
        raise KeyError(f'unknown vae distribution name: {name}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
