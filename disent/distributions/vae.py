from typing import Tuple, NamedTuple, Type

import torch
from torch.distributions import Normal, Distribution


# ========================================================================= #
# Vae Distributions                                                         #
# ========================================================================= #


class VaeDistribution(object):

    class Params:
        __slots__ = []

        def __init__(self, *args):
            assert len(args) == len(self.__slots__)
            for name, arg in zip(self.__slots__, args):
                setattr(self, name, arg)

        def __iter__(self):
            for name in self.__slots__:
                yield getattr(self, name)

    def raw_to_params(self, z: Tuple[torch.Tensor, ...]) -> Params:
        """
        Convert the raw z encoding to a params object
        for this specific set of distributions.
        """
        raise NotImplementedError

    def params_to_z(self, params: Params) -> torch.Tensor:
        """
        Get the deterministic z values to pass to the encoder.
        """
        return NotImplementedError

    def make(self, z_params: Params) -> Tuple[Distribution, Distribution]:
        """
        make the prior and posterior distributions
        """
        raise NotImplementedError

    def make_and_sample(self, z_params: Params) -> Tuple[Tuple[Distribution, Distribution], torch.Tensor]:
        """
        Return the parameterized prior and the approximate posterior distributions,
        as well as a sample from the approximate posterior using the 'reparameterization trick'.
        """
        posterior, prior = self.make(z_params)
        # sample from posterior -- reparameterization trick!
        # ie. z ~ q(z|x)
        z_sampled = posterior.rsample()
        # return values
        return (posterior, prior), z_sampled


class VaeDistributionNormal(VaeDistribution):

    class Params(VaeDistribution.Params):
        __slots__ = ['z_mean', 'z_logvar']

        def __init__(self, z_mean, z_logvar):
            super().__init__(z_mean, z_logvar)

    def raw_to_params(self, raw_z: Tuple[torch.Tensor, torch.Tensor]) -> Params:
        z_mean, z_logvar = raw_z
        return self.Params(z_mean, z_logvar)

    def params_to_z(self, params: Params) -> torch.Tensor:
        return params.z_mean

    def make(self, z_params: Params) -> Tuple[Normal, Normal]:
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
        # p: prior distribution
        prior = Normal(torch.zeros_like(z_mean), torch.ones_like(z_std))
        # q: approximate posterior distribution
        posterior = Normal(z_mean, z_std)
        # return values
        return posterior, prior


def make_vae_distribution(name: str) -> VaeDistribution:
    if name == 'normal':
        return VaeDistributionNormal()
    else:
        raise KeyError(f'unknown vae distribution name: {name}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

