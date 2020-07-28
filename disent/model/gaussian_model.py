import torch
from torch import Tensor
from disent.model.base import BaseDecoderModule, BaseGaussianEncoderModule


# ========================================================================= #
# gaussian encoder model                                                    #
# ========================================================================= #


class GaussianEncoderDecoderModel(BaseGaussianEncoderModule, BaseDecoderModule):

    def __init__(self, gaussian_encoder: BaseGaussianEncoderModule, decoder: BaseDecoderModule):
        assert gaussian_encoder.x_shape == decoder.x_shape, 'x_shape mismatch'
        assert gaussian_encoder.x_size == decoder.x_size, 'x_size mismatch - this should never happen if x_shape matches'
        assert gaussian_encoder.z_size == decoder.z_size, 'z_size mismatch'
        super().__init__(x_shape=gaussian_encoder.x_shape, z_size=gaussian_encoder.z_size)
        self._gaussian_encoder = gaussian_encoder
        self._decoder = decoder

    @staticmethod
    def sample_from_latent_distribution(z_mean: Tensor, z_logvar: Tensor) -> Tensor:
        """
        Randomly sample for z based on the parametrization of the gaussian normal with diagonal covariance.
        This is an implementation of the 'reparametrisation trick'.
        ie. z ~ p(z|x)
        Gaussian Encoder Model Distribution - pg. 25 in Variational Auto Encoders
        """
        std = torch.exp(0.5 * z_logvar)  # std == var^0.5 == e^(log(var^0.5)) == e^(0.5*log(var))
        eps = torch.randn_like(std)      # N(0, 1)
        return z_mean + (std * eps)      # mu + dot(std, eps)

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        """
        reconstruct the input:
        x -> encode  | z_mean, z_var
          -> z       | z ~ p(z|x)
          -> decode  |
          -> x_recon | no final activation
        """
        # encode
        z_mean, z_logvar = self.encode_gaussian(x)
        z = self.sample_from_latent_distribution(z_mean, z_logvar)
        # decode
        x_recon = self.decode(z)
        return x_recon, z_mean, z_logvar, z

    def forward_deterministic(self, x: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        # encode
        z_mean, z_logvar = self.encode_gaussian(x)
        z = z_mean
        # decode
        x_recon = self.decode(z)
        return x_recon, z_mean, z_logvar, z

    def encode_gaussian(self, x: Tensor) -> (Tensor, Tensor):
        """
        Compute the mean and logvar parametrisation for the gaussian
        normal distribution with diagonal covariance ie. the parametrisation for p(z|x).
        """
        z_mean, z_logvar = self._gaussian_encoder(x)
        return z_mean, z_logvar

    def encode_stochastic(self, x) -> Tensor:
        z_mean, z_logvar = self.encode_gaussian(x)
        return self.sample_from_latent_distribution(z_mean, z_logvar)

    def encode_deterministic(self, x) -> Tensor:
        z_mean, z_logvar = self.encode_gaussian(x)
        return z_mean

    def decode(self, z: Tensor) -> Tensor:
        """
        Compute the full reconstruction of the input from a latent vector, passing the
        decoder through a final activation. This is not numerically stable and should be
        removed in favour of activation in the loss functions

        TODO: Compute the partial reconstruction of the input from a latent vector.
              The final activation should not be included. This will always be sigmoid
              and is computed as part of the loss to improve numerical stability.
        """
        return self._decoder(z)

    def reconstruct(self, z: Tensor) -> Tensor:
        """
        Compute the full reconstruction of the input from a latent vector.
        Like decode but performs a final sigmoid activation.
        """
        return torch.sigmoid(self._decoder(z))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
