import torch
from torch import Tensor
from disent.model.base import BaseDecoderModule, BaseEncoderModule, BaseModule


# ========================================================================= #
# gaussian encoder model                                                    #
# ========================================================================= #


class AutoEncoder(BaseModule):

    def __init__(self, encoder: BaseEncoderModule, decoder: BaseDecoderModule):
        assert isinstance(encoder, BaseEncoderModule)
        assert isinstance(decoder, BaseDecoderModule)
        self.encode = encoder
        self.decode = decoder
        # check sizes
        assert encoder.z_multiplier == 1, 'z_multiplier must be 1 for encoder'
        assert encoder.x_shape == decoder.x_shape, 'x_shape mismatch'
        assert encoder.x_size == decoder.x_size, 'x_size mismatch - this should never happen if x_shape matches'
        assert encoder.z_size == decoder.z_size, 'z_size mismatch'
        # initialise
        super().__init__(x_shape=decoder.x_shape, z_size=decoder.z_size)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

# ========================================================================= #
# gaussian encoder model                                                    #
# ========================================================================= #


class GaussianAutoEncoder(BaseModule):

    def __init__(self, encoder: BaseEncoderModule, decoder: BaseDecoderModule):
        assert isinstance(encoder, BaseEncoderModule)
        assert isinstance(decoder, BaseDecoderModule)
        # check sizes
        assert encoder.z_multiplier == 2, 'z_multiplier must be 2 for gaussian encoder (encoder output is split into z_mean and z_logvar)'
        assert encoder.x_shape == decoder.x_shape, 'x_shape mismatch'
        assert encoder.x_size == decoder.x_size, 'x_size mismatch - this should never happen if x_shape matches'
        assert encoder.z_size == decoder.z_size, 'z_size mismatch'
        # initialise
        super().__init__(x_shape=decoder.x_shape, z_size=decoder.z_size)
        # assign
        self._encoder = encoder
        self._decoder = decoder


    @staticmethod
    def reparameterize(z_mean: Tensor, z_logvar: Tensor) -> Tensor:
        """
        Randomly sample for z based on the parametrization of the gaussian normal with diagonal covariance.
        This is an implementation of the 'reparameterization trick'.
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
        # TODO: cleanup
        raise RuntimeError('This has been disabled')
        # encode
        z_mean, z_logvar = self.encode_gaussian(x)
        z = self.reparameterize(z_mean, z_logvar)
        # decode
        x_recon = self.decode(z)
        return x_recon, z_mean, z_logvar, z

    def forward_deterministic(self, x: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        # TODO: cleanup
        raise RuntimeError('This has been disabled')
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
        z_params = self._encoder(x)
        z_mean, z_logvar = z_params[:, :self.z_size], z_params[:, self.z_size:]
        # check shapes
        self.assert_lengths(x, z_mean)
        self.assert_lengths(x, z_logvar)
        self.assert_z_valid(z_mean)
        self.assert_z_valid(z_logvar)
        # return
        return z_mean, z_logvar

    def encode_stochastic(self, x) -> Tensor:
        # TODO: cleanup
        raise RuntimeError('This has been disabled')
        z_mean, z_logvar = self.encode_gaussian(x)
        return self.reparameterize(z_mean, z_logvar)

    def encode_deterministic(self, x) -> Tensor:
        # TODO: cleanup
        raise RuntimeError('This has been disabled')
        z_mean, z_logvar = self.encode_gaussian(x)
        return z_mean

    def decode(self, z: Tensor) -> Tensor:
        """
        Compute the partial reconstruction of the input from a latent vector, the output is not passed
        through the final activation which can cause numerical errors if it is sigmoid.
        """
        # TODO: cleanup
        raise RuntimeError('This has been deprecated in favor of decode_partial')
        return self._decoder(z)

    def decode_partial(self, z: Tensor) -> Tensor:
        return self._decoder(z)

    def reconstruct(self, z: Tensor) -> Tensor:
        """
        Compute the full reconstruction of the input from a latent vector.
        Like decode but performs a final sigmoid activation.
        """
        # TODO: cleanup
        raise RuntimeError('This has been deprecated and should be renamed to decode which was being used')
        return torch.sigmoid(self._decoder(z))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
