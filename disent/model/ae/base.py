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

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        """
        reconstruct the input:
        x -> encode  | z_mean, z_var
          -> z       | z ~ p(z|x)
          -> decode  |
          -> x_recon | no final activation
        """
        raise RuntimeError('This has been disabled')

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

    def decode_partial(self, z: Tensor) -> Tensor:
        return self._decoder(z)

    def decode(self, z: Tensor) -> Tensor:
        """
        Compute the full reconstruction of the input from a latent vector.
        Like decode but performs a final sigmoid activation.
        """
        return torch.sigmoid(self._decoder(z))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
