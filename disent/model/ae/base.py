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
        # check sizes
        assert encoder.x_shape == decoder.x_shape, 'x_shape mismatch'
        assert encoder.x_size == decoder.x_size, 'x_size mismatch - this should never happen if x_shape matches'
        assert encoder.z_size == decoder.z_size, 'z_size mismatch'
        # initialise
        super().__init__(x_shape=decoder.x_shape, z_size=decoder.z_size, z_multiplier=encoder.z_multiplier)
        # assign
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, x):
        raise RuntimeError('This has been disabled')

    def encode(self, x):
        z_raw = self._encoder(x)
        # extract components if necessary
        if self._z_multiplier > 1:
            z_raw = tuple(
                z_raw[..., i*self.z_size:(i+1)*self.z_size]
                for i in range(self.z_multiplier)
            )
        # done
        return z_raw

    def decode(self, z: Tensor) -> Tensor:
        """
        decode the given representation.
        the returned tensor does not have an activation applied to it!
        """
        return self._decoder(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
