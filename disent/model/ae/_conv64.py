from torch import nn as nn, Tensor

from disent.model.base import BaseEncoderModule, BaseDecoderModule
from disent.model.common import Flatten3D, BatchView


# ========================================================================= #
# disentanglement_lib Conv models                                           #
# ========================================================================= #


class EncoderConv64(BaseEncoderModule):
    """
    From:
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
    # TODO: verify, things have changed...
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1, dropout=0.0):
        """
        Convolutional encoder used in beta-VAE paper for the chairs data.
        Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
        Concepts with a Constrained Variational BaseFramework"
        (https://openreview.net/forum?id=Sy2fzU9gl)
        """
        # checks
        assert tuple(x_shape[1:]) == (64, 64), 'This model only works with image size 64x64.'
        num_channels = x_shape[0]
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=4, stride=2, padding=2),
                nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2),
                nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=1),
                nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=1),
                nn.LeakyReLU(inplace=True),
            Flatten3D(),
            nn.Linear(1600, 256),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout),
            nn.Linear(256, self.z_total),
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


class DecoderConv64(BaseDecoderModule):
    """
    From:
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
    # TODO: verify, things have changed...
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1, dropout=0.0):
        """
        Convolutional decoder used in beta-VAE paper for the chairs data.
        Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
        Concepts with a Constrained Variational BaseFramework"
        (https://openreview.net/forum?id=Sy2fzU9gl)
        """
        assert tuple(x_shape[1:]) == (64, 64), 'This model only works with image size 64x64.'
        num_channels = x_shape[0]
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        self.model = nn.Sequential(
            nn.Linear(self.z_size, 256),
                nn.Dropout(dropout),
                nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1024),
                nn.LeakyReLU(inplace=True),
            BatchView([64, 4, 4]),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=num_channels, kernel_size=4, stride=2, padding=1),
        )

    def decode(self, z) -> Tensor:
        return self.model(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
