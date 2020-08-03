import gin
import torch.nn as nn
import numpy as np
from torch import Tensor
from disent.model.base import BaseEncoderModule, BaseDecoderModule, Flatten3D, Unsqueeze3D, BatchView


# ========================================================================= #
# simple 28x28 fully connected models                                       #
# ========================================================================= #


@gin.configurable('model.encoder.SimpleFC')
class EncoderSimpleFC(BaseEncoderModule):

    def __init__(self, x_shape=(3, 64, 64), h_size1=128, h_size2=128, z_size=6):
        super().__init__(x_shape=x_shape, z_size=z_size)
        self.model = nn.Sequential(
            Flatten3D(),
            nn.Linear(self.x_size, h_size1),
                nn.ReLU(True),
            nn.Linear(h_size1, h_size2),
                nn.ReLU(True),
            nn.Linear(h_size2, self.z_size)
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


@gin.configurable('model.decoder.SimpleFC')
class DecoderSimpleFC(BaseDecoderModule):

    def __init__(self, x_shape=(3, 64, 64), h_size1=128, h_size2=128, z_size=6):
        super().__init__(x_shape=x_shape, z_size=z_size)
        self.model = nn.Sequential(
            nn.Linear(self.z_size, h_size2),
                nn.ReLU(True),
            nn.Linear(h_size2, h_size1),
                nn.ReLU(True),
            nn.Linear(h_size1, self.x_size),
                BatchView(self.x_shape),
        )

    def decode(self, z):
        return self.model(z)


# ========================================================================= #
# simple 64x64 convolutional models                                         #
# ========================================================================= #


@gin.configurable('model.encoder.SimpleConv64')
class EncoderSimpleConv64(BaseEncoderModule):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""

    def __init__(self, x_shape=(3, 64, 64), z_size=6):
        # checks
        assert tuple(x_shape[1:]) == (64, 64), 'This model only works with image size 64x64.'
        num_channels = x_shape[0]
        super().__init__(x_shape=x_shape, z_size=z_size)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                Flatten3D(),
            nn.Linear(256, self.z_size)
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


@gin.configurable('model.decoder.SimpleConv64')
class DecoderSimpleConv64(BaseDecoderModule):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""

    def __init__(self, x_shape=(3, 64, 64), z_size=6):
        # checks
        assert tuple(x_shape[1:]) == (64, 64), 'This model only works with image size 64x64.'
        num_channels = x_shape[0]
        super().__init__(x_shape=x_shape, z_size=z_size)

        self.model = nn.Sequential(
            Unsqueeze3D(),
            nn.Conv2d(in_channels=self.z_size, out_channels=256, kernel_size=1, stride=2),
                nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2),
                nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2),
                nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=num_channels, kernel_size=3, stride=1)
        )
        # output shape = bs x 3 x 64 x 64

    def decode(self, x):
        return self.model(x)


# ========================================================================= #
# disentanglement_lib FC models                                             #
# ========================================================================= #


@gin.configurable('model.encoder.FC')
class EncoderFC(BaseEncoderModule):
    """
    From:
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6):
        """
        Fully connected encoder used in beta-VAE paper for the dSprites data.
        Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
        Concepts with a Constrained Variational BaseFramework"
        (https://openreview.net/forum?id=Sy2fzU9gl).
        """
        # checks
        super().__init__(x_shape=x_shape, z_size=z_size)

        self.model = nn.Sequential(
            Flatten3D(),
            nn.Linear(int(np.prod(x_shape)), 1200),
                nn.ReLU(True),
            nn.Linear(1200, 1200),
                nn.ReLU(True),
            nn.Linear(1200, self.z_size)
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


@gin.configurable('model.decoder.FC')
class DecoderFC(BaseDecoderModule):
    """
    From:
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6):
        """
        Fully connected encoder used in beta-VAE paper for the dSprites data.
        Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
        Concepts with a Constrained Variational BaseFramework"
        (https://openreview.net/forum?id=Sy2fzU9gl)
        """
        super().__init__(x_shape=x_shape, z_size=z_size)

        self.model = nn.Sequential(
            nn.Linear(self.z_size, 1200),
                nn.Tanh(),
            nn.Linear(1200, 1200),
                nn.Tanh(),
            nn.Linear(1200, 1200),
                nn.Tanh(),
            nn.Linear(1200, int(np.prod(x_shape))),
                BatchView(self.x_shape),
        )

    def decode(self, z) -> Tensor:
        return self.model(z)


# ========================================================================= #
# disentanglement_lib Conv models                                           #
# ========================================================================= #


@gin.configurable('model.encoder.Conv64')
class EncoderConv64(BaseEncoderModule):
    """
    From:
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, dropout=0.0):
        """
        Convolutional encoder used in beta-VAE paper for the chairs data.
        Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
        Concepts with a Constrained Variational BaseFramework"
        (https://openreview.net/forum?id=Sy2fzU9gl)
        """
        # checks
        assert tuple(x_shape[1:]) == (64, 64), 'This model only works with image size 64x64.'
        num_channels = x_shape[0]
        super().__init__(x_shape=x_shape, z_size=z_size)

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
            nn.Linear(256, self.z_size),
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


@gin.configurable('model.decoder.Conv64')
class DecoderConv64(BaseDecoderModule):
    """
    From:
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, dropout=0.0):
        """
        Convolutional decoder used in beta-VAE paper for the chairs data.
        Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
        Concepts with a Constrained Variational BaseFramework"
        (https://openreview.net/forum?id=Sy2fzU9gl)
        """
        assert tuple(x_shape[1:]) == (64, 64), 'This model only works with image size 64x64.'
        num_channels = x_shape[0]
        super().__init__(x_shape=x_shape, z_size=z_size)

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
