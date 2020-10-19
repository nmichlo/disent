import numpy as np
from torch import nn as nn, Tensor

from disent.model.base import BaseEncoderModule, BaseDecoderModule
from disent.model.common import Flatten3D, BatchView


# ========================================================================= #
# disentanglement_lib FC models                                             #
# ========================================================================= #


class EncoderFC(BaseEncoderModule):
    """
    From:
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
    # TODO: verify, things have changed...
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1):
        """
        Fully connected encoder used in beta-VAE paper for the dSprites data.
        Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
        Concepts with a Constrained Variational BaseFramework"
        (https://openreview.net/forum?id=Sy2fzU9gl).
        """
        # checks
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        self.model = nn.Sequential(
            Flatten3D(),
            nn.Linear(int(np.prod(x_shape)), 1200),
                nn.ReLU(True),
            nn.Linear(1200, 1200),
                nn.ReLU(True),
            nn.Linear(1200, self.z_total)
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


class DecoderFC(BaseDecoderModule):
    """
    From:
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
    # TODO: verify, things have changed...
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1):
        """
        Fully connected encoder used in beta-VAE paper for the dSprites data.
        Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
        Concepts with a Constrained Variational BaseFramework"
        (https://openreview.net/forum?id=Sy2fzU9gl)
        """
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

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
# END                                                                       #
# ========================================================================= #
