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

import numpy as np
from torch import nn
from torch import Tensor

from disent.model import DisentDecoder
from disent.model import DisentEncoder


# ========================================================================= #
# disentanglement_lib FC models                                             #
# ========================================================================= #


class EncoderFC(DisentEncoder):
    """
    Fully connected encoder used in beta-VAE paper for the dSprites data.
    Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Reference Implementation:
        - https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
        >>> def fc_encoder(input_tensor, num_latent):
        >>>     flattened = tf.layers.flatten(input_tensor)
        >>>     e1 = tf.layers.dense(flattened, 1200, activation=tf.nn.relu, name="e1")
        >>>     e2 = tf.layers.dense(e1,        1200, activation=tf.nn.relu, name="e2")
        >>>     means   = tf.layers.dense(e2, num_latent, activation=None)
        >>>     log_var = tf.layers.dense(e2, num_latent, activation=None)
        >>>     return means, log_var
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1):
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=int(np.prod(x_shape)), out_features=1200), nn.ReLU(True),
            nn.Linear(in_features=1200,                  out_features=1200), nn.ReLU(True),
            nn.Linear(in_features=1200,                  out_features=self.z_total)
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


class DecoderFC(DisentDecoder):
    """
    Fully connected encoder used in beta-VAE paper for the dSprites data.
    Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Reference Implementation:
        - https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
        >>> def fc_decoder(latent_tensor, output_shape):
        >>>     d1 = tf.layers.dense(latent_tensor, 1200, activation=tf.nn.tanh)
        >>>     d2 = tf.layers.dense(d1,            1200, activation=tf.nn.tanh)
        >>>     d3 = tf.layers.dense(d2,            1200, activation=tf.nn.tanh)
        >>>     d4 = tf.layers.dense(d3, np.prod(output_shape))
        >>>     return tf.reshape(d4, shape=[-1] + output_shape)
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1):
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=1200), nn.Tanh(),
            nn.Linear(in_features=1200,        out_features=1200), nn.Tanh(),
            nn.Linear(in_features=1200,        out_features=1200), nn.Tanh(),
            nn.Linear(in_features=1200,        out_features=int(np.prod(self.x_shape))),
            nn.Unflatten(dim=1, unflattened_size=self.x_shape),
        )

    def decode(self, z) -> Tensor:
        return self.model(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
