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

from torch import nn
from torch import Tensor

from disent.model import DisentDecoder
from disent.model import DisentEncoder


# ========================================================================= #
# disentanglement_lib Conv models                                           #
# ========================================================================= #


class EncoderConv64(DisentEncoder):
    """
    Convolutional encoder used in beta-VAE paper for the chairs data.
    Based on row 4-6 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Reference Implementation:
        - https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
        >>> def conv_encoder(input_tensor, num_latent):
        >>>     e1 = tf.layers.conv2d(inputs=input_tensor, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu, padding="same", name="e1",)
        >>>     e2 = tf.layers.conv2d(inputs=e1,           filters=32, kernel_size=4, strides=2, activation=tf.nn.relu, padding="same", name="e2",)
        >>>     e3 = tf.layers.conv2d(inputs=e2,           filters=64, kernel_size=2, strides=2, activation=tf.nn.relu, padding="same", name="e3",)  # TODO: this does not match beta-vae paper
        >>>     e4 = tf.layers.conv2d(inputs=e3,           filters=64, kernel_size=2, strides=2, activation=tf.nn.relu, padding="same", name="e4",)  # TODO: this does not match beta-vae paper
        >>>     flat_e4 = tf.layers.flatten(e4)
        >>>     e5      = tf.layers.dense(flat_e4, 256,        activation=tf.nn.relu, name="e5")
        >>>     means   = tf.layers.dense(e5,      num_latent, activation=None,       name="means")
        >>>     log_var = tf.layers.dense(e5,      num_latent, activation=None,       name="log_var")
        >>>     return means, log_var
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1):
        # checks
        (C, H, W) = x_shape
        assert (H, W) == (64, 64), 'This model only works with image size 64x64.'
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=C,  out_channels=32, kernel_size=4, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2), nn.ReLU(inplace=True),  # This was reverted to kernel size 4x4 from 2x2, to match beta-vae paper
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=2), nn.ReLU(inplace=True),  # This was reverted to kernel size 4x4 from 2x2, to match beta-vae paper
            nn.Flatten(),
            nn.Linear(in_features=1600, out_features=256), nn.ReLU(inplace=True),
            nn.Linear(in_features=256,  out_features=self.z_total),  # we combine the two networks in the reference implementation and use torch.chunk(2, dim=-1) to get mu & logvar
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


class DecoderConv64(DisentDecoder):
    """
    Convolutional decoder used in beta-VAE paper for the chairs data.
    Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Reference Implementation:
        - https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
        >>> def deconv_decoder(latent_tensor, output_shape):
        >>>     d1 = tf.layers.dense(latent_tensor,  256, activation=tf.nn.relu)
        >>>     d2 = tf.layers.dense(d1,            1024, activation=tf.nn.relu)
        >>>     d2_reshaped = tf.reshape(d2, shape=[-1, 4, 4, 64])
        >>>     d3 = tf.layers.conv2d_transpose(inputs=d2_reshaped, filters=64,              kernel_size=4, strides=2, activation=tf.nn.relu, padding="same")
        >>>     d4 = tf.layers.conv2d_transpose(inputs=d3,          filters=32,              kernel_size=4, strides=2, activation=tf.nn.relu, padding="same")
        >>>     d5 = tf.layers.conv2d_transpose(inputs=d4,          filters=32,              kernel_size=4, strides=2, activation=tf.nn.relu, padding="same")
        >>>     d6 = tf.layers.conv2d_transpose(inputs=d5,          filters=output_shape[2], kernel_size=4, strides=2,                        padding="same")
        >>>     return tf.reshape(d6, [-1] + output_shape)
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1):
        (C, H, W) = x_shape
        assert (H, W) == (64, 64), 'This model only works with image size 64x64.'
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=256),  nn.ReLU(inplace=True),
            nn.Linear(in_features=256,         out_features=1024), nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=[64, 4, 4]),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=C,  kernel_size=4, stride=2, padding=1),
        )

    def decode(self, z) -> Tensor:
        return self.model(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
