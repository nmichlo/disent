import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================= #
# Mish Activation Function TODO: move out                                   #
# ========================================================================= #


# @torch.jit.script
# def mish(input):
#     """
#     Mish Activation Function:
#     https://github.com/lessw2020/mish/blob/master/mish.py
#     https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/functional.py
#     """
#     return input * torch.tanh(F.softplus(input))


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


class View(nn.Module):
    """From: https://github.com/1Konny/Beta-VAE/blob/master/model.py"""
    def __init__(self, size):
        super().__init__()
        self.size = size
    def forward(self, tensor):
        return tensor.view(self.size)

class Unsqueeze3D(nn.Module):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x

class Flatten3D(nn.Module):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


# ========================================================================= #
# VAE                                                                       #
# ========================================================================= #


class EncoderSimpleFC(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim, h_dim1),
            nn.ReLU(True),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(True),
        )
        self.enc3mean = nn.Linear(h_dim2, z_dim)
        self.enc3logvar = nn.Linear(h_dim2, z_dim)

    def forward(self, x):
        x = self.model(x)
        return self.enc3mean(x), self.enc3logvar(x)

class DecoderSimpleFC(nn.Module):
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(z_dim, h_dim2),
            nn.ReLU(True),
            nn.Linear(h_dim2, h_dim1),
            nn.ReLU(True),
            nn.Linear(h_dim1, x_dim),
        )
    def forward(self, z):
        return self.model(z)






class VAE(nn.Module):
    # TODO: self.encoder = nn.Sequential()
    # TODO: self.decoder = nn.Sequential()
    # TODO: distributions = self.encoder(x) | z_mean, z_logvar = distributions[:, :self.z_dim], distributions[:, self.z_dim:]

    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        # encoder | p(z|x)
        self.enc1 = nn.Linear(x_dim, h_dim1)
        self.enc2 = nn.Linear(h_dim1, h_dim2)
        self.enc3mu = nn.Linear(h_dim2, z_dim)
        self.enc3var = nn.Linear(h_dim2, z_dim)

        # decoder | p(x|z)
        self.dec1 = nn.Linear(z_dim, h_dim2)
        self.dec2 = nn.Linear(h_dim2, h_dim1)
        self.dec3 = nn.Linear(h_dim1, x_dim)

    def sample_from_latent_distribution(self, z_mean, z_logvar):
        # sample | z ~ p(z|x)           | Gaussian Encoder Model Distribution - pg. 25 in Variational Auto Encoders
        std = torch.exp(0.5 * z_logvar)  # var^0.5 == std
        eps = torch.randn_like(std)  # N(0, 1)
        return z_mean + (std * eps)  # mu + dot(std, eps)

    def forward(self, x):
        # encode
        z_mean, z_logvar = self.encoder(x.view(-1, self.x_dim))
        z = self.sample_from_latent_distribution(z_mean, z_logvar)
        # decode
        x_recon = self.decoder(z).view(x.size())
        return x_recon, z_mean, z_logvar

    def encoder(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        z_mean, z_logvar = self.enc3mu(x), self.enc3var(x)
        return z_mean, z_logvar

    def decoder(self, z):
        z = F.relu(self.dec1(z))
        z = F.relu(self.dec2(z))
        # we compute the final sigmoid activation as part of the loss to improve numerical stability
        x_recon = self.dec3(z)  #  torch.sigmoid(self.dec3(z))
        return x_recon


# ========================================================================= #
# simple 28x28 fully connected models                                       #
# ========================================================================= #




# ========================================================================= #
# simple 64x64 convolutional models                                         #
# ========================================================================= #


class EncoderSimpleConv64(nn.Module):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""

    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__()
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.ReLU(True),
            Flatten3D(),
            nn.Linear(256, latent_dim, bias=True)
        )

    def forward(self, x):
        return self.model(x)


class DecoderSimpleConv64(nn.Module):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""

    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__()
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.model = nn.Sequential(
            Unsqueeze3D(),
            nn.Conv2d(latent_dim, 256, 1, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, num_channels, 3, 1)
        )
        # output shape = bs x 3 x 64 x 64

    def forward(self, x):
        return self.model(x)


# ========================================================================= #
# xy network - TODO: DEPRECATED!                                            #
# ========================================================================= #


# class XYNet(nn.Module):
#     def __init__(self, size=8, arch='decoder'):
#         super().__init__()
#         assert arch in {'encoder', 'decoder', 'full'}
#         self.size = size
#         self.arch = arch
#         if arch != 'decoder':  # encoder
#             self.enc1 = nn.Linear(size ** 2, size ** 2)
#             self.enc2 = nn.Linear(size ** 2, size ** 2)
#             self.enc3 = nn.Linear(size ** 2, size ** 2)
#             self.enc4 = nn.Linear(size ** 2, 2)
#         if arch != 'encoder':  # decoder
#             self.dec1 = nn.Linear(2, size ** 2)
#             self.dec2 = nn.Linear(size ** 2, size ** 2)
#             self.dec3 = nn.Linear(size ** 2, size ** 2)
#             self.dec4 = nn.Linear(size ** 2, size ** 2)
#
#     def encode(self, x):
#         x = x.view((-1, self.size ** 2))
#         x = F.relu(self.enc1(x))
#         x = F.relu(self.enc2(x))
#         x = F.relu(self.enc3(x))
#         x = torch.tanh(self.enc4(x))
#         return x
#
#     def decode(self, x):
#         x = F.relu(self.dec1(x))
#         x = F.relu(self.dec2(x))
#         x = F.relu(self.dec3(x))
#         x = torch.sigmoid(self.dec4(x))
#         return x.view(-1, self.size * self.size)
#
#     def forward(self, x):
#         if self.arch != 'decoder':  # encoder
#             x = self.encode(x)
#         if self.arch != 'encoder':  # decoder
#             x = self.decode(x)
#         return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
