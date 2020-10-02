from typing import List

import torch
from torch import Tensor
from torchvision.models import vgg19_bn
from torch.nn import functional as F

from disent.frameworks.vae.unsupervised import BetaVae
from disent.frameworks.vae.loss import kl_normal_loss


# ========================================================================= #
# Dfc Vae                                                                   #
# ========================================================================= #


class DfcVae(BetaVae):
    """
    Deep Feature Consistent Variational Autoencoder.
    https://arxiv.org/abs/1610.00291
    - Uses features generated from a pretrained model as the loss.
    """

    def __init__(
            self,
            make_optimizer_fn,
            make_model_fn,
            batch_augment=None,
            beta=4,
            feature_layers=None
    ):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, beta=beta)
        # make dfc loss
        self._loss_module = DfcLossModule(feature_layers=feature_layers)

    def compute_training_loss(self, batch, batch_idx):
        (x,), (x_targ,) = batch['x'], batch['x_targ']

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        z_mean, z_logvar = self.encode_gaussian(x)
        # sample from latent distribution
        z = self.reparameterize(z_mean, z_logvar)
        # reconstruct without the final activation
        x_recon = self.decode_partial(z)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        feature_loss = self._loss_module.compute_loss(x_recon, x_targ)
        pixel_loss = F.mse_loss(torch.sigmoid(x_recon), x_targ, reduction='mean')  # E[log p(x|z)] we typically use binary cross entropy with logits
        recon_loss = (pixel_loss + feature_loss) * 0.5
        # KL divergence
        kl_loss = kl_normal_loss(z_mean, z_logvar)     # D_kl(q(z|x) || p(z|x))
        # compute kl regularisation
        kl_reg_loss = self.kl_regularization(kl_loss)
        # compute combined loss
        loss = recon_loss + kl_reg_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': recon_loss,
            'kl_reg_loss': kl_reg_loss,
            'kl_loss': kl_loss,
            'elbo': -(recon_loss + kl_loss),
            'pixel_loss': pixel_loss,
            'feature_loss': feature_loss,
        }


# ========================================================================= #
# Helper Loss                                                               #
# ========================================================================= #


class DfcLossModule(torch.nn.Module):
    """
    Loss function for the Deep Feature Consistent Variational Autoencoder.
    https://arxiv.org/abs/1610.00291
    - reference implementation is from:
    """

    def __init__(self, feature_layers=None):
        """
        :param feature_layers: List of string of IDs of feature layers in pretrained model
        """
        super().__init__()
        # feature layers to use
        self.feature_layers = set(['14', '24', '34', '43'] if (feature_layers is None) else feature_layers)
        # feature network
        self.feature_network = vgg19_bn(pretrained=True)
        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False
        # Evaluation Mode
        self.feature_network.eval()

    def __call__(self, x_recon, x_targ):
        return self.compute_loss(x_recon, x_targ)

    def compute_loss(self, x_recon, x_targ):
        features_recon = self._extract_features(x_recon)
        features_targ = self._extract_features(x_targ)
        # compute losses
        feature_loss = 0.0
        for (f_recon, f_targ) in zip(features_recon, features_targ):
            feature_loss += F.mse_loss(f_recon, f_targ, reduction='mean')
        return feature_loss

    def _extract_features(self, inputs: Tensor) -> List[Tensor]:
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param inputs: (Tensor) [B x C x H x W]
        :return: List of the extracted features
        """
        features = []
        result = inputs
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if key in self.feature_layers:
                features.append(result)
        return features


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
