import torch
import torch.nn.functional as F


# ========================================================================= #
# Vae Loss Functions                                                        #
# ========================================================================= #


class AeLoss(object):

    def activate(self, x):
        """
        The final activation of the model.
        - Never use this in a training loop.
        """
        raise NotImplementedError

    def training_loss(self, x_partial_recon, x_targ):
        """
        Takes in an **unactivated** tensor from the model
        as well as an original target from the dataset.
        :return: The computed mean loss
        """
        raise NotImplementedError


class AeLossMse(AeLoss):

    def activate(self, x):
        return x

    def training_loss(self, x_partial_recon, x_targ):
        return F.mse_loss(x_partial_recon, x_targ, reduction='mean')


class AeLossBce(AeLoss):

    def activate(self, x):
        return torch.sigmoid(x)

    def training_loss(self, x_partial_recon, x_targ):
        return F.binary_cross_entropy_with_logits(x_partial_recon, x_targ, reduction='mean')


def make_vae_recon_loss(name) -> AeLoss:
    if name == 'mse':
        return AeLossMse()
    elif name == 'bce':
        return AeLossBce()
    else:
        raise KeyError(f'Invalid vae reconstruction loss: {name}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

