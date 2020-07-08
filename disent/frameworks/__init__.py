from .unsupervised.vae import VaeLoss
from .unsupervised.betavae import BetaVaeLoss
from .unsupervised.betavae import BetaVaeHLoss
from .semisupervised.adavae import AdaVaeLoss
from .supervised.gadavae import GuidedAdaVaeLoss

# ========================================================================= #
# __init__                                                                  #
# ========================================================================= #


def make_vae_loss(name):
    if 'vae' == name:
        return VaeLoss()
    elif 'beta-vae' == name:
        return BetaVaeLoss(beta=4)
    elif 'beta-vae-h' == name:
        raise NotImplementedError('beta-vae-h loss is not yet implemented')
    elif 'ada-gvae' == name:
        return AdaVaeLoss(beta=4, average_mode='gvae')
    elif 'ada-ml-vae' == name:
        return AdaVaeLoss(beta=4, average_mode='ml-vae')
    elif 'g-ada-gvae' == name:
        return GuidedAdaVaeLoss(beta=4, average_mode='gvae')
    elif 'g-ada-ml-vae' == name:
        return GuidedAdaVaeLoss(beta=4, average_mode='ml-vae')
    else:
        raise KeyError(f'Unsupported VAE Framework: {name}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
