from .unsupervised.vae import VaeLoss
from .unsupervised.betavae import BetaVaeLoss
from .unsupervised.betavae import BetaVaeHLoss
from .semisupervised.adavae import AdaGVaeLoss
from .semisupervised.adavae import AdaMlVaeLoss


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
        return AdaGVaeLoss(beta=4)
    elif 'ada-ml-vae' == name:
        return AdaMlVaeLoss(beta=4)
    else:
        raise KeyError(f'Unsupported Ground Truth Dataset: {name}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
