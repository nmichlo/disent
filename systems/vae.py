from experiments.system import BaseLightningModule
from loss import BetaVaeLoss, VaeLoss
from model import VAE


# ========================================================================= #
# xy system                                                                 #
# ========================================================================= #


class VaeSystem(BaseLightningModule):

    def __init__(self):
        super().__init__(
            model=VAE(784, 512, 256, 10),
            loss=BetaVaeLoss(),
            optimizer='radam',
            dataset='mnist',
            lr=0.01,
            batch_size=256
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_recon, z_mean, z_logvar = self.forward(x)
        loss = self.loss(x, x_recon, z_mean, z_logvar)
        return {
            'loss': loss,
            'log': {
                'train_loss': loss
            }
        }

# ========================================================================= #
# Main                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    system = VaeSystem()
    system.quick_train(epochs=10)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
