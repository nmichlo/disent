from disent.loss.loss import BetaVaeLoss
from disent.model.encoders_decoders import DecoderSimpleFC, EncoderSimpleFC
from disent.model.gaussian_encoder_model import GaussianEncoderModel
from disent.systems.base import BaseLightningModule


# ========================================================================= #
# xy system                                                                 #
# ========================================================================= #


class VaeSystem(BaseLightningModule):

    def __init__(self):
        super().__init__(
            model=GaussianEncoderModel(
                EncoderSimpleFC(),
                DecoderSimpleFC(),
            ),
            loss=BetaVaeLoss(),
            optimizer='radam',
            dataset='mnist',
            lr=0.01,
            batch_size=256
        )

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, z_mean, z_logvar, z = self.forward(x)
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
