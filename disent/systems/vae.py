import torchvision

from disent.dataset.util import PairedVariationDataset
from disent.loss.loss import AdaGVaeLoss, BetaVaeLoss
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
            loss=AdaGVaeLoss(),
            optimizer='radam',
            dataset='mnist',
            lr=0.01,
            batch_size=8
        )

        from disent.dataset.ground_truth.shapes3d import Shapes3dDataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(28),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
        ])
        self.dataset_train = Shapes3dDataset(transform=transform)
        self.dataset_train = PairedVariationDataset(self.dataset_train, k='uniform')

    def training_step(self, batch, batch_idx):
        x, x2 = batch

        x_recon, z_mean, z_logvar, z = self.forward(x)
        x2_recon, z2_mean, z2_logvar, z2 = self.forward(x2)

        loss = self.loss(x, x_recon, z_mean, z_logvar, z, x2, x2_recon, z2_mean, z2_logvar, z2)
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
