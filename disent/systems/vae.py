import torchvision
from torch.utils.data import DataLoader

from disent.dataset.ground_truth.xygrid import XYDataset
from disent.dataset.util import PairedVariationDataset
from disent.loss.loss import AdaGVaeLoss, BetaVaeLoss, VaeLoss
from disent.model.encoders_decoders import DecoderSimpleFC, EncoderSimpleFC
from disent.model.gaussian_encoder_model import GaussianEncoderModel
from disent.systems.base import BaseLightningModule

import numpy as np


# ========================================================================= #
# xy system                                                                 #
# ========================================================================= #


class VaeSystem(BaseLightningModule):

    def __init__(self):
        super().__init__(
            model=GaussianEncoderModel(
                EncoderSimpleFC(z_dim=10),
                DecoderSimpleFC(z_dim=10),
            ),
            loss=AdaGVaeLoss(),
            optimizer='radam',
            dataset='3dshapes',
            lr=0.001,
            batch_size=64
        )

        from disent.dataset.ground_truth.shapes3d import Shapes3dDataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(28),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
        ])
        self.dataset_train = Shapes3dDataset(transform=transform)
        # self.paired_dataset = PairedVariationDataset(self.paired_dataset, k='uniform')

        # self.dataset_train = XYDataset(width=28, transform=torchvision.transforms.ToTensor())
        self.dataset_train = PairedVariationDataset(self.dataset_train, k='uniform')

    def training_step(self, batch, batch_idx):
        if isinstance(self.dataset_train, PairedVariationDataset):
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

        else:
            x = batch
            x_recon, z_mean, z_logvar, z = self.forward(x)
            loss_components = self.loss(x, x_recon, z_mean, z_logvar, z)

            return {
                'loss': loss_components['loss'],
                'elbo': loss_components['elbo'],
                'log': {
                    'train_loss': loss_components['loss']
                }
            }

# ========================================================================= #
# Main                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    system = VaeSystem()
    system.quick_train(epochs=100)

    for x in DataLoader(XYDataset(width=28, transform=torchvision.transforms.ToTensor()), batch_size=16):
        x_recon, _, _, _ = system.model.forward(x, deterministic=True)
        print(x_recon.shape)
        for i, item in enumerate(x_recon.cpu().detach().numpy()):
            print(f'{i+1}:')
            print(np.round(item, 2))
            print()
        break


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
