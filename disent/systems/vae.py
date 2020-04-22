from types import SimpleNamespace
import torch
from pytorch_lightning import Trainer
from torch.utils.data import Dataset
from disent.dataset.ground_truth.base import GroundTruthDataset, PairedVariationDataset, RandomPairDataset
from disent.factory import make_ground_truth_dataset, make_model, make_vae_loss, make_optimizer
import pytorch_lightning as pl
from pytorch_lightning import loggers


# ========================================================================= #
# xy system                                                                 #
# ========================================================================= #


class VaeSystem(pl.LightningModule):
    """
    Base system that wraps a model. Includes factories for datasets, optimizers and loss.
    """

    def __init__(
            self,
            model='simple-fc',
            loss='ada-gvae',
            optimizer='radam',
            dataset_train='mnist',
            hparams=None
    ):
        super().__init__()

        # default hparams
        if hparams is None:
            hparams = SimpleNamespace(lr=0.001, batch_size=64, num_workers=4, k='uniform')
        if isinstance(hparams, dict):
            hparams = SimpleNamespace(**hparams)

        # vars
        self.hparams = hparams

        # make
        self.model = make_model(model, z_dim=10) if isinstance(model, str) else model
        self.loss = make_vae_loss(loss) if isinstance(loss, str) else loss
        self.optimizer = optimizer
        self.dataset_train: Dataset = make_ground_truth_dataset(dataset_train) if isinstance(dataset_train, str) else dataset_train

        # convert dataset for paired loss
        if self.loss.is_pair_loss:
            if isinstance(self.dataset_train, GroundTruthDataset):
                self.dataset_train = PairedVariationDataset(self.dataset_train, k=hparams.k)
            else:
                self.dataset_train = RandomPairDataset(self.dataset_train)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        if self.loss.is_pair_loss:
            x, x2 = batch
            x_recon, z_mean, z_logvar, z = self.forward(x)
            x2_recon, z2_mean, z2_logvar, z2 = self.forward(x2)
            losses = self.loss(x, x_recon, z_mean, z_logvar, z, x2, x2_recon, z2_mean, z2_logvar, z2)
        else:
            x = batch
            x_recon, z_mean, z_logvar, z = self.forward(x)
            losses = self.loss(x, x_recon, z_mean, z_logvar, z)

        return {
            'loss': losses['loss'],
            'log': {
                'train_loss': losses['loss']
            }
        }

    def configure_optimizers(self):
        return make_optimizer('radam', self.parameters(), lr=self.hparams.lr)

    @pl.data_loader
    def train_dataloader(self):
        # Sample of data used to fit the model.
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def quick_train(self, epochs=10, show_progress=True, *args, **kwargs) -> Trainer:
        trainer = Trainer(max_epochs=epochs, show_progress_bar=show_progress, *args, **kwargs)
        trainer.fit(self)
        return trainer


# ========================================================================= #
# Main                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    system = VaeSystem()
    system.quick_train(
        epochs=1,
        loggers=loggers.TensorBoardLogger('logs/')
    )

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
