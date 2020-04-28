from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning import Trainer

from disent.util import load_model, save_model
from disent.loss import make_vae_loss
from disent.model import make_model, make_optimizer
from disent.dataset import make_ground_truth_dataset
from disent.dataset.ground_truth.base import (GroundTruthData, GroundTruthDataset, PairedVariationDataset,
                                              RandomPairDataset)


# ========================================================================= #
# xy system                                                                 #
# ========================================================================= #
from disent.visualize.util import get_data


class VaeSystem(pl.LightningModule):
    """
    Base system that wraps a model. Includes factories for datasets, optimizers and loss.
    """

    def __init__(
            self,
            model='simple-fc',
            loss='vae',
            optimizer='radam',
            dataset_train='3dshapes',
            hparams=None
    ):
        super().__init__()

        # parameters
        self.my_hparams = SimpleNamespace(**{
            # defaults
            **dict(
                lr=0.001,
                batch_size=64,
                num_workers=4,
                k='uniform',
                z_dim=6,
            ),
            # custom values
            **(hparams if hparams else {})
        })

        # make
        self.model = make_model(model, z_dim=self.my_hparams.z_dim) if isinstance(model, str) else model
        self.loss = make_vae_loss(loss) if isinstance(loss, str) else loss
        self.optimizer = optimizer
        self.dataset_train: Dataset = make_ground_truth_dataset(dataset_train)

        # convert dataset for paired loss
        if self.loss.is_pair_loss:
            if isinstance(self.dataset_train, GroundTruthData):
                self.dataset_train = PairedVariationDataset(self.dataset_train, k=self.my_hparams.k)
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
            # 'log': {
            #     'train_loss': losses['loss']
            # }
        }

    def configure_optimizers(self):
        return make_optimizer('radam', self.parameters(), lr=self.my_hparams.lr)

    @pl.data_loader
    def train_dataloader(self):
        # Sample of data used to fit the model.
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.my_hparams.batch_size,
            num_workers=self.my_hparams.num_workers,
            shuffle=True
        )

    def quick_train(self, epochs=10, steps=None, show_progress=True, *args, **kwargs) -> Trainer:
        # warn if GPUS are not avaiable
        if torch.cuda.is_available():
            gpus = 1
        else:
            gpus = None
            print('[\033[93mWARNING\033[0m]: cuda is not available!')
        # train
        trainer = Trainer(
            max_epochs=epochs,
            max_steps=steps,
            gpus=gpus,
            show_progress_bar=show_progress,
            checkpoint_callback=False,  # dont save checkpoints
            *args, **kwargs
        )
        trainer.fit(self)
        return trainer


# ========================================================================= #
# Main                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    # system = VaeSystem(dataset_train='dsprites', model='simple-fc', loss='vae', hparams=dict(num_workers=8, batch_size=64, z_dim=6))
    # print('Training')
    # trainer = system.quick_train(
    #     steps=16,
    #     loggers=None, # loggers.TensorBoardLogger('logs/')
    # )
    #
    # print('Saving')
    # trainer.save_checkpoint("temp.model")
    #
    # # print('Loading')
    # loaded_system = VaeSystem.load_from_checkpoint(
    #     checkpoint_path="temp.model",
    #     # constructor
    #     dataset_train='dsprites', model='simple-fc', loss='vae', hparams=dict(num_workers=8, batch_size=64, z_dim=6)
    # )
    #
    # # print('Done!')
    #
    # params = torch.load("temp.model")
    # print(list(params['state_dict'].keys()))
    # print(params['state_dict']['model.gaussian_encoder.model.1.weight'])


    # model = make_model('simple-fc', z_dim=6)

    system = VaeSystem(dataset_train='xygrid', model='simple-fc', loss='vae',
                       hparams=dict(num_workers=8, batch_size=64, z_dim=6))
    system.quick_train()
    save_model(system, 'temp.model')
    load_model(system, 'temp.model')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
