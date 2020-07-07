from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from disent.frameworks import make_vae_loss
from disent.frameworks.semisupervised.adavae import InterceptZMixin
from disent.model import make_model, make_optimizer
from disent.dataset import make_ground_truth_dataset
from disent.dataset.ground_truth.base import (GroundTruthData, PairedVariationDataset, RandomPairDataset)

from disent.util import load_model, save_model

# ========================================================================= #
# xy system                                                                 #
# ========================================================================= #


@dataclass
class HParams:
    # MODEL
    model: str = 'simple-fc'
    z_size: int = 6
    # OPTIMIZER
    optimizer: str = 'radam'
    lr: float = 0.001
    weight_decay: float = 0.
    # LOSS
    loss: str = 'vae'
    # DATASET
    dataset: str = '3dshapes'
    try_in_memory: bool = False
    batch_size: int = 64
    num_workers: int = 4
    # PAIR DATASET
    k: str = 'uniform'

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k in dir(self)
            if not (k.startswith('_') or k ==  'to_dict')
        }


class VaeSystem(pl.LightningModule):
    """
    Base system that wraps a model. Includes factories for datasets, optimizers and loss.
    """

    def __init__(self, hparams: HParams=None):
        super().__init__()

        # parameters
        self.hparams: HParams = hparams if isinstance(hparams, HParams) else HParams(**(hparams if hparams else {}))
        # make
        self.model = make_model(self.hparams.model, z_size=self.hparams.z_size)
        self.loss = make_vae_loss(self.hparams.loss)
        self.dataset_train: Dataset = make_ground_truth_dataset(self.hparams.dataset, try_in_memory=self.hparams.try_in_memory)
        # convert dataset for paired loss
        if self.loss.is_pair_loss:
            if isinstance(self.dataset_train, GroundTruthData):
                self.dataset_train_pairs = PairedVariationDataset(self.dataset_train, k=self.hparams.k)
            else:
                self.dataset_train_pairs = RandomPairDataset(self.dataset_train)

    def training_step(self, batch, batch_idx):
        if self.loss.is_pair_loss:
            return self._train_step_pair(batch, batch_idx)
        else:
            return self._train_step_single(batch, batch_idx)

    def _train_step_single(self, batch, batch_idx):
        x = batch
        x_recon, z_mean, z_logvar, z = self.forward(x)
        losses = self.loss(x, x_recon, z_mean, z_logvar, z)
        # log & train
        return {'loss': losses['loss'], 'log': {'train_loss': losses['loss']}}

    def _train_step_pair(self, batch, batch_idx):
        x, x2 = batch
        # feed forward
        # TODO: this is hacky and moves functionality out of the right places
        z_mean, z_logvar = self.model.encode_gaussian(x)
        z2_mean, z2_logvar = self.model.encode_gaussian(x2)
        if isinstance(self.loss, InterceptZMixin):
            z_mean, z_logvar, z2_mean, z2_logvar = self.loss.intercept_z_pair(z_mean, z_logvar, z2_mean, z2_logvar)
        z = self.model.sample_from_latent_distribution(z_mean, z_logvar)
        z2 = self.model.sample_from_latent_distribution(z2_mean, z2_logvar)
        x_recon = self.model.decode(z)
        x2_recon = self.model.decode(z2)
        # compute loss
        losses = self.loss(x, x_recon, z_mean, z_logvar, z, x2, x2_recon, z2_mean, z2_logvar, z2)
        # log & train
        return {'loss': losses['loss'], 'log': {'train_loss': losses['loss']}}

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        return make_optimizer(self.hparams.optimizer, self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @pl.data_loader
    def train_dataloader(self):
        # Sample of data used to fit the model.
        return torch.utils.data.DataLoader(
            self.dataset_train_pairs if self.loss.is_pair_loss else self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
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
    # system = VaeSystem(dataset_train='dsprites', model='simple-fc', loss='vae', hparams=dict(num_workers=8, batch_size=64, z_size=6))
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
    #     dataset_train='dsprites', model='simple-fc', loss='vae', hparams=dict(num_workers=8, batch_size=64, z_size=6)
    # )
    #
    # # print('Done!')
    #
    # params = torch.load("temp.model")
    # print(list(params['state_dict'].keys()))
    # print(params['state_dict']['model.gaussian_encoder.model.1.weight'])

    # model = make_model('simple-fc', z_size=6)

    system = VaeSystem(HParams(loss='ada-gvae'))
    system.quick_train()
    save_model(system, 'data/model/temp.model')
    load_model(system, 'data/model/temp.model')

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
