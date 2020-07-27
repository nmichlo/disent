from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from disent.frameworks.semisupervised.adavae import InterceptZMixin
from disent.dataset.ground_truth.base import (GroundTruthData, PairedVariationDataset, RandomPairDataset, SupervisedTripletDataset)
from disent.util import chunked


# ========================================================================= #
# general system                                                            #
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
    beta: float = 4
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
        self.params: HParams = hparams if isinstance(hparams, HParams) else HParams(**(hparams if hparams else {}))
        # make
        # TODO: THIS IS BEING REPLACED WITH HYDRA
        raise NotImplementedError
        self.model = DEPRICATED_make_model(self.params.model, z_size=self.params.z_size)

        # TODO: THIS IS BEING REPLACED WITH HYDRA
        raise NotImplementedError
        self.loss = DEPRICATED_make_vae_loss(self.params.loss, self.params.beta)

        # TODO: THIS IS BEING REPLACED WITH HYDRA
        raise NotImplementedError
        self.dataset: Dataset = DEPRICATED_make_ground_truth_dataset(self.params.dataset, try_in_memory=self.params.try_in_memory)

        # convert dataset for paired loss
        if self.loss.required_observations == 1:
            self.dataset_train = self.dataset
        elif self.loss.required_observations == 2:
            if isinstance(self.dataset, GroundTruthData):
                self.dataset_train = PairedVariationDataset(self.dataset, k=self.params.k)
            else:
                self.dataset_train = RandomPairDataset(self.dataset)
        elif self.loss.required_observations == 3:
            assert isinstance(self.dataset, GroundTruthData)
            self.dataset_train = SupervisedTripletDataset(self.dataset)
        else:
            raise NotImplementedError(f'Unsupported number of observations required per step: n > 3')

    def training_step(self, batch, batch_idx):
        """
        Generalised training step that can handle any number of observations in each sample.
        (loss.is_single + loss.is_pair + loss.is_triplet + ...)
        This is slightly slow, and thus it could be advantageous to specialise on the more simple cases.
        """
        # handle single case
        if isinstance(batch, torch.Tensor):
            batch = [batch]
        assert len(batch) == self.loss.required_observations, f'Incorrect number of observations ({len(batch)}) for loss: {self.loss.__class__.__name__} ({self.loss.required_observations})'

        # encode [z_mean, z_logvar, ...]
        z_params = [z_component for x in batch for z_component in self.model.encode_gaussian(x)]
        # intercept and mutate if needed [z_mean, z_logvar, ...]
        if isinstance(self.loss, InterceptZMixin):
            z_params = self.loss.intercept_z(*z_params)
        # reparametrize [z, ...]
        zs = [self.model.sample_from_latent_distribution(z_mean, z_logvar) for z_mean, z_logvar in chunked(z_params, 2)]
        # reconstruct [x_recon, ...]
        x_recons = [self.model.decode(z) for z in zs]

        # compute loss [x, x_recon, z_mean, z_logvar, z, ...]
        loss_params = [param for (x, x_recon, (z_mean, z_logvar), z) in zip(batch, x_recons, chunked(z_params, 2), zs) for param in (x, x_recon, z_mean, z_logvar, z)]
        losses = self.loss(*loss_params)

        # log & train
        return {'loss': losses['loss'], 'log': {'train_loss': losses['loss']}}

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        # TODO: THIS IS BEING REPLACED WITH HYDRA
        raise NotImplementedError
        return DEPRICATED_make_optimizer(self.params.optimizer, self.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

    @pl.data_loader
    def train_dataloader(self):
        # Sample of data used to fit the model.
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
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
# END                                                                       #
# ========================================================================= #
