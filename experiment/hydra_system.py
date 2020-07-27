import os

import numpy as np
import torch
import torch.utils.data
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import pytorch_lightning as pl

from disent.dataset import split_dataset
from disent.dataset.ground_truth.base import GroundTruthDataset
from disent.frameworks.semisupervised.adavae import InterceptZMixin
from disent.frameworks.unsupervised.vae import VaeLoss
from disent.model import GaussianEncoderDecoderModel
from disent.util import make_box_str

import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


# ========================================================================= #
# runner                                                                    #
# ========================================================================= #


class UnsupervisedVaeSystem(pl.LightningModule):

    def __init__(self, hparams: DictConfig):
        super().__init__()

        # hyper-parameters
        self.hparams = hparams

        # vae model
        self.model = GaussianEncoderDecoderModel(
            hydra.utils.instantiate(self.hparams.model.encoder.cls),
            hydra.utils.instantiate(self.hparams.model.decoder.cls)
        )

        # framework
        self.framework: VaeLoss = hydra.utils.instantiate(self.hparams.framework.cls)

        # data
        self.dataset_train = None
        self.dataset_test = None

    def prepare_data(self) -> None:
        # *NB* Do not set model parameters here.
        # - trainer.prepare_data_per_node controls behavior.
        # Instantiate data once to download and prepare if needed.
        hydra.utils.instantiate(self.hparams.dataset.cls)  # TODO: this should be in_memory=False by default

    def setup(self, stage: str):
        dataset = GroundTruthDataset(
            ground_truth_data=hydra.utils.instantiate(self.hparams.dataset.cls),
            transform=torchvision.transforms.Compose([
                hydra.utils.instantiate(transform_cls)
                for transform_cls in self.hparams.dataset.transforms
            ])
        )
        self.dataset_train, self.dataset_test = split_dataset(dataset, self.hparams.dataset.train_ratio)

    def training_step(self, batch, batch_idx):
        x = batch
        # encode
        z_mean, z_logvar = self.model.encode_gaussian(x)
        # intercept and mutate if needed
        if isinstance(self.framework, InterceptZMixin):
            z_mean, z_logvar = self.framework.intercept_z(z_mean, z_logvar)
        # reparametrize
        z = self.model.sample_from_latent_distribution(z_mean, z_logvar)
        # reconstruct
        x_recon = self.model.decode(z)
        # compute loss
        losses = self.framework.compute_loss(x, x_recon, z_mean, z_logvar, z)
        # log & train
        loss = losses['loss']
        loss_log_dict = {'train_loss': loss}
        return {
            'loss': loss,
            'log': loss_log_dict,
            # 'progress_bar': loss_log_dict
        }

    def validation_step(self, batch, batch_idx):
        x = batch
        # encode
        z_mean, z_logvar = self.model.encode_gaussian(x)
        # reconstruct
        x_recon = self.model.decode(z_mean)
        # compute loss
        losses = self.framework.compute_loss(x, x_recon, z_mean, z_logvar, z_mean)
        # log & train
        return {
            'val_loss': losses['loss'],
        }

    def validation_epoch_end(self, outputs):
        loss_dict = {k: np.mean([output[k] for output in outputs]) for k in outputs[0].keys()}
        return {
            'val_loss': loss_dict['val_loss'],
            'progress_bar': loss_dict,
            'log': loss_dict,
        }

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        return hydra.utils.instantiate(
            self.hparams.optimizer.cls,
            self.model.parameters()
        )

    @pl.data_loader
    def train_dataloader(self):
        # Sample of data used to fit the model.
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.hparams.dataset.batch_size,
            num_workers=self.hparams.dataset.num_workers,
            shuffle=True
        )

    @pl.data_loader
    def val_dataloader(self):
        # Sample of data used to fit the model.
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.hparams.dataset.batch_size,
            num_workers=self.hparams.dataset.num_workers,
            shuffle=True
        )


# ========================================================================= #
# RUNNER                                                                    #
# ========================================================================= #


@hydra.main(config_path='config', config_name="config")
def main(cfg: DictConfig):
    # print hydra config
    log.info(make_box_str(cfg.pretty()))

    # warn about CUDA
    cuda = cfg.trainer.get('cuda', torch.cuda.is_available())
    if not torch.cuda.is_available():
        if cuda:
            log.error('trainer.cuda=True but CUDA is not available on this machine!')
            exit()
        else:
            log.warning('CUDA is not available on this machine!')
    else:
        if not cuda:
            log.warning('CUDA is available but is not being used!')

    # create trainer loggers
    loggers = []
    if cfg.logging.wandb.enabled:
        loggers.append(WandbLogger(
            name=cfg.logging.wandb.name,
            project=cfg.logging.wandb.project,
            group=cfg.logging.wandb.get('group', None),
            tags=cfg.logging.wandb.get('tags', None),
            entity=cfg.logging.get('entity', None),
            save_dir=os.path.join(cfg.logging.logs_root, 'wandb'),
        ))
    if cfg.logging.tensorboard.enabled:
        loggers.append(TensorBoardLogger(
            save_dir=os.path.join(cfg.logging.logs_root, 'tensorboard')
        ))

    # check data preparation
    prepare_data_per_node = cfg.trainer.get('prepare_data_per_node', True)
    if not os.path.isabs(cfg.dataset.data_dir) and prepare_data_per_node:
        log.warning(
            f'trainer.prepare_data_per_node={repr(prepare_data_per_node)} but '
            f'dataset.data_dir={repr(cfg.dataset.data_dir)} is a relative path which '
            f'may be an error! Try specifying an absolute path that is guaranteed to '
            f'be unique from each node, eg. dataset.data_dir=/tmp/dataset'
        )

    # make & train system
    system = UnsupervisedVaeSystem(cfg)
    trainer = pl.Trainer(
        logger=loggers,
        gpus=1 if cuda else 0,
        max_epochs=cfg.trainer.get('epochs', 1000),
        max_steps=cfg.trainer.get('steps', None),
        prepare_data_per_node=prepare_data_per_node,
    )
    trainer.fit(system)


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    main()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


#         # transforms for images
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             # transforms.Normalize((0.1307,), (0.3081,))
#         ])
#
#         # prepare transforms standard to MNIST
#         self.mnist_train, self.mnist_val = random_split(
#             MNIST('data/dataset', train=True, download=True, transform=transform),
#             [55000, 5000]
#         )