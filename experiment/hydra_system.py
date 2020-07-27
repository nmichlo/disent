import os

import torch
import torch.utils.data
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import pytorch_lightning as pl

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
        # dataset
        self.data = hydra.utils.instantiate(self.hparams.dataset.cls)
        self.dataset_train = GroundTruthDataset(
            ground_truth_data=self.data,
            transform=torchvision.transforms.ToTensor()
        )
        # vae model
        self.model = GaussianEncoderDecoderModel(
            hydra.utils.instantiate(self.hparams.model.encoder.cls),
            hydra.utils.instantiate(self.hparams.model.decoder.cls)
        )
        # framework
        self.framework: VaeLoss = hydra.utils.instantiate(self.hparams.framework.cls)

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
        return {'loss': losses['loss'], 'log': {'train_loss': losses['loss']}}

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
        log.warning('CUDA is not available on this machine!')
    elif not cuda:
        log.warning('CUDA is available but not being used!')

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

    # make & train system
    system = UnsupervisedVaeSystem(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.get('epochs', 1000),
        max_steps=cfg.trainer.get('steps', None),
        gpus=1 if cuda else 0,
        logger=loggers
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