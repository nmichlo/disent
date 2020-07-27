
import os

from experiment.util.slurm import slurm_run


def make_system():

    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, random_split
    from torch.nn import functional as F
    from torchvision.datasets import MNIST
    from torchvision import transforms
    import os

    class LightningVAE(pl.LightningModule):

        def __init__(self):
            super().__init__()
            # mnist images are (1, 28, 28) (channels, width, height)
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(28 * 28, 256), torch.nn.ReLU(inplace=True),
                torch.nn.Linear(256, 128),     torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, 10)
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(10, 128),  torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, 256), torch.nn.ReLU(inplace=True),
                torch.nn.Linear(256, 28 * 28),
            )

        def forward(self, x):
            batch_size, channels, width, height = x.size()
            # (b, 1, 28, 28) -> (b, 1*28*28)
            z = self.encoder(x.view(batch_size, -1))
            x_recon = self.decoder(z)
            return x_recon.view(x.shape)

        def training_step(self, train_batch, batch_idx):
            x, y = train_batch
            logits = self.forward(x)
            loss = F.binary_cross_entropy_with_logits(logits, x)
            return {
                'loss': loss,
                'log': {'train_loss': loss}
            }

        def validation_step(self, val_batch, batch_idx):
            x, y = val_batch
            logits = self.forward(x)
            loss = F.binary_cross_entropy_with_logits(logits, x)
            return {
                'val_loss': loss
            }

        def validation_epoch_end(self, outputs):
            # called at the end of the validation epoch
            # outputs is an array with what you returned in validation_step for each batch
            # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            return {
                'avg_val_loss': avg_loss,
                'log': {'val_loss': avg_loss}
            }

        def prepare_data(self):
            # transforms for images
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])

            # prepare transforms standard to MNIST
            self.mnist_train, self.mnist_val = random_split(
                MNIST('data/dataset', train=True, download=True, transform=transform),
                [55000, 5000]
            )

        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=64, num_workers=12)

        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=64, num_workers=12)

        # def test_dataloader(self):
        #     return DataLoader(self.mnist_test, batch_size=64)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

    return LightningVAE()


LOGS_DIR_SUBMITIT = 'logs/submitit'
LOGS_DIR_WANDB = 'logs/wandb'
LOGS_DIR_COMET = 'logs/comet'
LOGS_DIR_TENSORBOARD = 'logs/tensorboard'

import submitit


def main():
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

    model = make_system()

    trainer = pl.Trainer(
        max_epochs=5,
        gpus=1,
        logger=[
            WandbLogger(
                name=f'disent-{submitit.JobEnvironment().job_id}',
                group=None,  # https://docs.wandb.com/library/advanced/grouping
                tags=None,
                project='disentanglement',
                entity=None,  # the team posting this run
                save_dir=LOGS_DIR_WANDB,
            ),
            TensorBoardLogger(save_dir=LOGS_DIR_TENSORBOARD)
        ]
    )

    trainer.fit(model)


if __name__ == '__main__':

    slurm_run(
        main,
        join=False,
        logs_dir=LOGS_DIR_SUBMITIT,
        # slurm args
        timeout_min=60,
        slurm_partition='batch',
        slurm_job_name='disent',
    )
