import pytorch_lightning as pl
import torch.utils.data
from pytorch_lightning import Trainer
from disent.factory import make_optimizer, make_datasets


# ========================================================================= #
# system                                                                   #
# ========================================================================= #


class BaseLightningModule(pl.LightningModule):
    def __init__(self, model, loss, optimizer='radam', dataset='mnist', lr=0.001, batch_size=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # model
        self.model = model
        # optimizer
        self.optimizer = optimizer
        self.lr = lr
        # loss
        self.loss = loss
        # dataset
        self.batch_size = batch_size
        self.dataset_train, self.dataset_test = make_datasets(dataset)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {
            'loss': loss,
            'log': {
                'train_loss': loss
            }
        }

    def configure_optimizers(self):
        return make_optimizer(self.model.parameters(), self.optimizer, lr=self.lr)

    def _make_dataset(self, train):
        if self.dataset == 'mnist':
            return

    @pl.data_loader
    def train_dataloader(self):
        # Sample of data used to fit the model.
        return torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size)

    # @pl.data_loader
    # def val_dataloader(self):
        # Sample of data used to provide an unbiased evaluation of a model fit on the training dataset
        # while tuning model hyperparameters. The evaluation becomes more biased as skill on the validation
        # dataset is incorporated into the model configuration.
        # return torch.utils.data.DataLoader(self.dataset_test, batch_size=self.batch_size)

    # @pl.data_loader
    # def test_dataloader(self):
    #     # Sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.
    #     return torch.utils.data.DataLoader(self.dataset_test, batch_size=self.batch_size)

    # def validation_step(self, batch, batch_idx):
    #     return {
    #         'val_loss': self.training_step(batch, batch_idx)['loss'],
    #         # 'val_in': batch,
    #         # 'val_out': self.forward(batch),
    #     }
    #
    # def validation_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #
    #     # Print Extra Info
    #     if self.validation_count % 50 == 0:
    #         print(f'[{self.validation_count}] Ended Validation: {avg_loss}')
    #     self.validation_count += 1
    #
    #     return {
    #         'avg_val_loss': avg_loss,
    #         'log': {'val_loss': avg_loss}
    #     }

    def quick_train(self, epochs=10, show_progress=True, *args, **kwargs) -> Trainer:
        trainer = Trainer(max_epochs=epochs, show_progress_bar=show_progress, *args, **kwargs)
        trainer.fit(self)
        return trainer



# ========================================================================= #
# END                                                                       #
# ========================================================================= #
