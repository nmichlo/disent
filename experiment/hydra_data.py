import hydra
import torch.utils.data
import pytorch_lightning as pl
import torchvision
from omegaconf import DictConfig

from disent.dataset.groundtruth import GroundTruthDataset
from disent.transform.groundtruth import GroundTruthDatasetBatchAugment
from experiment.util.hydra_utils import instantiate_recursive


# ========================================================================= #
# DATASET                                                                   #
# ========================================================================= #


class HydraDataModule(pl.LightningDataModule):

    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.hparams = hparams
        # transform: prepares data from datasets
        self.data_transform = instantiate_recursive(self.hparams.dataset.transform)
        # input_transform: applied to anything before being fed through the model
        # input_transform_aug: augment data for inputs, then apply input_transform
        augment = instantiate_recursive(self.hparams.augment.transform)
        self.input_transform_noaug = instantiate_recursive(self.hparams.framework.input_transform)
        self.input_transform_aug = torchvision.transforms.Compose([augment, self.input_transform_noaug])
        assert callable(self.data_transform)
        assert callable(augment)
        assert callable(self.input_transform_noaug)
        # batch_augment: augments transformed data for inputs, should be applied across a batch, same as self.augment
        self.batch_augment = GroundTruthDatasetBatchAugment(transform=self.input_transform_aug)
        # datasets
        self._dataset_train: GroundTruthDataset = None
        self.dataset_train_noaug: GroundTruthDataset = None
        self.dataset_train_aug: GroundTruthDataset = None

    def prepare_data(self) -> None:
        # *NB* Do not set model parameters here.
        # - Instantiate data once to download and prepare if needed.
        # - trainer.prepare_data_per_node affects this functions behavior per node.
        data = self.hparams.dataset.data.copy()
        if 'in_memory' in data:
            del data['in_memory']
        hydra.utils.instantiate(data)

    def setup(self, stage=None) -> None:
        # ground truth data
        data = hydra.utils.instantiate(self.hparams.dataset.data)
        # Wrap the data for the framework some datasets need triplets, pairs, etc.
        # Augmentation is done inside the frameworks so that it can be done on the GPU, otherwise things are very slow.
        self._dataset_train = hydra.utils.instantiate(self.hparams.framework.data_wrapper, ground_truth_data=data, transform=self.data_transform, augment=None)
        self.dataset_train_noaug = hydra.utils.instantiate(self.hparams.framework.data_wrapper, ground_truth_data=data, transform=self.data_transform, augment=self.input_transform_noaug)
        self.dataset_train_aug = hydra.utils.instantiate(self.hparams.framework.data_wrapper, ground_truth_data=data, transform=self.data_transform, augment=self.input_transform_aug)
        assert isinstance(self.dataset_train_noaug, GroundTruthDataset)
        assert isinstance(self.dataset_train_aug, GroundTruthDataset)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Training Dataset:
    #     The sample of data used to fit the model.
    # Validation Dataset:
    #     Data used to provide an unbiased evaluation of a model fit on the
    #     training dataset while tuning model hyperparameters. The
    #     evaluation becomes more biased as skill on the validation
    #     dataset is incorporated into the model configuration.
    # Test Dataset:
    #     The sample of data used to provide an unbiased evaluation of a
    #     final model fit on the training dataset.
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def train_dataloader(self):
        """Training Dataset: Sample of data used to fit the model"""
        return torch.utils.data.DataLoader(
            self._dataset_train,
            batch_size=self.hparams.dataset.batch_size,
            num_workers=self.hparams.dataset.num_workers,
            shuffle=True
        )