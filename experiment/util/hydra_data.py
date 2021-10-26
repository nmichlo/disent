#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import logging
import warnings

import hydra
import torch.utils.data
import pytorch_lightning as pl
from omegaconf import DictConfig

from disent.dataset import DisentDataset
from disent.dataset.transform import DisentDatasetTransform


log = logging.getLogger(__name__)


# ========================================================================= #
# DISENT DATASET MODULE                                                     #
# TODO: possible implementation outline for disent                          #
# ========================================================================= #


# class DisentDatasetModule(pl.LightningDataModule):
#
#     def prepare_data(self, *args, **kwargs):
#         raise NotImplementedError
#
#     def setup(self, stage: Optional[str] = None):
#         raise NotImplementedError
#
#     # DATASET HANDLING
#
#     @property
#     def dataset_train(self) -> DisentDataset:
#         # this property should check `framework_applies_augment` to return the
#         # dataset with the correct transforms/augments applied.
#         # - train_dataloader() should create the DataLoader from this dataset object
#         raise NotImplementedError
#
#     # FRAMEWORK AUGMENTATION HANDLING
#
#     @property
#     def framework_applies_augment(self) -> bool:
#         # if we augment the data in the framework rather, we can augment on the GPU instead
#         # framework needs manual handling of this augmentation mode
#         raise NotImplementedError
#
#     def framework_augment(self, batch):
#         # the augment to be applied if `framework_applies_augment` is `True`, otherwise
#         # this method should do nothing!
#         raise NotImplementedError


# ========================================================================= #
# DATASET                                                                   #
# ========================================================================= #


class HydraDataModule(pl.LightningDataModule):

    def __init__(self, hparams: DictConfig):
        super().__init__()
        # support pytorch lightning < 1.4
        if not hasattr(self, 'hparams'):
            self.hparams = DictConfig(hparams)
        else:
            self.hparams.update(hparams)
        # transform: prepares data from datasets
        self.data_transform = hydra.utils.instantiate(self.hparams.dataset.transform)
        assert (self.data_transform is None) or callable(self.data_transform)
        # input_transform_aug: augment data for inputs, then apply input_transform
        self.input_transform = hydra.utils.instantiate(self.hparams.augment.augment_cls)
        assert (self.input_transform is None) or callable(self.input_transform)
        # batch_augment: augments transformed data for inputs, should be applied across a batch
        # which version of the dataset we need to use if GPU augmentation is enabled or not.
        # - corresponds to below in train_dataloader()
        if self.hparams.dsettings.dataset.gpu_augment:
            # TODO: this is outdated!
            self.batch_augment = DisentDatasetTransform(transform=self.input_transform)
            warnings.warn('`gpu_augment=True` is outdated and may no longer be equivalent to `gpu_augment=False`')
        else:
            self.batch_augment = None
        # datasets initialised in setup()
        self.dataset_train_noaug: DisentDataset = None
        self.dataset_train_aug: DisentDataset = None

    def prepare_data(self) -> None:
        # *NB* Do not set model parameters here.
        # - Instantiate data once to download and prepare if needed.
        # - trainer.prepare_data_per_node affects this functions behavior per node.
        data = dict(self.hparams.dataset.data)
        if 'in_memory' in data:
            del data['in_memory']
        # create the data
        # - we instantiate the data twice, once here and once in setup otherwise
        #   things could go wrong. We try be efficient about it by removing the
        #   in_memory argument if it exists.
        log.info(f'Data - Preparation & Downloading')
        hydra.utils.instantiate(data)

    def setup(self, stage=None) -> None:
        # ground truth data
        log.info(f'Data - Instance')
        data = hydra.utils.instantiate(self.hparams.dataset.data)
        # Wrap the data for the framework some datasets need triplets, pairs, etc.
        # Augmentation is done inside the frameworks so that it can be done on the GPU, otherwise things are very slow.
        self.dataset_train_noaug = DisentDataset(data, hydra.utils.instantiate(self.hparams.sampling._sampler_.sampler_cls), transform=self.data_transform, augment=None)
        self.dataset_train_aug = DisentDataset(data, hydra.utils.instantiate(self.hparams.sampling._sampler_.sampler_cls), transform=self.data_transform, augment=self.input_transform)

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
        """
        Training Dataset: Sample of data used to fit the model.
        """
        # Select which version of the dataset we need to use if GPU augmentation is enabled or not.
        # - corresponds to above in __init__()
        if self.hparams.dsettings.dataset.gpu_augment:
            dataset = self.dataset_train_noaug
        else:
            dataset = self.dataset_train_aug
        # get default kwargs
        default_kwargs = {
            'shuffle': True,
            # This should usually be TRUE if cuda is enabled.
            # About 20% faster with the xysquares dataset, RTX 2060 Rev. A, and Intel i7-3930K
            'pin_memory': self.hparams.dsettings.trainer.cuda
        }
        # get config kwargs
        kwargs = self.hparams.dataloader
        # check required keys
        if ('batch_size' not in kwargs) or ('num_workers' not in kwargs):
            raise KeyError(f'`dataset.dataloader` must contain keys: ["batch_size", "num_workers"], got: {sorted(kwargs.keys())}')
        # create dataloader
        return torch.utils.data.DataLoader(dataset=dataset, **{**default_kwargs, **kwargs})
