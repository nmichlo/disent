import math

import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.dataset.transform import ToImgTensorF32
from disent.frameworks.vae import BetaVae
from disent.metrics import metric_dci
from disent.metrics import metric_mig
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64
from disent.model.ae import EncoderConv64
from disent.util import is_test_run  # you can ignore and remove this

# make the ground-truth data
gt_data = XYObjectData()
# split the data using built-in functions (no longer ground-truth datasets, but subsets)
data_train, data_val = random_split(
    gt_data,
    [
        int(math.floor(len(gt_data) * 0.7)),
        int(math.ceil(len(gt_data) * 0.3)),
    ],
)
# create the disent datasets
gt_dataset = DisentDataset(gt_data, transform=ToImgTensorF32())  # .is_ground_truth == True
dataset_train = DisentDataset(data_train, transform=ToImgTensorF32())  # .is_ground_truth == False
dataset_val = DisentDataset(data_val, transform=ToImgTensorF32())  # .is_ground_truth == False
# create the data loaders
dataloader_train = DataLoader(dataset=dataset_train, batch_size=4, shuffle=True, num_workers=0)
dataloader_val = DataLoader(dataset=dataset_val, batch_size=4, shuffle=True, num_workers=0)

# create the pytorch lightning system
module: L.LightningModule = BetaVae(
    model=AutoEncoder(
        encoder=EncoderConv64(x_shape=gt_data.x_shape, z_size=6, z_multiplier=2),
        decoder=DecoderConv64(x_shape=gt_data.x_shape, z_size=6),
    ),
    cfg=BetaVae.cfg(optimizer="adam", optimizer_kwargs=dict(lr=1e-3), loss_reduction="mean_sum", beta=4),
)

# train the model
trainer = L.Trainer(logger=False, enable_checkpointing=False, fast_dev_run=is_test_run())
trainer.fit(module, dataloader_train, dataloader_val)

# compute metrics
# - we cannot guarantee which device the representation is on
get_repr = lambda x: module.encode(x.to(module.device))
# - We cannot compute disentanglement metrics over the split datasets `dataset_train` & `dataset_val`
#   because they are no longer ground-truth datasets, we can only use `gt_dataset`
print(metric_dci(gt_dataset, get_repr, num_train=10 if is_test_run() else 1000, num_test=5 if is_test_run() else 500))
print(metric_mig(gt_dataset, get_repr, num_train=20 if is_test_run() else 2000))
