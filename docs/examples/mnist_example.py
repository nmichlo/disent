import os

import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from disent.dataset import DisentDataset
from disent.dataset.sampling import RandomSampler
from disent.dataset.transform import ToImgTensorF32
from disent.frameworks.vae import AdaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderFC
from disent.model.ae import EncoderFC


# modify the mnist dataset to only return images, not labels
class MNIST(datasets.MNIST):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


# make mnist dataset -- adjust num_samples here to match framework. TODO: add tests that can fail with a warning -- dataset downloading is not always reliable
data_folder = os.path.abspath(os.path.join(__file__, "../data/dataset"))
dataset_train = DisentDataset(
    MNIST(data_folder, train=True, download=True, transform=ToImgTensorF32()), sampler=RandomSampler(num_samples=2)
)
dataset_test = MNIST(data_folder, train=False, download=True, transform=ToImgTensorF32())

# create the dataloaders
# - if you use `num_workers != 0` in the DataLoader, the make sure to
#   wrap `trainer.fit` with `if __name__ == '__main__': ...`
dataloader_train = DataLoader(dataset=dataset_train, batch_size=128, shuffle=True, num_workers=0)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, shuffle=True, num_workers=0)

# create the model
module = AdaVae(
    model=AutoEncoder(
        encoder=EncoderFC(x_shape=(1, 28, 28), z_size=9, z_multiplier=2),
        decoder=DecoderFC(x_shape=(1, 28, 28), z_size=9),
    ),
    cfg=AdaVae.cfg(
        optimizer="adam",
        optimizer_kwargs=dict(lr=1e-3),
        beta=4,
        recon_loss="mse",
        loss_reduction="mean_sum",  # "mean_sum" is the traditional loss reduction mode, rather than "mean"
    ),
)

# train the model
trainer = L.Trainer(
    logger=False, enable_checkpointing=False, max_steps=2048
)  # callbacks=[VaeLatentCycleLoggingCallback(every_n_steps=250, plt_show=True)]
trainer.fit(module, dataloader_train)

# move back to gpu & manually encode some observation
for xs in tqdm(dataloader_test, desc="Custom Evaluation"):
    zs = module.encode(xs.to(module.device))
