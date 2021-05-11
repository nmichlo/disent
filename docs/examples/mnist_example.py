# TODO: update this example for automatic testing

from collections import Sequence

import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from disent.dataset.random import RandomDataset
from disent.frameworks.vae.weaklysupervised import AdaVae
from disent.model.ae import AutoEncoder
from disent.model.ae import DecoderConv64Alt
from disent.model.ae import EncoderConv64Alt
from disent.transform import ToStandardisedTensor


class MNIST(datasets.MNIST, Sequence):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


# make mnist dataset -- adjust num_samples here to match framework
dataset_train = RandomDataset(MNIST('./data/dataset/mnist', train=True,  download=True, transform=ToStandardisedTensor(size=64)), num_samples=2)
dataset_test  =               MNIST('./data/dataset/mnist', train=False, download=True, transform=ToStandardisedTensor(size=64))

# create the dataloaders
dataloader_train = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
dataloader_test  = DataLoader(dataset=dataset_test,  batch_size=64, shuffle=True)

# create the model
module = AdaVae(
    make_optimizer_fn=lambda params: Adam(params, lr=1e-3),
    make_model_fn=lambda: AutoEncoder(
        encoder=EncoderConv64Alt(x_shape=(1, 64, 64), z_size=9, z_multiplier=2),
        decoder=DecoderConv64Alt(x_shape=(1, 64, 64), z_size=9),
    ),
    cfg=AdaVae.cfg(beta=4, recon_loss='mse', loss_reduction='mean_sum')  # "mean_sum" is the traditional reduction, rather than "mean"
)

# train model
trainer = pl.Trainer(logger=False, checkpoint_callback=False, max_steps=65535, gpus=1)  # callbacks=[VaeLatentCycleLoggingCallback(every_n_steps=250, plt_show=True)]
trainer.fit(module, dataloader_train)

# move back to gpu & manually encode some observation
module.cuda()
for xs in tqdm(dataloader_test, desc='Custom Evaluation'):
    zs = module.encode(xs.to(module.device))
