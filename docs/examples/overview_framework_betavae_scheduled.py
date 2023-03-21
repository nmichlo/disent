import lightning as L
from torch.utils.data import DataLoader

from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.dataset.transform import ToImgTensorF32
from disent.frameworks.vae import BetaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64
from disent.model.ae import EncoderConv64
from disent.schedule import CyclicSchedule
from disent.util import is_test_run  # you can ignore and remove this

# prepare the data
data = XYObjectData()
dataset = DisentDataset(data, transform=ToImgTensorF32())
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# create the pytorch lightning system
module: L.LightningModule = BetaVae(
    model=AutoEncoder(
        encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
        decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
    ),
    cfg=BetaVae.cfg(optimizer="adam", optimizer_kwargs=dict(lr=1e-3), loss_reduction="mean_sum", beta=4),
)

# register the scheduler with the DisentFramework
# - cyclic scheduler from: https://arxiv.org/abs/1903.10145
module.register_schedule(
    "beta",
    CyclicSchedule(
        period=1024,  # repeat every: trainer.global_step % period
    ),
)

# train the model
trainer = L.Trainer(logger=False, enable_checkpointing=False, fast_dev_run=is_test_run())
trainer.fit(module, dataloader)
