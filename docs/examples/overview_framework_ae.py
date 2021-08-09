import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import DataLoader
from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.dataset.sampling import SingleSampler
from disent.frameworks.ae import Ae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64, EncoderConv64
from disent.nn.transform import ToStandardisedTensor
from disent.util import is_test_run  # you can ignore and remove this


# prepare the data
data = XYObjectData()
dataset = DisentDataset(data, transform=ToStandardisedTensor())
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# create the pytorch lightning system
module: pl.LightningModule = Ae(
    model=AutoEncoder(
        encoder=EncoderConv64(x_shape=data.x_shape, z_size=6),
        decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
    ),
    cfg=Ae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=1e-3), loss_reduction='mean_sum')
)

# train the model
trainer = pl.Trainer(logger=False, checkpoint_callback=False, fast_dev_run=is_test_run())
trainer.fit(module, dataloader)
