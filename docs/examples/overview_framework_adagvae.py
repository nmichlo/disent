import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import DataLoader
from disent.dataset import DisentDataset
from disent.dataset.data import XYSquaresData
from disent.dataset.sampling import GroundTruthPairOrigSampler
from disent.frameworks.vae import AdaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64, EncoderConv64
from disent.nn.transform import ToStandardisedTensor
from disent.util import is_test_run  # you can ignore and remove this


# prepare the data
data = XYSquaresData()
dataset = DisentDataset(data, GroundTruthPairOrigSampler(), transform=ToStandardisedTensor())
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# create the pytorch lightning system
module: pl.LightningModule = AdaVae(
    make_optimizer_fn=lambda params: Adam(params, lr=1e-3),
    make_model_fn=lambda: AutoEncoder(
        encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
        decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
    ),
    cfg=AdaVae.cfg(loss_reduction='mean_sum', beta=4, ada_average_mode='gvae', ada_thresh_mode='kl')
)

# train the model
trainer = pl.Trainer(logger=False, checkpoint_callback=False, fast_dev_run=is_test_run())
trainer.fit(module, dataloader)
