import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from disent.data.groundtruth import XYSquaresData, GroundTruthData
from disent.dataset.groundtruth import GroundTruthDataset
from disent.frameworks.vae.unsupervised import BetaVae
from disent.model.ae import EncoderConv64, DecoderConv64, GaussianAutoEncoder
from disent.transform import ToStandardisedTensor

data: GroundTruthData = XYSquaresData()
dataset: Dataset = GroundTruthDataset(data, transform=ToStandardisedTensor())
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

module: pl.LightningModule = BetaVae(
    make_optimizer_fn=lambda params: Adam(params, lr=1e-3),
    make_model_fn=lambda: GaussianAutoEncoder(
        encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
        decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
    ),
    cfg=BetaVae.cfg(beta=4)
)

trainer = pl.Trainer(logger=False, checkpoint_callback=False)
trainer.fit(module, dataloader)