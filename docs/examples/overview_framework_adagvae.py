import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from disent.data.groundtruth import XYSquaresData, GroundTruthData
from disent.dataset.groundtruth import GroundTruthDatasetOrigWeakPairs
from disent.frameworks.vae.weaklysupervised import AdaVae
from disent.model.ae import EncoderConv64, DecoderConv64, GaussianAutoEncoder
from disent.transform import ToStandardisedTensor

data: GroundTruthData = XYSquaresData()
dataset: Dataset = GroundTruthDatasetOrigWeakPairs(data, transform=ToStandardisedTensor())
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

module: pl.LightningModule = AdaVae(
    make_optimizer_fn=lambda params: Adam(params, lr=1e-3),
    make_model_fn=lambda: GaussianAutoEncoder(
        encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
        decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
    ),
    cfg=AdaVae.cfg(beta=4, average_mode='gvae', symmetric_kl=False)
)

trainer = pl.Trainer(logger=False, checkpoint_callback=False)
trainer.fit(module, dataloader)