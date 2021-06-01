import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from disent.data.groundtruth import GroundTruthData, XYSquaresData
from disent.dataset.groundtruth import GroundTruthDatasetOrigWeakPairs
from disent.frameworks.vae import AdaVae
from disent.nn.model.ae import AutoEncoder, DecoderConv64, EncoderConv64
from disent.nn.transform import ToStandardisedTensor
from disent.util import is_test_run


data: GroundTruthData = XYSquaresData()
dataset: Dataset = GroundTruthDatasetOrigWeakPairs(data, transform=ToStandardisedTensor())
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

module: pl.LightningModule = AdaVae(
    make_optimizer_fn=lambda params: Adam(params, lr=1e-3),
    make_model_fn=lambda: AutoEncoder(
        encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
        decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
    ),
    cfg=AdaVae.cfg(beta=4, ada_average_mode='gvae', ada_thresh_mode='kl')
)

trainer = pl.Trainer(logger=False, checkpoint_callback=False, fast_dev_run=is_test_run())
trainer.fit(module, dataloader)