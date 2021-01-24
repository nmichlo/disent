import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from disent.data.groundtruth import XYSquaresData, GroundTruthData
from disent.dataset.groundtruth import GroundTruthDataset
from disent.frameworks.ae.unsupervised import AE
from disent.model.ae import EncoderConv64, DecoderConv64, AutoEncoder
from disent.transform import ToStandardisedTensor

data: GroundTruthData = XYSquaresData()
dataset: Dataset = GroundTruthDataset(data, transform=ToStandardisedTensor())
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

module: pl.LightningModule = AE(
    make_optimizer_fn=lambda params: Adam(params, lr=1e-3),
    make_model_fn=lambda: AutoEncoder(
        encoder=EncoderConv64(x_shape=data.x_shape, z_size=6),
        decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
    ),
    cfg=AE.cfg()
)

trainer = pl.Trainer(logger=False, checkpoint_callback=False)
trainer.fit(module, dataloader)