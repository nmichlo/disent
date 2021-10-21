import pytorch_lightning as pl
from torch.utils.data import DataLoader
from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.frameworks.vae import BetaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64, EncoderConv64
from disent.dataset.transform import ToImgTensorF32
from disent.metrics import metric_dci, metric_mig
from disent.util import is_test_run

data = XYObjectData()
dataset = DisentDataset(data, transform=ToImgTensorF32(), augment=None)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

def make_vae(beta):
    return BetaVae(
        model=AutoEncoder(
            encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
            decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
        ),
        cfg=BetaVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=1e-3), beta=beta)
    )

def train(module):
    trainer = pl.Trainer(logger=False, checkpoint_callback=False, max_steps=256, fast_dev_run=is_test_run())
    trainer.fit(module, dataloader)

    # we cannot guarantee which device the representation is on
    get_repr = lambda x: module.encode(x.to(module.device))

    # evaluate
    return {
        **metric_dci(dataset, get_repr, num_train=10 if is_test_run() else 1000, num_test=5 if is_test_run() else 500, boost_mode='sklearn'),
        **metric_mig(dataset, get_repr, num_train=20 if is_test_run() else 2000),
    }

a_results = train(make_vae(beta=4))
b_results = train(make_vae(beta=0.01))

print('beta=4:   ', a_results)
print('beta=0.01:', b_results)
