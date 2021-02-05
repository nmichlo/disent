import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import DataLoader
from disent.data.groundtruth import XYObjectData
from disent.dataset.groundtruth import GroundTruthDataset
from disent.frameworks.vae.unsupervised import BetaVae
from disent.metrics import metric_dci, metric_mig
from disent.model.ae import EncoderConv64, DecoderConv64, GaussianAutoEncoder
from disent.transform import ToStandardisedTensor
from disent.util import is_test_run, test_run_int

data = XYObjectData()
dataset = GroundTruthDataset(data, transform=ToStandardisedTensor())
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

def make_vae(beta):
    return BetaVae(
        make_optimizer_fn=lambda params: Adam(params, lr=5e-3),
        make_model_fn=lambda: GaussianAutoEncoder(
            encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
            decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
        ),
        cfg=BetaVae.cfg(beta=beta)
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