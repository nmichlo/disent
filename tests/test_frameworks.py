#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

from dataclasses import asdict

import pytest
import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import DataLoader

from disent.data.groundtruth import XYObjectData
from disent.dataset.groundtruth import GroundTruthDataset
from disent.dataset.groundtruth import GroundTruthDatasetPairs
from disent.dataset.groundtruth import GroundTruthDatasetTriples
from disent.frameworks.ae.unsupervised import *
from disent.frameworks.vae.supervised import *
from disent.frameworks.vae.unsupervised import *
from disent.frameworks.vae.weaklysupervised import *
from disent.model.ae import AutoEncoder
from disent.model.ae import DecoderConv64
from disent.model.ae import EncoderConv64
from disent.transform import ToStandardisedTensor


# ========================================================================= #
# TEST FRAMEWORKS                                                           #
# ========================================================================= #


@pytest.mark.parametrize(['z_multiplier', 'DataWrapper', 'Framework', 'cfg_kwargs'], [
    # AE - unsupervised
    (1, GroundTruthDataset,        AE,                   dict()),
    # VAE - unsupervised
    (2, GroundTruthDataset,        Vae,                  dict()),
    (2, GroundTruthDataset,        BetaVae,              dict()),
    (2, GroundTruthDataset,        DfcVae,               dict(recon_loss='bce')),
    # VAE - weakly supervised
    (2, GroundTruthDatasetPairs,   AdaVae,               dict()),
    (2, GroundTruthDatasetPairs,   SwappedTargetAdaVae,  dict()),
    (2, GroundTruthDatasetPairs,   SwappedTargetBetaVae, dict()),
    (2, GroundTruthDatasetPairs,   AugPosTripletVae,     dict()),
    # VAE - supervised
    (2, GroundTruthDatasetTriples, TripletVae,           dict()),
    (2, GroundTruthDatasetTriples, BoundedAdaVae,        dict()),
    (2, GroundTruthDatasetTriples, GuidedAdaVae,         dict()),
    (2, GroundTruthDatasetTriples, TripletBoundedAdaVae, dict()),
    (2, GroundTruthDatasetTriples, TripletGuidedAdaVae,  dict()),
    (2, GroundTruthDatasetTriples, AdaTripletVae,        dict()),
])
def test_frameworks(z_multiplier, DataWrapper, Framework, cfg_kwargs):

    data = XYObjectData()
    dataset = DataWrapper(data, transform=ToStandardisedTensor())
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    framework = Framework(
        make_optimizer_fn=lambda params: Adam(params, lr=1e-3),
        make_model_fn=lambda: AutoEncoder(
            encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=z_multiplier),
            decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
        ),
        cfg=Framework.cfg(**cfg_kwargs)
    )

    trainer = pl.Trainer(logger=False, checkpoint_callback=False, max_steps=256, fast_dev_run=True)
    trainer.fit(framework, dataloader)


def test_framrwork_config_defaults():
    # we test that defaults are working recursively
    assert asdict(Vae.cfg(kl_loss_mode='approx')) == dict(
        recon_loss='mse',
        loss_reduction='batch_mean',
        latent_distribution='normal',
        kl_loss_mode='approx'
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
