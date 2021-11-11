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
from functools import partial

import pytest
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.dataset.sampling import GroundTruthSingleSampler
from disent.dataset.sampling import GroundTruthPairSampler
from disent.dataset.sampling import GroundTruthTripleSampler
from disent.frameworks.ae import *
from disent.frameworks.vae import *
from disent.model import AutoEncoder
from disent.model.ae import DecoderLinear
from disent.model.ae import EncoderLinear
from disent.dataset.transform import ToImgTensorF32


# ========================================================================= #
# TEST FRAMEWORKS                                                           #
# ========================================================================= #


@pytest.mark.parametrize(['Framework', 'cfg_kwargs', 'Data'], [
    # AE - unsupervised
    (Ae,                   dict(),                                                                      XYObjectData),
    # AE - weakly supervised
    # <n/a>
    # AE - supervised
    (TripletAe,            dict(),                                                                      XYObjectData),
    # VAE - unsupervised
    (Vae,                  dict(),                                                                      XYObjectData),
    (BetaVae,              dict(),                                                                      XYObjectData),
    (DipVae,               dict(),                                                                      XYObjectData),
    (DipVae,               dict(dip_mode='i'),                                                          XYObjectData),
    (InfoVae,              dict(),                                                                      XYObjectData),
    (DfcVae,               dict(),                                                                      XYObjectData),
    (DfcVae,               dict(),                                                                      partial(XYObjectData, rgb=False)),
    (BetaTcVae,            dict(),                                                                      XYObjectData),
    # VAE - weakly supervised
    (AdaVae,               dict(),                                                                      XYObjectData),
    (AdaVae,               dict(ada_average_mode='ml-vae'),                                             XYObjectData),
    (AdaGVaeMinimal,       dict(),                                                                      XYObjectData),
    # VAE - supervised
    (TripletVae,           dict(),                                                                      XYObjectData),
    (TripletVae,           dict(disable_decoder=True, disable_reg_loss=True, disable_posterior_scale=0.5), XYObjectData),
])
def test_frameworks(Framework, cfg_kwargs, Data):
    DataSampler = {
        1: GroundTruthSingleSampler,
        2: GroundTruthPairSampler,
        3: GroundTruthTripleSampler,
    }[Framework.REQUIRED_OBS]

    data = XYObjectData() if (Data is None) else Data()
    dataset = DisentDataset(data, DataSampler(), transform=ToImgTensorF32())
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    framework = Framework(
        model=AutoEncoder(
            encoder=EncoderLinear(x_shape=data.x_shape, z_size=6, z_multiplier=2 if issubclass(Framework, Vae) else 1),
            decoder=DecoderLinear(x_shape=data.x_shape, z_size=6),
        ),
        cfg=Framework.cfg(**cfg_kwargs)
    )

    trainer = pl.Trainer(logger=False, checkpoint_callback=False, max_steps=256, fast_dev_run=True)
    trainer.fit(framework, dataloader)


def test_framework_config_defaults():
    # import torch
    # we test that defaults are working recursively
    assert asdict(BetaVae.cfg()) == dict(
        optimizer='adam',
        optimizer_kwargs=None,
        recon_loss='mse',
        disable_aug_loss=False,
        disable_decoder=False,
        disable_posterior_scale=None,
        disable_rec_loss=False,
        disable_reg_loss=False,
        loss_reduction='mean',
        latent_distribution='normal',
        kl_loss_mode='direct',
        beta=0.003,
    )
    assert asdict(BetaVae.cfg(recon_loss='bce', kl_loss_mode='approx')) == dict(
        optimizer='adam',
        optimizer_kwargs=None,
        recon_loss='bce',
        disable_aug_loss=False,
        disable_decoder=False,
        disable_posterior_scale=None,
        disable_rec_loss=False,
        disable_reg_loss=False,
        loss_reduction='mean',
        latent_distribution='normal',
        kl_loss_mode='approx',
        beta=0.003,
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
