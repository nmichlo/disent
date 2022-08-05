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

import pickle
from dataclasses import asdict
from functools import partial

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.dataset.sampling import GroundTruthSingleSampler
from disent.dataset.sampling import GroundTruthPairSampler
from disent.dataset.sampling import GroundTruthTripleSampler
from disent.dataset.sampling import RandomSampler
from disent.frameworks.ae import *
from disent.frameworks.vae import *
from disent.model import AutoEncoder
from disent.model.ae import DecoderLinear
from disent.model.ae import EncoderLinear
from disent.dataset.transform import ToImgTensorF32


# ========================================================================= #
# TEST FRAMEWORKS                                                           #
# ========================================================================= #
from disent.util.seeds import seed
from disent.util.seeds import TempNumpySeed
from docs.examples.extend_experiment.code.weaklysupervised__si_adavae import SwappedInputAdaVae
from docs.examples.extend_experiment.code.weaklysupervised__si_betavae import SwappedInputBetaVae


_TEST_FRAMEWORKS = [
    # AE - unsupervised
    (Ae,                   dict(),                                                                      XYObjectData),
    # AE - unsupervised - EXP
    (DataOverlapTripletAe, dict(overlap_mine_triplet_mode='hard_neg'),                                  XYObjectData),
    # AE - weakly supervised
    # <n/a>
    # AE - weakly supervised - EXP
    (AdaAe,                dict(),                                                                      XYObjectData),
    # AE - supervised
    (TripletAe,            dict(),                                                                      XYObjectData),
    # AE - supervised - EXP
    (AdaNegTripletAe,      dict(),                                                                      XYObjectData),
    # VAE - unsupervised
    (Vae,                  dict(),                                                                      XYObjectData),
    (BetaVae,              dict(),                                                                      XYObjectData),
    (DipVae,               dict(),                                                                      XYObjectData),
    (DipVae,               dict(dip_mode='i'),                                                          XYObjectData),
    (InfoVae,              dict(),                                                                      XYObjectData),
    (DfcVae,               dict(),                                                                      XYObjectData),
    (DfcVae,               dict(),                                                                      partial(XYObjectData, rgb=False)),
    (BetaTcVae,            dict(),                                                                      XYObjectData),
    # VAE - unsupervised - EXP
    (DataOverlapTripletVae,dict(overlap_mine_triplet_mode='none'),                                      XYObjectData),
    (DataOverlapTripletVae,dict(overlap_mine_triplet_mode='semi_hard_neg'),                             XYObjectData),
    (DataOverlapTripletVae,dict(overlap_mine_triplet_mode='hard_neg'),                                  XYObjectData),
    (DataOverlapTripletVae,dict(overlap_mine_triplet_mode='hard_pos'),                                  XYObjectData),
    (DataOverlapTripletVae,dict(overlap_mine_triplet_mode='easy_pos'),                                  XYObjectData),
    # VAE - weakly supervised
    (AdaVae,               dict(),                                                                      XYObjectData),
    (AdaVae,               dict(ada_average_mode='ml-vae'),                                             XYObjectData),
    (AdaGVaeMinimal,       dict(),                                                                      XYObjectData),
    # VAE - weakly supervised - EXP
    (SwappedInputAdaVae,  dict(swap_chance=1.0),                                                       XYObjectData),
    (SwappedInputBetaVae, dict(swap_chance=1.0),                                                       XYObjectData),
    # VAE - supervised
    (TripletVae,           dict(),                                                                      XYObjectData),
    (TripletVae,           dict(detach_decoder=True, disable_reg_loss=True),                            XYObjectData),
]


@pytest.mark.parametrize(['Framework', 'cfg_kwargs', 'Data'], _TEST_FRAMEWORKS)
def test_frameworks(Framework, cfg_kwargs, Data):
    DataSampler = {
        1: GroundTruthSingleSampler,
        2: GroundTruthPairSampler,
        3: GroundTruthTripleSampler,
    }[Framework.REQUIRED_OBS]

    data = XYObjectData() if (Data is None) else Data()
    dataset = DisentDataset(data, DataSampler(), transform=ToImgTensorF32())
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

    framework = Framework(
        model=AutoEncoder(
            encoder=EncoderLinear(x_shape=data.x_shape, z_size=6, z_multiplier=2 if issubclass(Framework, Vae) else 1),
            decoder=DecoderLinear(x_shape=data.x_shape, z_size=6),
        ),
        cfg=Framework.cfg(**cfg_kwargs)
    )

    # test pickling before training
    pickle.dumps(framework)

    # train!
    trainer = pl.Trainer(logger=False, checkpoint_callback=False, max_steps=256, fast_dev_run=True)
    trainer.fit(framework, dataloader)

    # test pickling after training, something may have changed!
    pickle.dumps(framework)


@pytest.mark.parametrize(['Framework', 'cfg_kwargs', 'Data'], _TEST_FRAMEWORKS)
def test_framework_pickling(Framework, cfg_kwargs, Data):
    framework = Framework(
        model=AutoEncoder(
            encoder=EncoderLinear(x_shape=(64, 64, 3), z_size=6, z_multiplier=2 if issubclass(Framework, Vae) else 1),
            decoder=DecoderLinear(x_shape=(64, 64, 3), z_size=6),
        ),
        cfg=Framework.cfg(**cfg_kwargs)
    )
    # test pickling!
    pickle.dumps(framework)


def test_framework_config_defaults():
    # import torch
    # we test that defaults are working recursively
    assert asdict(BetaVae.cfg()) == dict(
        optimizer='adam',
        optimizer_kwargs=None,
        recon_loss='mse',
        disable_aug_loss=False,
        detach_decoder=False,
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
        detach_decoder=False,
        disable_rec_loss=False,
        disable_reg_loss=False,
        loss_reduction='mean',
        latent_distribution='normal',
        kl_loss_mode='approx',
        beta=0.003,
    )


def test_ada_vae_similarity():

    seed(42)

    data = XYObjectData()
    dataset = DisentDataset(data, sampler=RandomSampler(num_samples=2), transform=ToImgTensorF32())
    dataloader = DataLoader(dataset, batch_size=3, num_workers=0)

    model = AutoEncoder(
        encoder=EncoderLinear(x_shape=data.x_shape, z_size=25, z_multiplier=2),
        decoder=DecoderLinear(x_shape=data.x_shape, z_size=25, z_multiplier=1),
    )

    adavae0 = AdaGVaeMinimal(model=model, cfg=AdaGVaeMinimal.cfg())
    adavae1 = AdaVae(model=model, cfg=AdaVae.cfg())
    adavae2 = AdaVae(model=model, cfg=AdaVae.cfg(
        ada_average_mode='gvae',
        ada_thresh_mode='symmetric_kl',
        ada_thresh_ratio=0.5,
    ))

    batch = next(iter(dataloader))

    # TODO: add a TempNumpySeed equivalent for torch
    seed(777)
    result0a = adavae0.do_training_step(batch, 0)
    seed(777)
    result0b = adavae0.do_training_step(batch, 0)
    assert torch.allclose(result0a, result0b), f'{result0a} does not match {result0b}'

    seed(777)
    result1a = adavae1.do_training_step(batch, 0)
    seed(777)
    result1b = adavae1.do_training_step(batch, 0)
    assert torch.allclose(result1a, result1b), f'{result1a} does not match {result1b}'

    seed(777)
    result2a = adavae2.do_training_step(batch, 0)
    seed(777)
    result2b = adavae2.do_training_step(batch, 0)
    assert torch.allclose(result2a, result2b), f'{result2a} does not match {result2b}'

    # check similar
    assert torch.allclose(result0a, result1a), f'{result0a} does not match {result1a}'
    assert torch.allclose(result1a, result2a), f'{result1a} does not match {result2a}'


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
