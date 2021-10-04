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
from torch.optim import Adam
from torch.utils.data import DataLoader

from disent.dataset.data import XYObjectData
from disent.dataset import DisentDataset
from disent.dataset.sampling import GroundTruthSingleSampler
from disent.dataset.sampling import GroundTruthPairSampler
from disent.dataset.sampling import GroundTruthTripleSampler
from disent.frameworks.ae import *
from disent.frameworks.ae.experimental import *   # pragma: delete-on-release
from disent.frameworks.vae import *
from disent.frameworks.vae.experimental import *  # pragma: delete-on-release
from disent.model import AutoEncoder
from disent.model.ae import DecoderTest
from disent.model.ae import EncoderTest
from disent.nn.transform import ToStandardisedTensor


# ========================================================================= #
# TEST FRAMEWORKS                                                           #
# ========================================================================= #


@pytest.mark.parametrize(['Framework', 'cfg_kwargs', 'Data'], [
    # AE - unsupervised
    (Ae,                   dict(),                                                                      XYObjectData),
    # AE - unsupervised - EXP                                                                                           # pragma: delete-on-release
    (DataOverlapTripletAe, dict(overlap_mine_triplet_mode='hard_neg'),                                  XYObjectData),  # pragma: delete-on-release
    # AE - weakly supervised
    # <n/a>
    # AE - weakly supervised - EXP                                                                                      # pragma: delete-on-release
    (AdaAe,                dict(),                                                                      XYObjectData),  # pragma: delete-on-release
    # AE - supervised
    (TripletAe,            dict(),                                                                      XYObjectData),
    # AE - supervised - EXP                                                                                             # pragma: delete-on-release
    (AdaNegTripletAe,      dict(),                                                                      XYObjectData),  # pragma: delete-on-release
    # VAE - unsupervised
    (Vae,                  dict(),                                                                      XYObjectData),
    (BetaVae,              dict(),                                                                      XYObjectData),
    (DipVae,               dict(),                                                                      XYObjectData),
    (DipVae,               dict(dip_mode='i'),                                                          XYObjectData),
    (InfoVae,              dict(),                                                                      XYObjectData),
    (DfcVae,               dict(),                                                                      XYObjectData),
    (DfcVae,               dict(),                                                                      partial(XYObjectData, rgb=False)),
    (BetaTcVae,            dict(),                                                                      XYObjectData),
    # VAE - unsupervised - EXP                                                                                          # pragma: delete-on-release
    (DataOverlapTripletVae,dict(overlap_mine_triplet_mode='none'),                                      XYObjectData),  # pragma: delete-on-release
    (DataOverlapTripletVae,dict(overlap_mine_triplet_mode='semi_hard_neg'),                             XYObjectData),  # pragma: delete-on-release
    (DataOverlapTripletVae,dict(overlap_mine_triplet_mode='hard_neg'),                                  XYObjectData),  # pragma: delete-on-release
    (DataOverlapTripletVae,dict(overlap_mine_triplet_mode='hard_pos'),                                  XYObjectData),  # pragma: delete-on-release
    (DataOverlapTripletVae,dict(overlap_mine_triplet_mode='easy_pos'),                                  XYObjectData),  # pragma: delete-on-release
    (DataOverlapRankVae,   dict(),                                                                      XYObjectData),  # pragma: delete-on-release
    # VAE - weakly supervised
    (AdaVae,               dict(),                                                                      XYObjectData),
    (AdaVae,               dict(ada_average_mode='ml-vae'),                                             XYObjectData),
    # VAE - weakly supervised - EXP                                                                                     # pragma: delete-on-release
    (SwappedTargetAdaVae,  dict(swap_chance=1.0),                                                       XYObjectData),  # pragma: delete-on-release
    (SwappedTargetBetaVae, dict(swap_chance=1.0),                                                       XYObjectData),  # pragma: delete-on-release
    (AugPosTripletVae,     dict(),                                                                      XYObjectData),  # pragma: delete-on-release
    # VAE - supervised
    (TripletVae,           dict(),                                                                      XYObjectData),
    (TripletVae,           dict(disable_decoder=True, disable_reg_loss=True, disable_posterior_scale=0.5), XYObjectData),
    # VAE - supervised - EXP                                                                                            # pragma: delete-on-release
    (BoundedAdaVae,        dict(),                                                                      XYObjectData),  # pragma: delete-on-release
    (GuidedAdaVae,         dict(),                                                                      XYObjectData),  # pragma: delete-on-release
    (GuidedAdaVae,         dict(gada_anchor_ave_mode='thresh'),                                         XYObjectData),  # pragma: delete-on-release
    (TripletBoundedAdaVae, dict(),                                                                      XYObjectData),  # pragma: delete-on-release
    (TripletGuidedAdaVae,  dict(),                                                                      XYObjectData),  # pragma: delete-on-release
    (AdaTripletVae,        dict(),                                                                      XYObjectData),  # pragma: delete-on-release
    (AdaAveTripletVae,     dict(adat_share_mask_mode='posterior'),                                      XYObjectData),  # pragma: delete-on-release
    (AdaAveTripletVae,     dict(adat_share_mask_mode='sample'),                                         XYObjectData),  # pragma: delete-on-release
    (AdaAveTripletVae,     dict(adat_share_mask_mode='sample_each'),                                    XYObjectData),  # pragma: delete-on-release
])
def test_frameworks(Framework, cfg_kwargs, Data):
    DataSampler = {
        1: GroundTruthSingleSampler,
        2: GroundTruthPairSampler,
        3: GroundTruthTripleSampler,
    }[Framework.REQUIRED_OBS]

    data = XYObjectData() if (Data is None) else Data()
    dataset = DisentDataset(data, DataSampler(), transform=ToStandardisedTensor())
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    framework = Framework(
        model=AutoEncoder(
            encoder=EncoderTest(x_shape=data.x_shape, z_size=6, z_multiplier=2 if issubclass(Framework, Vae) else 1),
            decoder=DecoderTest(x_shape=data.x_shape, z_size=6),
        ),
        cfg=Framework.cfg(**cfg_kwargs)
    )

    trainer = pl.Trainer(logger=False, checkpoint_callback=False, max_steps=256, fast_dev_run=True)
    trainer.fit(framework, dataloader)


def test_framework_config_defaults():
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
