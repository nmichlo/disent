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

import logging
from typing import Callable

import numpy as np
import torch

from disent.util import to_numpy
from disent.util.visualize import vis_util


log = logging.getLogger(__name__)


# ========================================================================= #
# Visualise Latent Cycle - Modes                                            #
# ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
# Copyright 2018 The DisentanglementLib Authors. All rights reserved.       #
# Licensed under the Apache License, Version 2.0                            #
# https://github.com/google-research/disentanglement_lib                    #
# Copyright applies to this subsection only.                                #
# ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
# CHANGES:                                                                  #
# - extracted from original code                                            #
# - was not split into functions in this was                                #
# - TODO: convert these functions to torch.Tensors                          #
# ========================================================================= #


def _z_std_gaussian_cycle(base_z, z_means, z_logvars, z_idx, num_frames):
    # Cycle through quantiles of a standard Gaussian.
    zs = np.repeat(np.expand_dims(base_z, 0), num_frames, axis=0)
    zs[:, z_idx] = vis_util.cycle_gaussian(base_z[z_idx], num_frames, loc=0, scale=1)
    return zs


def _z_fitted_gaussian_cycle(base_z, z_means, z_logvars, z_idx, num_frames):
    # Cycle through quantiles of a fitted Gaussian.
    zs = np.repeat(np.expand_dims(base_z, 0), num_frames, axis=0)
    loc = np.mean(z_means[:, z_idx])
    total_variance = np.mean(np.exp(z_logvars[:, z_idx])) + np.var(z_means[:, z_idx])
    zs[:, z_idx] = vis_util.cycle_gaussian(base_z[z_idx], num_frames, loc=loc, scale=np.sqrt(total_variance))
    return zs


def _z_fixed_interval_cycle(base_z, z_means, z_logvars, z_idx, num_frames):
    # Cycle through [-2, 2] interval.
    zs = np.repeat(np.expand_dims(base_z, 0), num_frames, axis=0)
    zs[:, z_idx] = vis_util.cycle_interval(base_z[z_idx], num_frames, -2., 2.)
    return zs


def _z_conf_interval_cycle(base_z, z_means, z_logvars, z_idx, num_frames):
    # Cycle linearly through +-2 std dev of a fitted Gaussian.
    zs = np.repeat(np.expand_dims(base_z, 0), num_frames, axis=0)
    loc = np.mean(z_means[:, z_idx])
    total_variance = np.mean(np.exp(z_logvars[:, z_idx])) + np.var(z_means[:, z_idx])
    scale = np.sqrt(total_variance)
    zs[:, z_idx] = vis_util.cycle_interval(base_z[z_idx], num_frames, loc - 2. * scale, loc + 2. * scale)
    return zs


def _z_minmax_interval_cycle(base_z, z_means, z_logvars, z_idx, num_frames):
    # Cycle linearly through minmax of a fitted Gaussian.
    zs = np.repeat(np.expand_dims(base_z, 0), num_frames, axis=0)
    zs[:, z_idx] = vis_util.cycle_interval(base_z[z_idx], num_frames, np.min(z_means[:, z_idx]), np.max(z_means[:, z_idx]))
    return zs


_LATENT_CYCLE_MODES_MAP = {
    'std_gaussian_cycle': _z_std_gaussian_cycle,
    'fitted_gaussian_cycle': _z_fitted_gaussian_cycle,
    'fixed_interval_cycle': _z_fixed_interval_cycle,
    'conf_interval_cycle': _z_conf_interval_cycle,
    'minmax_interval_cycle': _z_minmax_interval_cycle,
}


def make_latent_zs_cycle(
    base_z: torch.Tensor,
    z_means: torch.Tensor,
    z_logvars: torch.Tensor,
    z_idx: int,
    num_frames: int,
    mode: str = 'minmax_interval_cycle',
) -> torch.Tensor:
    # get mode
    if mode not in _LATENT_CYCLE_MODES_MAP:
        raise KeyError(f'Unsupported mode: {repr(mode)} not in {set(_LATENT_CYCLE_MODES_MAP)}')
    z_gen_func = _LATENT_CYCLE_MODES_MAP[mode]
    # checks
    assert base_z.ndim == 1
    assert base_z.shape == z_means.shape[1:]
    assert z_means.ndim == z_logvars.ndim == 2
    assert z_means.shape == z_logvars.shape
    assert len(z_means) > 1, f'not enough representations to average, number of z_means should be greater than 1, got: {z_means.shape}'
    # make cycle
    z_cycle = z_gen_func(to_numpy(base_z), to_numpy(z_means), to_numpy(z_logvars), z_idx, num_frames)
    return torch.from_numpy(z_cycle)


# ========================================================================= #
# Visualise Latent Cycles                                                   #
# ========================================================================= #


# TODO: this should be moved into the VAE and AE classes
def make_decoded_latent_cycles(
    decoder_func: Callable[[torch.Tensor], torch.Tensor],
    z_means: torch.Tensor,
    z_logvars: torch.Tensor,
    mode: str = 'minmax_interval_cycle',
    num_animations: int = 4,
    num_frames: int = 20,
    decoder_device=None,
) -> torch.Tensor:
    # generate multiple latent traversal visualisations
    animations = []
    for i in range(num_animations):
        frames = []
        for z_idx in range(z_means.shape[1]):
            z = make_latent_zs_cycle(z_means[i], z_means, z_logvars, z_idx, num_frames, mode=mode)
            z = torch.as_tensor(z, device=decoder_device)
            frames.append(decoder_func(z))
        animations.append(torch.stack(frames, dim=0))
    animations = torch.stack(animations, dim=0)
    # return everything
    return animations  # (num_animations, z_size, num_frames, C, H, W)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
