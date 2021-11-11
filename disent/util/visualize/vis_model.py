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
import numpy as np
import torch

from disent.util import to_numpy
from disent.util.visualize import vis_util
from disent.util.visualize.vis_util import make_animated_image_grid
from disent.util.visualize.vis_util import reconstructions_to_images


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


# ========================================================================= #
# Visualise Latent Cycles                                                   #
# ========================================================================= #


# TODO: this function should not convert output to images, it should just be
#       left as is. That way we don't need to pass in the recon_min and recon_max
def latent_cycle(decoder_func, z_means, z_logvars, mode='fixed_interval_cycle', num_animations=4, num_frames=20, decoder_device=None, recon_min=0., recon_max=1.) -> np.ndarray:
    assert len(z_means) > 1 and len(z_logvars) > 1, 'not enough samples to average'
    # convert
    z_means, z_logvars = to_numpy(z_means), to_numpy(z_logvars)
    # get mode
    if mode not in _LATENT_CYCLE_MODES_MAP:
        raise KeyError(f'Unsupported mode: {repr(mode)} not in {set(_LATENT_CYCLE_MODES_MAP)}')
    z_gen_func = _LATENT_CYCLE_MODES_MAP[mode]
    animations = []
    for i, base_z in enumerate(z_means[:num_animations]):
        frames = []
        for j in range(z_means.shape[1]):
            z = z_gen_func(base_z, z_means, z_logvars, j, num_frames)
            z = torch.as_tensor(z, device=decoder_device)
            frames.append(decoder_func(z))
        animations.append(frames)
    return reconstructions_to_images(animations, recon_min=recon_min, recon_max=recon_max)


def latent_cycle_grid_animation(decoder_func, z_means, z_logvars, mode='fixed_interval_cycle', num_frames=21, pad=4, border=True, bg_color=0.5, decoder_device=None, tensor_style_channels=True, always_rgb=True, return_stills=False, to_uint8=False, recon_min=0., recon_max=1.) -> np.ndarray:
    # produce latent cycle animation & merge frames
    stills = latent_cycle(decoder_func, z_means, z_logvars, mode=mode, num_animations=1, num_frames=num_frames, decoder_device=decoder_device, recon_min=recon_min, recon_max=recon_max)[0]
    # check and add missing channel if needed (convert greyscale to rgb images)
    if always_rgb:
        assert stills.shape[-1] in {1, 3}, f'Invalid number of image channels: {stills.shape} ({stills.shape[-1]})'
        if stills.shape[-1] == 1:
            stills = np.repeat(stills, 3, axis=-1)
    # create animation
    frames = make_animated_image_grid(stills, pad=pad, border=border, bg_color=bg_color)
    # move channels to end
    if tensor_style_channels:
        if return_stills:
            stills = np.transpose(stills, [0, 1, 4, 2, 3])
        frames = np.transpose(frames, [0, 3, 1, 2])
    # convert to uint8
    if to_uint8:
        if return_stills:
            stills = np.clip(stills*255, 0, 255).astype('uint8')
        frames = np.clip(frames*255, 0, 255).astype('uint8')
    # done!
    if return_stills:
        return frames, stills
    return frames


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
