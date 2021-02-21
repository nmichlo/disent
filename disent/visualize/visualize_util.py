

import logging
import warnings

import numpy as np
from PIL import Image
import scipy.stats
from disent.util import to_numpy


log = logging.getLogger(__name__)


# ========================================================================= #
# operations on Images/Animations                                           #
# ========================================================================= #


def make_image_grid(images, pad=8, border=True, bg_color=0.5, num_cols=None):
    """
    Convert a list of images into a single image that is a grid of those images.
    :param images: list of input images (all the same size)
    :param pad: the number of pixels between images
    :param is_border: if there should be a border around the grid
    :param bg_color: the background color to use for padding (can be a float, int or RGB tuple)
    :param num_cols: the number of output columns in the grid. None for auto square, -1 for rows==1, or > 0 for that many cols.
    :return: single output image.
    """
    # get image sizes
    img_shape, ndim = np.array(images[0].shape), images[0].ndim
    assert ndim == 2 or ndim == 3, 'images have wrong number of channels'
    assert np.all(img_shape == img.shape for img in images), 'Images are not the same shape!'
    # get image size and channels
    img_size = img_shape[:2]
    if ndim == 3:
        assert (img_shape[2] == 1) or (img_shape[2] == 3), 'Invalid number of channels for an image.'
    # grid sizes
    num_rows, num_cols = _get_size(len(images), num_cols=num_cols)
    grid_size = (img_size + pad) * [num_rows, num_cols] + (pad if border else -pad)
    # image sizes including padding on one side
    deltas = img_size + pad
    offset = pad if border else 0
    # make image
    grid = np.full_like(images, fill_value=bg_color, shape=(*grid_size, *img_shape[2:]))
    # fill image
    for i, img in enumerate(images):
        y0, x0 = offset + deltas * [i // num_cols, i % num_cols]
        y1, x1 = img_size + [y0, x0]
        grid[y0:y1, x0:x1, ...] = img
    return grid


def make_animation_grid(list_of_animated_images, pad=8, border=True, bg_color=0.5, num_cols=None):
    full_size_images = []
    for single_images in zip(*list_of_animated_images):
        full_size_images.append(make_image_grid(single_images, pad=pad, border=border, bg_color=bg_color, num_cols=num_cols))
    return to_numpy(full_size_images)


# ========================================================================= #
# Calculations/Heuristics                                                   #
# ========================================================================= #


def _get_size(n, num_cols=None):
    """
    Determine the number of rows and columns, given the total number of elements n.
    - if num_cols is None:     rows x cols is as square as possible
    - if num_cols is a number: minimum number of rows needed is returned.
    - if num_cols <= 0:        only 1 row is returned
    :return: (num_rows, num_cols)
    """
    if num_cols is None:
        num_cols = int(np.ceil(n ** 0.5))
    elif num_cols <= 0:
        num_cols = n
    num_rows = (n + num_cols - 1) // num_cols
    return num_rows, num_cols


# ========================================================================= #
# Index Cycle Generators | FROM: disentanglement_lib                        #
# ========================================================================= #


def cycle_factor(starting_index, num_indices, num_frames):
    """
    Cycles through the state space in a single cycle.
    eg. starting_index=4, num_indices=5, num_frames=8 returns: [4, 3, 2, 1, 0, 1, 2, 3]
    eg. starting_index=2, num_indices=5, num_frames=8 returns: [2, 4, 4, 3, 2, 0, 0, 1]
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    Copyright 2018 The DisentanglementLib Authors. All rights reserved.
    Licensed under the Apache License, Version 2.0
    https://github.com/google-research/disentanglement_lib
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    """
    grid = np.linspace(starting_index, starting_index + 2 * num_indices, num=num_frames, endpoint=False)
    grid = np.array(np.ceil(grid), dtype=np.int64)
    grid -= np.maximum(0, 2 * grid - 2 * num_indices + 1)
    grid += np.maximum(0, -2 * grid - 1)
    return grid


def cycle_gaussian(starting_value, num_frames, loc=0., scale=1.):
    """
    Cycles through the quantiles of a Gaussian in a single cycle.
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    Copyright 2018 The DisentanglementLib Authors. All rights reserved.
    Licensed under the Apache License, Version 2.0
    https://github.com/google-research/disentanglement_lib
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    """
    starting_prob = scipy.stats.norm.cdf(starting_value, loc=loc, scale=scale)
    grid = np.linspace(starting_prob, starting_prob + 2., num=num_frames, endpoint=False)
    grid -= np.maximum(0, 2 * grid - 2)
    grid += np.maximum(0, -2 * grid)
    grid = np.minimum(grid, 0.999)
    grid = np.maximum(grid, 0.001)
    return np.array([scipy.stats.norm.ppf(i, loc=loc, scale=scale) for i in grid])


def cycle_interval(starting_value, num_frames, min_val, max_val):
    """
    Cycles through the state space in a single cycle.
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    Copyright 2018 The DisentanglementLib Authors. All rights reserved.
    Licensed under the Apache License, Version 2.0
    https://github.com/google-research/disentanglement_lib
    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    """
    starting_in_01 = (starting_value - min_val) / (max_val - min_val)
    starting_in_01 = np.nan_to_num(starting_in_01)  # handle division by zero, prints warning
    grid = np.linspace(starting_in_01, starting_in_01 + 2., num=num_frames, endpoint=False)
    grid -= np.maximum(0, 2 * grid - 2)
    grid += np.maximum(0, -2 * grid)
    return grid * (max_val - min_val) + min_val


# ========================================================================= #
# Conversion/Util                                                           #
# ========================================================================= #


def reconstructions_to_images(recon, mode='float', moveaxis=True):
    """
    Convert a batch of reconstructions to images.
    A batch in this case consists of an arbitrary number of dimensions of an array,
    with the last 3 dimensions making up the actual image. For example: (..., channels, size, size)

    NOTE: This function might not be efficient for large amounts of
          data due to assertions and initial recursive conversions to a numpy array.

    NOTE: kornia has a similar function!
    """
    img = to_numpy(recon)
    # checks
    assert img.ndim >= 3
    assert img.dtype in (np.float32, np.float64)
    if np.min(img) < 0 or np.max(img) > 1:
        warnings.warn('images are being clipped between 0 and 1')
    img = np.clip(img, 0, 1)
    # move channels axis
    if moveaxis:
        # TODO: automatically detect
        img = np.moveaxis(img, -3, -1)
    # convert
    if mode == 'float':
        return img
    elif mode == 'int':
        return np.uint8(img * 255)
    elif mode == 'pil':
        img = np.uint8(img * 255)
        # all the cases (even ndim == 3)... bravo numpy, bravo!
        images = [Image.fromarray(img[idx]) for idx in np.ndindex(img.shape[:-3])]
        images = np.array(images, dtype=object).reshape(img.shape[:-3])
        return images
    else:
        raise KeyError(f'Invalid mode: {repr(mode)} not in { {"float", "int", "pil"} }')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
