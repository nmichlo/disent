# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ========================================================================
#
# ADAPTED FROM Google's disentanglement_lib:
# https://github.com/google-research/disentanglement_lib
#
# Modified for pytorch and disent by Nathan Michlo

"""
Utility functions for the visualization code.
"""

import math
import numpy as np
from PIL import Image
import scipy.stats
import imageio
from disent.util import make_logger, to_numpy

log = make_logger()

# ========================================================================= #
# Saving Images/Animations | FROM: disentanglement_lib                      #
# ========================================================================= #


def save_image(image, image_path):
    """Saves an image in the [0,1]-valued Numpy array to image_path.

    Args:
      image: Numpy array of shape (height, width, {1,3}) with values in [0, 1].
      image_path: String with path to output image.
    """
    assert 0 <= np.max(image) <= 1
    assert 0 <= np.min(image) <= 1

    # Copy the single channel if we are provided a grayscale image.
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    image = np.ascontiguousarray(image)
    image *= 255
    image = image.astype("uint8")

    with open(image_path, "wb") as path:
        img = Image.fromarray(image, mode="RGB")
        img.save(path)


def minimal_square_save_images(images, image_path):
    """Saves images in list of [0,1]-valued np.arrays on a grid.

    Args:
      images: List of Numpy arrays of shape (height, width, {1,3}) with values in
        [0, 1].
      image_path: String with path to output image.

    *NB* PREVIOUSLY in disentanglement_lib known as grid_save_images
    """
    assert 0 <= np.max(images) <= 1
    assert 0 <= np.min(images) <= 1
    # minimum square size
    side_length = int(math.floor(math.sqrt(len(images))))
    # joi
    image_rows = [
        np.concatenate(images[side_length*i:side_length*(i+1)], axis=0)
        for i in range(side_length)
    ]
    tiled_image = np.concatenate(image_rows, axis=1)
    save_image(tiled_image, image_path)


def grid_save_images(image_grid, image_path):
    assert 0 <= np.max(image_grid) <= 1
    assert 0 <= np.min(image_grid) <= 1
    image_rows = [np.concatenate(row, axis=0) for row in image_grid]
    image = np.concatenate(image_rows, axis=1)
    save_image(image, image_path)


def save_gridified_animation(list_of_animated_images, image_path, fps):
    assert list_of_animated_images.dtype == np.uint8, f'{list_of_animated_images.dtype}'
    full_size_images = gridify_animation(list_of_animated_images)
    imageio.mimwrite(image_path, full_size_images, fps=fps)


# ========================================================================= #
# operations on Images/Animations | FROM: disentanglement_lib               #
# ========================================================================= #


def padded_grid(images, num_rows=None, padding_px=8, value=None):
    """Creates a grid with padding in between images."""
    num_images = len(images)
    if num_rows is None:
        num_rows = best_num_rows(num_images)

    # Computes how many empty images we need to add.
    num_cols = int(np.ceil(float(num_images) / num_rows))
    num_missing = num_rows * num_cols - num_images

    # Add the empty images at the end.
    all_images = images + [np.ones_like(images[0])] * num_missing

    # Create the final grid.
    rows = [padded_stack(all_images[i * num_cols:(i + 1) * num_cols], padding_px, 1, value=value) for i in range(num_rows)]
    return padded_stack(rows, padding_px, axis=0, value=value)


def padded_stack(images, padding_px=8, axis=0, value=None):
    """Stacks images along axis with padding in between images."""
    padding_arr = padding_array(images[0], padding_px, axis, value=value)
    new_images = [images[0]]
    for image in images[1:]:
        new_images.append(padding_arr)
        new_images.append(image)
    return np.concatenate(new_images, axis=axis)


def padding_array(image, padding_px, axis, value=None):
    """Creates padding image of proper shape to pad image along the axis."""
    shape = list(image.shape)
    shape[axis] = padding_px
    if value is None:
        return np.ones(shape, dtype=image.dtype)
    else:
        try:
            len(value)
        except:
            value = [value] * shape[-1]
        assert len(value) == shape[-1]
        shape[-1] = 1
        return np.tile(value, shape)

def pad_around(image, padding_px=8, axis=None, value=None):
    """Adds a padding around each image."""
    # If axis is None, pad both the first and the second axis.
    if axis is None:
        image = pad_around(image, padding_px, axis=0, value=value)
        axis = 1
    padding_arr = padding_array(image, padding_px, axis, value=value)
    return np.concatenate([padding_arr, image, padding_arr], axis=axis)


def gridify_animation(list_of_animated_images, padding_px=8, value=None):
    full_size_images = []
    for single_images in zip(*list_of_animated_images):
        full_size_images.append(pad_around(padded_grid(list(single_images), padding_px=padding_px, value=value), padding_px=padding_px, value=value))
    return to_numpy(full_size_images)


# def add_below(image, padding_px=8, value=None):
#     """Adds a footer below."""
#     if len(image.shape) == 2:
#         image = np.expand_dims(image, -1)
#     if image.shape[2] == 1:
#         image = np.repeat(image, 3, 2)
#     if image.shape[2] != 3:
#         raise ValueError("Could not convert image to have three channels.")
#
#     with tf.gfile.Open(resources.get_file("disentanglement_lib.png"), "rb") as f:
#         footer = np.array(Image.open(f).convert("RGB")) * 1.0 / 255.
#
#     missing_px = image.shape[1] - footer.shape[1]
#     if missing_px < 0:
#         return image
#     if missing_px > 0:
#         padding_arr = padding_array(footer, missing_px, axis=1, value=value)
#         footer = np.concatenate([padding_arr, footer], axis=1)
#
#     return padded_stack([image, footer], padding_px, axis=0, value=value)


# ========================================================================= #
# Calculations/Heuristics | FROM: disentanglement_lib                       #
# ========================================================================= #


def best_num_rows(num_elements, max_ratio=4):
    """Automatically selects a smart number of rows."""
    best_remainder = num_elements
    best_i = None
    i = int(np.sqrt(num_elements))
    while True:
        if num_elements > max_ratio * i * i:
            return best_i
        remainder = (i - num_elements % i) % i
        if remainder == 0:
            return i
        if remainder < best_remainder:
            best_remainder = remainder
            best_i = i
        i -= 1


# ========================================================================= #
# Index Cycle Generators | FROM: disentanglement_lib                        #
# ========================================================================= #


def cycle_factor(starting_index, num_indices, num_frames):
    """Cycles through the state space in a single cycle.
    eg. starting_index=4, num_indices=5, num_frames=8 returns: [4, 3, 2, 1, 0, 1, 2, 3]
    eg. starting_index=2, num_indices=5, num_frames=8 returns: [2, 4, 4, 3, 2, 0, 0, 1]
    """
    grid = np.linspace(starting_index, starting_index + 2 * num_indices, num=num_frames, endpoint=False)
    grid = np.array(np.ceil(grid), dtype=np.int64)
    grid -= np.maximum(0, 2 * grid - 2 * num_indices + 1)
    grid += np.maximum(0, -2 * grid - 1)
    return grid


def cycle_gaussian(starting_value, num_frames, loc=0., scale=1.):
    """Cycles through the quantiles of a Gaussian in a single cycle."""
    starting_prob = scipy.stats.norm.cdf(starting_value, loc=loc, scale=scale)
    grid = np.linspace(starting_prob, starting_prob + 2., num=num_frames, endpoint=False)
    grid -= np.maximum(0, 2 * grid - 2)
    grid += np.maximum(0, -2 * grid)
    grid = np.minimum(grid, 0.999)
    grid = np.maximum(grid, 0.001)
    return np.array([scipy.stats.norm.ppf(i, loc=loc, scale=scale) for i in grid])


def cycle_interval(starting_value, num_frames, min_val, max_val):
    """Cycles through the state space in a single cycle."""
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
    """
    img = to_numpy(recon)
    # checks
    assert img.ndim >= 3
    assert img.dtype in (np.float32, np.float64)
    if np.min(img) < 0 or np.max(img) > 1:
        log.warning('image has been clipped')
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
        # WOW! I did not expect that to work for
        # all the cases (ndim == 3)... bravo numpy, bravo!
        images = [Image.fromarray(img[idx]) for idx in np.ndindex(img.shape[:-3])]
        images = np.array(images, dtype=object).reshape(img.shape[:-3])
        return images
    else:
        raise KeyError(f'Invalid mode: {repr(mode)} not in { {"float", "int", "pil"} }')


# ========================================================================= #
# numpy operations                                                          #
# TODO: replace disentanglement_lib methods                                 #
# ========================================================================= #


# def make_image_grid(images, pad=0):
#     # variables
#     grid_width = int(np.ceil(len(images) ** 0.5))
#     img_shape = np.array(images[0].shape)
#     img_size, img_channels = img_shape[:2], img_shape[2]
#     dy, dx = img_size + pad
#     grid_size = (img_size + pad) * grid_width - pad
#     # make image
#     grid = np.zeros_like(images, shape=(*grid_size, img_channels))
#     for i, img in enumerate(images):
#         iy, ix = i // grid_size, i % grid_size
#         grid[dy*iy:dy*(iy+1), dx*ix:dx*(ix+1), :] = img
#     # return made image
#     return grid
#
# def save_frames_as_animation(frames, out_file, fps=30):
#     import imageio
#     with imageio.get_writer(out_file, fps=fps, mode='I') as writer:
#         for frame in frames:
#             writer.append_data(frame)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
