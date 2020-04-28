from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from disent.dataset import make_ground_truth_data
from disent.dataset.ground_truth.base import GroundTruthData
from disent.util import to_numpy


def get_data(data: Union[str, GroundTruthData]) -> GroundTruthData:
    if isinstance(data, str):
        data = make_ground_truth_data(data, try_in_memory=False)
    return data


# ========================================================================= #
# visualise_util                                                            #
# ========================================================================= #


def reconstruction_to_image(recon):
    """
    convert a single reconstruction to an image
    """
    # recover images
    img = np.moveaxis(to_numpy(recon), 0, -1)
    img = np.uint8(img * 255)
    # convert images
    return Image.fromarray(img)

def reconstructions_to_images(x_recon):
    """
    convert a batch reconstruction to images
    """
    # recover images
    imgs = np.moveaxis(to_numpy(x_recon), 1, -1)
    imgs = np.uint8(imgs * 255)
    # convert images
    return [Image.fromarray(img) for img in imgs]


# ========================================================================= #
# matplotlib                                                                #
# ========================================================================= #


def plt_subplots_grid(h, w, figsize_ratio=1, space=0):
    fig, axs = plt.subplots(h, w, figsize=(w*figsize_ratio, h*figsize_ratio))
    axs = np.array(axs)
    # remove spacing
    fig.subplots_adjust(wspace=space*figsize_ratio, hspace=space*figsize_ratio)
    # remove axis
    for ax in axs.flatten():
        ax.axis('off')
    return fig, axs.reshape(h, w)

def plt_images_grid(image_grid, figsize_ratio=1, space=0, swap_xy=False):
    # swap x and y axis
    if swap_xy:
        image_grid = np.moveaxis(image_grid, 0, 1)
    # get size of grid
    h, w = np.array(image_grid.shape[:2])
    # make subplots
    fig, axs = plt_subplots_grid(h, w, figsize_ratio, space)
    # show images in subplots
    for ax_row, img_row in zip(axs, image_grid):
        for ax, img in zip(ax_row, img_row):
            ax.imshow(img)
    return fig, axs

def plt_images_minimal_square(image_list, figsize_ratio=1, space=0):
    # minimal square
    size = int(np.ceil(len(image_list) ** 0.5))
    # DISPLAY IMAGES
    fig, axs = plt_subplots_grid(size, size, figsize_ratio, space)
    for ax, img in zip(np.array(axs).flatten(), image_list):
        ax.imshow(img)
    return fig, axs


# ========================================================================= #
# numpy                                                                     #
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
