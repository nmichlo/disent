from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from disent.dataset import make_ground_truth_data, make_ground_truth_dataset
from disent.dataset.ground_truth.base import GroundTruthData, GroundTruthDataset
from disent.util import to_numpy


def get_data(data: Union[str, GroundTruthData]) -> GroundTruthData:
    if isinstance(data, str):
        data = make_ground_truth_data(data, try_in_memory=False)
    return data

def get_dataset(dataset: Union[str, GroundTruthDataset]):
    if isinstance(dataset, str):
        dataset = make_ground_truth_dataset(dataset, try_in_memory=False)
    return dataset

# ========================================================================= #
# visualise_util                                                            #
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
    assert 0 <= np.min(img) <= 1
    assert 0 <= np.max(img) <= 1
    # move channels axis
    if moveaxis:
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
# notebook                                                                  #
# ========================================================================= #


def notebook_display_animation(frames):
    """
    Display an animation within an IPython notebook.
    - Internally converts the list of frames into bytes
      representing a gif images and then displays them.
    """
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "all"
    from IPython import display
    from imageio import mimwrite
    # convert to gif and get bytes | https://imageio.readthedocs.io/en/stable/userapi.html
    data = mimwrite('<bytes>', to_numpy(frames), format='gif')
    # diplay gif image in notebook | https://github.com/ipython/ipython/issues/10045#issuecomment-522697219
    display.Image(data=data, format='gif')


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
