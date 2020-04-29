from typing import Union

import numpy as np
from PIL import Image

from disent.dataset import make_ground_truth_data, make_ground_truth_dataset
from disent.dataset.ground_truth.base import GroundTruthData, GroundTruthDataset
from disent.util import to_numpy


# ========================================================================= #
# dataset util TODO: move elsewhere
# ========================================================================= #


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
