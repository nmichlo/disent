import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from disent.util import to_numpy


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

def plt_make_subplots_grid(images, figsize=(8, 8), space=0):
    with torch.no_grad():
        # minimal square
        size = int(np.ceil(len(images) ** 0.5))
        # DISPLAY IMAGES
        fig, axs = plt.subplots(size, size, figsize=figsize)
        fig.subplots_adjust(wspace=space, hspace=space)
        axs = np.array(axs).reshape(-1)
        for ax, img in zip(axs, images):
            ax.imshow(img)
            ax.axis('off')
    return fig, axs

def make_image_grid(images, pad=0):
    # variables
    grid_width = int(np.ceil(len(images) ** 0.5))
    img_shape = np.array(images[0].shape)
    img_size, img_channels = img_shape[:2], img_shape[2]
    dy, dx = img_size + pad
    grid_size = (img_size + pad) * grid_width - pad
    # make image
    grid = np.zeros_like(images, shape=(*grid_size, img_channels))
    for i, img in enumerate(images):
        iy, ix = i // grid_size, i % grid_size
        grid[dy*iy:dy*(iy+1), dx*ix:dx*(ix+1), :] = img
    # return made image
    return grid


def save_frames_as_animation(frames, out_file, fps=30):
    import imageio
    with imageio.get_writer(out_file, fps=fps, mode='I') as writer:
        for frame in frames:
            writer.append_data(frame)



# ========================================================================= #
# END                                                                       #
# ========================================================================= #
