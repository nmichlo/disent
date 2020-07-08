import numpy as np
import matplotlib.pyplot as plt
import torch

from disent.systems.vae import VaeSystem
from disent.util import TempNumpySeed, to_numpy
from disent.visualize.visualize_dataset import sample_dataset_animations, sample_dataset_still_images
from disent.visualize.visualize_model import (LATENT_CYCLE_MODES, latent_random_samples, latent_traversals,
                                              sample_observations_and_reconstruct, latent_cycle)
from disent.visualize.visualize_util import gridify_animation, reconstructions_to_images


# ========================================================================= #
# matplotlib                                                                #
# ========================================================================= #


def plt_subplots_grid(h, w, figsize_ratio=1., space=0):
    fig, axs = plt.subplots(h, w, figsize=(w*figsize_ratio, h*figsize_ratio))
    axs = np.array(axs)
    # remove spacing
    fig.subplots_adjust(wspace=space*figsize_ratio, hspace=space*figsize_ratio)
    # remove axis
    for ax in axs.flatten():
        ax.axis('off')
    return fig, axs.reshape(h, w)

def plt_images_grid(image_grid, figsize_ratio=1., space=0, swap_xy=False):
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

def plt_images_minimal_square(image_list, figsize_ratio=1., space=0):
    # minimal square
    size = int(np.ceil(len(image_list) ** 0.5))
    # DISPLAY IMAGES
    fig, axs = plt_subplots_grid(size, size, figsize_ratio, space)
    for ax, img in zip(np.array(axs).flatten(), image_list):
        # TODO: this should not be here, image list should be PIL images first
        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = img[:, :, 0]
        ax.imshow(img)
    return fig, axs


# ========================================================================= #
# notebook                                                                  #
# ========================================================================= #

def notebook_display_animation(frames, fps=10, display_id=None):
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
    data = mimwrite('<bytes>', to_numpy(frames), format='gif', fps=fps)
    # diplay gif image in notebook | https://github.com/ipython/ipython/issues/10045#issuecomment-522697219
    image = display.Image(data=data, format='gif')
    if display_id:
        display.update_display(image, display_id=display_id)
        return display_id
    else:
        return display.display(image, display_id=True).display_id

def notebook_display_image(img, format='png', display_id=None):
    """
    Display a raw image within an IPython notebook
    - Internally converts the image to bytes
      representing a png image and then displays it.
    """
    assert format in {'png', 'jpg'}, 'Invalid format, must be one of: {"png", "jpg"}'
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "all"
    from IPython import display
    from imageio import imwrite
    # convert to png and get bytes | https://imageio.readthedocs.io/en/stable/userapi.html
    data = imwrite('<bytes>', to_numpy(img), format=format)
    # diplay png image in notebook | https://github.com/ipython/ipython/issues/10045#issuecomment-522697219
    image = display.Image(data=data, format=format)
    if display_id:
        display.update_display(image, display_id=display_id)
        return display_id
    else:
        return display.display(image, display_id=True).display_id

# ========================================================================= #
# notebook - dataset visualisation                                          #
# ========================================================================= #

def plt_sample_dataset_still_images(data='3dshapes', num_samples=16, mode='spread', figsize_ratio=0.75):
    images = sample_dataset_still_images(data=data, num_samples=num_samples, mode=mode)
    plt_images_grid(images, figsize_ratio=figsize_ratio)

def plt_sample_dataset_animation(data='3dshapes', num_frames=9, figsize_ratio=0.75, seed=None):
    frames = sample_dataset_animations(data, num_animations=1, num_frames=num_frames, seed=seed)[0]
    plt_images_grid(frames, figsize_ratio=figsize_ratio)

def notebook_display_sample_dataset_animation(data='3dshapes', num_frames=30, fps=10, seed=None, display_id=None):
    frames = sample_dataset_animations(data, num_animations=1, num_frames=num_frames, seed=seed)[0]
    frames = gridify_animation(frames)
    return notebook_display_animation(frames, fps=fps, display_id=display_id)


# ========================================================================= #
# notebook - model/system visualisation                                     #
# ========================================================================= #


def _plt_latent_random_samples(decoder_fn, z_size, num_samples=16, figsize_ratio=0.75):
    images = latent_random_samples(decoder_fn, z_size, num_samples)
    plt_images_minimal_square(images, figsize_ratio=figsize_ratio)

def plt_latent_random_samples(system: 'VaeSystem', num_samples=16, figsize_ratio=0.75):
    _plt_latent_random_samples(system.model.decode, system.model.z_size, num_samples=num_samples, figsize_ratio=figsize_ratio)


# TODO, these are useless, convert to just using the VaeSystem
def _plt_traverse_latent_space(decoder_fn, z_mean, dimensions=None, values=None, figsize_ratio=0.75):
    traversals = latent_traversals(decoder_fn, z_mean, dimensions=dimensions, values=values)
    for images_grid in traversals:
        plt_images_grid(images_grid, figsize_ratio=figsize_ratio)

def plt_traverse_latent_space(system, num_samples=1, dimensions=None, values=11, figsize_ratio=0.75, seed=None):
    # TODO: this is not general
    with TempNumpySeed(seed):
        obs = system.dataset.sample_observations(num_samples).cuda()
    z_mean, z_logvar = to_numpy(system.model.encode_gaussian(obs))
    _plt_traverse_latent_space(system.model.decode, z_mean, dimensions=dimensions, values=values, figsize_ratio=figsize_ratio)


def _notebook_display_traverse_latent_space(decoder_fn, z_mean, dimensions=None, values=None, fps=10, display_ids=None):
    traversals = latent_traversals(decoder_fn, z_mean, dimensions=dimensions, values=values)
    traversals = reconstructions_to_images(traversals, 'int', moveaxis=False)
    new_display_ids = []
    for i, frames in enumerate(traversals):
        frames = gridify_animation(frames)
        d_id = notebook_display_animation(frames, fps=fps, display_id=display_ids[i] if display_ids else None)
        new_display_ids.append(d_id)
    return new_display_ids if new_display_ids else None

def notebook_display_traverse_latent_space(system, num_samples=1, dimensions=None, values=21, fps=10, display_ids=None, seed=None):
    # TODO: this is not general
    with TempNumpySeed(seed):
        obs = system.dataset.sample_observations(num_samples).cuda()
    z_mean, z_logvar = to_numpy(system.model.encode_gaussian(obs))
    return _notebook_display_traverse_latent_space(system.model.decode, z_mean, dimensions=dimensions, values=values, fps=fps, display_ids=display_ids)


def _plt_sample_observations_and_reconstruct(gaussian_encoder_fn, decoder_fn, dataset, num_samples=16, figsize_ratio=0.75, seed=None):
    obs, x_recon = sample_observations_and_reconstruct(gaussian_encoder_fn, decoder_fn, dataset, num_samples=num_samples, seed=seed)
    plt_images_minimal_square(obs, figsize_ratio=figsize_ratio)
    plt_images_minimal_square(x_recon, figsize_ratio=figsize_ratio)

def plt_sample_observations_and_reconstruct(system, num_samples=16, figsize_ratio=0.75, seed=None):
    _plt_sample_observations_and_reconstruct(system.model.encode_gaussian, system.model.decode, system.dataset, num_samples=num_samples, figsize_ratio=figsize_ratio, seed=seed)

def _notebook_display_latent_cycle(decoder_fn, z_means, z_logvars, mode='fixed_interval_cycle', num_animations=1, num_frames=20, fps=10):
    animations = latent_cycle(decoder_fn, z_means, z_logvars, mode=mode, num_animations=num_animations, num_frames=num_frames)
    animations = reconstructions_to_images(animations, mode='int', moveaxis=False)  # axis already moved above
    for i, images_frames in enumerate(animations):
        frames = gridify_animation(images_frames)
        notebook_display_animation(frames, fps=fps)

def notebook_display_latent_cycle(system, mode='fixed_interval_cycle', num_animations=1, num_test_samples=64, num_frames=21, fps=8, obs=None, seed=None):
    if obs is None:
        # TODO: this is not general
        with TempNumpySeed(seed):
            obs = system.dataset.sample_observations(num_test_samples).cuda()
    else:
        # TODO: this is not general
        obs = torch.as_tensor(obs).cuda()
    z_mean, z_logvar = to_numpy(system.model.encode_gaussian(obs))
    _notebook_display_latent_cycle(system.model.decode, z_mean, z_logvar, mode=mode, num_animations=num_animations, num_frames=num_frames, fps=fps)

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
