import numpy as np
import matplotlib.pyplot as plt
import torch

from disent.systems.vae import VaeSystem
from disent.util import to_numpy
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
        ax.imshow(img)
    return fig, axs


# ========================================================================= #
# notebook                                                                  #
# ========================================================================= #

def notebook_display_animation(frames, fps=10):
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
    display.display(image)

# ========================================================================= #
# notebook - dataset visualisation                                          #
# ========================================================================= #

def plt_sample_dataset_still_images(data='3dshapes', num_samples=16, mode='spread', figsize_ratio=0.75):
    images = sample_dataset_still_images(data=data, num_samples=num_samples, mode=mode)
    plt_images_grid(images, figsize_ratio=figsize_ratio)

def plt_sample_dataset_animation(data='3dshapes', num_frames=9, figsize_ratio=0.75):
    frames = sample_dataset_animations(data, num_animations=1, num_frames=num_frames)[0]
    plt_images_grid(frames, figsize_ratio=figsize_ratio)

def notebook_display_sample_dataset_animation(data='3dshapes', num_frames=30, fps=10):
    frames = sample_dataset_animations(data, num_animations=1, num_frames=num_frames)[0]
    frames = gridify_animation(frames)
    notebook_display_animation(frames, fps=fps)


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

def plt_traverse_latent_space(system, num_samples=1, dimensions=None, values=11, figsize_ratio=0.75):
    obs = torch.stack(system.dataset_train.sample_observations(num_samples)).to(system.device)
    z_mean, z_logvar = to_numpy(system.model.encode_gaussian(obs))
    _plt_traverse_latent_space(system.model.decode, z_mean, dimensions=dimensions, values=values, figsize_ratio=figsize_ratio)


def _notebook_display_traverse_latent_space(decoder_fn, z_mean, dimensions=None, values=None, fps=10):
    traversals = latent_traversals(decoder_fn, z_mean, dimensions=dimensions, values=values)
    traversals = reconstructions_to_images(traversals, 'int', moveaxis=False)
    for frames in traversals:
        frames = gridify_animation(frames)
        notebook_display_animation(frames, fps=fps)

def notebook_display_traverse_latent_space(system, num_samples=1, dimensions=None, values=21, fps=10):
    obs = torch.stack(system.dataset_train.sample_observations(num_samples)).to(system.device)
    z_mean, z_logvar = to_numpy(system.model.encode_gaussian(obs))
    _notebook_display_traverse_latent_space(system.model.decode, z_mean, dimensions=dimensions, values=values, fps=fps)


def _plt_sample_observations_and_reconstruct(gaussian_encoder_fn, decoder_fn, dataset, num_samples=16, figsize_ratio=0.75):
    obs, x_recon = sample_observations_and_reconstruct(gaussian_encoder_fn, decoder_fn, dataset, num_samples=num_samples)
    plt_images_minimal_square(obs, figsize_ratio=figsize_ratio)
    plt_images_minimal_square(x_recon, figsize_ratio=figsize_ratio)

def plt_sample_observations_and_reconstruct(system, num_samples=16, figsize_ratio=0.75):
    _plt_sample_observations_and_reconstruct(system.model.encode_gaussian, system.model.decode, system.dataset_train, num_samples=num_samples, figsize_ratio=figsize_ratio)

def _notebook_display_latent_cycle(decoder_fn, z_means, z_logvars, mode='fixed_interval_cycle', num_animations=1, num_frames=20, fps=10):
    animations = latent_cycle(decoder_fn, z_means, z_logvars, mode=mode, num_animations=num_animations, num_frames=num_frames)
    animations = reconstructions_to_images(animations, mode='int', moveaxis=False)  # axis already moved above
    for i, images_frames in enumerate(animations):
        frames = gridify_animation(images_frames)
        notebook_display_animation(frames, fps=fps)

def notebook_display_latent_cycle(system, mode='fixed_interval_cycle', num_animations=1, num_test_samples=64, num_frames=21, fps=8, obs=None):
    if obs is None:
        obs = torch.stack(system.dataset_train.sample_observations(num_test_samples)).to(system.device)
    else:
        obs = torch.as_tensor(obs).to(system.device)
    z_mean, z_logvar = to_numpy(system.model.encode_gaussian(obs))
    _notebook_display_latent_cycle(system.model.decode, z_mean, z_logvar, mode=mode, num_animations=num_animations, num_frames=num_frames, fps=fps)

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
