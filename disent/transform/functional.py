import torch
import torchvision.transforms.functional as F_tv


def noop(obs):
    """
    Transform that does absolutely nothing!
    """
    return obs


def check_tensor(obs, low=0., high=1., dtype=torch.float32):
    """
    Check that the input is a tensor, its datatype matches, and
    that it is in the required range.
    """
    # check is a tensor
    assert torch.is_tensor(obs), 'observation is not a tensor'
    # check type
    if dtype is not None:
        assert obs.dtype == dtype, f'tensor type {obs.dtype} is not required type {dtype}'
    # check range | TODO: are assertion strings inefficient?
    assert low <= obs.min(), f'minimum value of tensor {obs.min()} is less than allowed minimum value: {low}'
    assert obs.max() <= high, f'maximum value of tensor {obs.max()} is greater than allowed maximum value: {high}'
    # DONE!
    return obs


def to_standardised_tensor(obs, size=None, check=True):
    """
    Basic transform that should be applied to
    any dataset before augmentation.

    1. resize if size is specified
    2. convert to tensor in range [0, 1]
    """
    # resize image
    if size is not None:
        obs = F_tv.to_pil_image(obs)
        obs = F_tv.resize(obs, size=size)
    # transform to tensor
    obs = F_tv.to_tensor(obs)
    # check that tensor is valid
    if check:
        obs = check_tensor(obs, low=0, high=1, dtype=torch.float32)
    return obs
