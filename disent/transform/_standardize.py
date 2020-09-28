import torch
import torchvision


class CheckTensor(object):

    def __init__(self, low=0, high=1, dtype=torch.float32):
        self._low = low
        self._high = high
        self._dtype = dtype

    def __call__(self, obs):
        # check is a tensor
        assert torch.is_tensor(obs), 'observation is not a tensor'
        # check type
        assert obs.dtype == self._dtype, f'tensor type {obs.dtype} is not required type {self._dtype}'
        # check range
        assert self._low <= obs.min(), f'minimum value of tensor {obs.min()} is less than allowed minimum value: {self._low}'
        assert obs.max() <= self._high, f'maximum value of tensor {obs.max()} is greater than allowed maximum value: {self._high}'
        # DONE!
        return obs


class ToStandardisedTensor(object):
    """
    Basic transform that should be applied to
    any dataset before augmentation.

    1. resize if size is specified
    2. convert to tensor in range [0, 1]
    """

    def __init__(self, size=None):
        transforms = []
        # resize image
        if size is not None:
            transforms.append(torchvision.transforms.ToPILImage())
            transforms.append(torchvision.transforms.Resize(size=size))
        # transform to tensor
        transforms.append(torchvision.transforms.ToTensor())
        # final transform
        self._transform = torchvision.transforms.Compose(transforms)

    def __call__(self, obs) -> torch.Tensor:
        return self._transform(obs)


class NormalizeTensor(object):
    """
    Basic transform that should be applied after augmentation before
    being passed to a model as the input.

    1. check that tensor is in range [0, 1]
    2. normalise tensor in range [-1, 1]
    """

    def __init__(self):
        self._transform = torchvision.transforms.Compose([
            CheckTensor(low=0, high=1, dtype=torch.float32),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    def __call__(self, obs) -> torch.Tensor:
        return self._transform(obs)


