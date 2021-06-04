#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~


# ========================================================================= #
# Augment                                                                   #
# ========================================================================= #


class DisentDatasetTransform(object):
    """
    Applies transforms to batches generated from dataloaders of
    datasets from: disent.dataset.groundtruth
    """

    def __init__(self, transform=None, transform_targ=None):
        self.transform = transform
        self.transform_targ = transform_targ

    def __call__(self, batch):
        # transform inputs
        if self.transform is not None:
            if 'x' not in batch:
                batch['x'] = batch['x_targ']
            batch['x'] = _apply_transform_to_batch_dict(batch['x'], self.transform)
        # transform targets
        if self.transform_targ is not None:
            batch['x_targ'] = _apply_transform_to_batch_dict(batch['x_targ'], self.transform_targ)
        # done!
        return batch

    def __repr__(self):
        return f'{self.__class__.__name__}(transform={repr(self.transform)}, transform_targ={repr(self.transform_targ)})'


def _apply_transform_to_batch_dict(batch, transform):
    if isinstance(batch, tuple):
        return tuple(transform(obs) for obs in batch)
    if isinstance(batch, list):
        return list(transform(obs) for obs in batch)
    else:
        return transform(batch)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
