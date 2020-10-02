

class GroundTruthDatasetBatchAugment(object):
    """
    Applies transforms to batches generated from dataloaders of
    datasets from: disent.dataset.groundtruth
    """

    def __init__(self, transform=None, transform_targ=None):
        self.transform = transform
        self.transform_targ = transform_targ

    def __call__(self, batch):
        # transform inputs
        if self.transform:
            batch = _apply_transform_to_batch_dict(batch, 'x', self.transform)
        # transform targets
        if self.transform_targ:
            batch = _apply_transform_to_batch_dict(batch, 'x_targ', self.transform_targ)
        # done!
        return batch

    def __repr__(self):
        return f'{self.__class__.__name__}(transform={repr(self.transform)}, transform_targ={repr(self.transform_targ)})'


def _apply_transform_to_batch_dict(batch, key, transform):
    observations = batch[key]
    if isinstance(observations, tuple):
        observations = tuple([transform(obs) for obs in observations])
    if isinstance(observations, list):
        observations = [transform(obs) for obs in observations]
    else:
        observations = transform(observations)
    batch[key] = observations
    return batch
