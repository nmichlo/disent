

class GroundTruthDatasetAugment(object):
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
            batch = _apply_transform_to_batch_dict(batch, 'x_targ', self.transform)
        # done!
        return batch


def _apply_transform_to_batch_dict(batch, key, transform):
    observations = batch[key]
    if isinstance(observations, (list, tuple)):
        observations = [transform(obs) for obs in observations]
    else:
        observations = transform(observations)
    batch[key] = observations
    return batch
