from disent.dataset.ground_truth.base import DownloadableGroundTruthDataset


# ========================================================================= #
# dataset_norb                                                              #
# ========================================================================= #


class SmallNorbDataset(DownloadableGroundTruthDataset):
    """
    Small NORB Dataset
    - https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/

    Files:
        - direct hdf5: https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5
        - direct npz: https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz

    # reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/norb.py
    """

    # The instance in each category is randomly sampled when generating the images.
    factor_names = ('category', 'elevation', 'azimuth', 'lighting_condition')
    factor_sizes = (5, 9, 18, 6)
    observation_shape = (64, 64, 1)

    dataset_url = None

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def _get_observation(self, indices):
        raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
