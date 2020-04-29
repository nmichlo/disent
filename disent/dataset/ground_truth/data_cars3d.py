from disent.dataset.ground_truth.base import DownloadableGroundTruthData


# ========================================================================= #
# dataset_cars3d                                                            #
# ========================================================================= #


class Cars3dData(DownloadableGroundTruthData):
    """
    Cars3D Dataset
    - Deep Visual Analogy-Making (https://papers.nips.cc/paper/5845-deep-visual-analogy-making)
      http://www.scottreed.info

    Files:
        - http://www.scottreed.info/files/nips2015-analogy-data.tar.gz

    # reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/cars3d.py
    """

    factor_names = ('elevation', 'azimuth', 'object_type')
    factor_sizes = (4, 24, 183)
    observation_shape = (64, 64, 3)

    dataset_url = None

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def __getitem__(self, indices):
        raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
