from disent.dataset.ground_truth.base import DownloadableGroundTruthData

# ========================================================================= #
# mpi3d                                                                   #
# ========================================================================= #

class Mpi3dData(DownloadableGroundTruthData):
    """
    MPI3D Dataset
    - https://github.com/rr-learning/disentanglement_dataset

    Files:
        - toy:       https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz
        - realistic: https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_realistic.npz
        - real:      https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz

    reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/mpi3d.py
    """

    factor_names = ("object_color", "object_shape", "object_size", "camera_height", "background_color", "first_dof", "second_dof")
    factor_sizes = (4, 4, 2, 3, 3, 40, 40)
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
