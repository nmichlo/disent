from disent.dataset.ground_truth.base import GroundTruthDataset

# ========================================================================= #
# mpi3d                                                                   #
# ========================================================================= #

class Mpi3dDataset(GroundTruthDataset):
    """
    MPI3D Dataset
    - https://github.com/rr-learning/disentanglement_dataset

    Files:
        - toy:       https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz
        - realistic: https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_realistic.npz
        - real:      https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz

        https://storage.googleapis.com/3d-shapes/3dshapes.h5
    """

    factor_names = ()
    factor_sizes = ()
    observation_shape = ()

    def __getitem__(self, indices):
        pass
# ========================================================================= #
# END                                                                       #
# ========================================================================= #
