from disent.data.groundtruth.base import Hdf5PreprocessedGroundTruthData


# ========================================================================= #
# dataset_dsprites                                                          #
# ========================================================================= #


class DSpritesData(Hdf5PreprocessedGroundTruthData):
    """
    DSprites Dataset
    - beta-VAE: Learning Basic Visual Concepts with a Constrained Variational BaseFramework
      (https://github.com/deepmind/dsprites-dataset)

    Files:
        - direct npz: https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
                      approx 2.5 GB loaded into memory
        - direct hdf5: https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5
                       default chunk size is (23040, 2, 4), dataset is (737280, 64, 64) uint8.

    # reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/dsprites.py
    """

    factor_names = ('shape', 'scale', 'orientation', 'position_x', 'position_y')
    factor_sizes = (3, 6, 40, 32, 32)  # TOTAL: 737280
    observation_shape = (64, 64, 1)  # TODO: reference implementation has colour variants

    dataset_url = 'https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5'

    hdf5_name = 'imgs'
    # minimum chunk size, no compression but good for random accesses
    hdf5_chunk_size = (1, 64, 64)

    def __init__(self, data_dir='data/dataset/dsprites', in_memory=False, force_download=False, force_preprocess=False):
        super().__init__(data_dir=data_dir, in_memory=in_memory, force_download=force_download, force_preprocess=force_preprocess)

    def __getitem__(self, idx):
        return super().__getitem__(idx) * 255  # for some reason uint8 is used as datatype, but only in range 0-1


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    from tqdm import tqdm

    for dat in tqdm(DSpritesData(in_memory=True, force_preprocess=True)):
        pass
