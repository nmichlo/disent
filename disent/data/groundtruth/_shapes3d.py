from disent.data.groundtruth.base import Hdf5PreprocessedGroundTruthData


# ========================================================================= #
# shapes3d                                                                  #
# ========================================================================= #


class Shapes3dData(Hdf5PreprocessedGroundTruthData):
    """
    3D Shapes Dataset:
    - https://github.com/deepmind/3d-shapes

    Files:
        - direct:   https://storage.googleapis.com/3d-shapes/3dshapes.h5
          redirect: https://storage.cloud.google.com/3d-shapes/3dshapes.h5
          info:     https://console.cloud.google.com/storage/browser/_details/3d-shapes/3dshapes.h5

    reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/shapes3d.py
    """

    dataset_url = 'https://storage.googleapis.com/3d-shapes/3dshapes.h5'

    factor_names = ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')
    factor_sizes = (10, 10, 10, 8, 4, 15)  # TOTAL: 480000
    observation_shape = (64, 64, 3)

    hdf5_name = 'images'
    # minimum chunk size, no compression but good for random accesses
    hdf5_chunk_size = (1, 64, 64, 3)

    def __init__(self, data_dir='data/dataset/3dshapes', in_memory=False, force_download=False, force_preprocess=False):
        super().__init__(data_dir=data_dir, in_memory=in_memory, force_download=force_download, force_preprocess=force_preprocess)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    dataset = Shapes3dData(data_dir='data/dataset/shapes3d-1-64-64-3')
    # pair_dataset = PairedVariationDataset(dataset, k='uniform')

    # # test that dimensions are resampled correctly, and only differ by a certain number of factors, not all.
    # for i in range(10):
    #     idx = np.random.randint(len(dataset))
    #     a, b = pair_dataset.sample_pair_factors(idx)
    #     print(all(dataset.idx_to_pos(idx) == a), '|', a, '&', b, ':', [int(v) for v in (a == b)])
    #     a, b = dataset.pos_to_idx([a, b])
    #     print(a, b)
    #     dataset[a], dataset[b]
