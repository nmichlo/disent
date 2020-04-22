import h5py
import torch
from PIL import Image
from disent.dataset.ground_truth.base import PreprocessedDownloadableGroundTruthDataset


# ========================================================================= #
# shapes3d                                                                  #
# ========================================================================= #


class Shapes3dDataset(PreprocessedDownloadableGroundTruthDataset):
    """
    3D Shapes Dataset:
    - https://github.com/deepmind/3d-shapes

    Files:
        - direct:   https://storage.googleapis.com/3d-shapes/3dshapes.h5
          redirect: https://storage.cloud.google.com/3d-shapes/3dshapes.h5
          info:     https://console.cloud.google.com/storage/browser/_details/3d-shapes/3dshapes.h5
    """

    dataset_url = 'https://storage.googleapis.com/3d-shapes/3dshapes.h5'

    factor_names = ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')
    factor_sizes = (10, 10, 10, 8, 4, 15)
    observation_shape = (64, 64, 3)

    def __init__(self, data_dir='data', transform=None, force_download=False, force_preprocess=False):
        super().__init__(data_dir=data_dir, force_download=force_download, force_preprocess=force_preprocess)
        self.transform = transform

    def __getitem__(self, indices):
        if torch.is_tensor(indices):
            indices = indices.tolist()

        with h5py.File(self.dataset_path, 'r') as db:
            image = db['images'][indices]

        # https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
        # PIL Image so that this is consistent with other datasets
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image

    def _preprocess_dataset(self, path_src, path_dst):
        from disent.dataset.util.hdf5 import hdf5_resave_dataset

        # hdf5_3dshapes_resave(image_chunks=15000, image_block_size=4, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)  # *[ 29.285 MiB] 3dshapes_gzip4_15000-4-1_4096__3.23eps.h5     |    3.23 | 175.781 MiB | 234.375 KiB
        # hdf5_3dshapes_resave(image_chunks=256, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)   # *[ 74.613 MiB] 3dshapes_gzip4_256-16-1_4096__150.57eps.h5    |  150.57 |   3.000 MiB |  64.000 KiB
        # hdf5_3dshapes_resave(image_chunks=128, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)   # %[ 82.435 MiB] 3dshapes_gzip4_128-16-1_4096__254.62eps.h5    |  254.62 |   1.500 MiB |  32.000 KiB
        # hdf5_3dshapes_resave(image_chunks=64, image_block_size=16, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)    # *[ 97.912 MiB] 3dshapes_gzip4_64-16-1_4096__1954.21eps.h5    | 1954.21 | 768.000 KiB |  16.000 KiB
        # hdf5_3dshapes_resave(image_chunks=64, image_block_size=32, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)    # %[106.542 MiB] 3dshapes_gzip4_64-32-1_4096__3027.06eps.h5    | 3027.06 | 768.000 KiB |  64.000 KiB
        # hdf5_3dshapes_resave(image_chunks=64, image_block_size=16, image_channels=3, label_chunks=4096, stop_early=False, dry_run=False, test=True)    # *[121.457 MiB] 3dshapes_gzip4_64-16-3_4096__3458.68eps.h5    | 3458.68 | 768.000 KiB |  48.000 KiB
        # hdf5_3dshapes_resave(image_chunks=32, image_block_size=32, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)    # *[131.429 MiB] 3dshapes_gzip4_32-32-1_4096__2976.50eps.h5    | 2976.50 | 384.000 KiB |  32.000 KiB
        # hdf5_3dshapes_resave(image_chunks=32, image_block_size=64, image_channels=1, label_chunks=4096, stop_early=False, dry_run=False, test=True)    # %[175.186 MiB] 3dshapes_gzip4_32-64-1_4096__3747.35eps.h5    | 3747.35 | 384.000 KiB | 128.000 KiB
        # hdf5_3dshapes_resave(image_chunks=32, image_block_size=32, image_channels=3, label_chunks=4096, stop_early=False, dry_run=False, test=True)    # *[196.880 MiB] 3dshapes_gzip4_32-32-3_4096__4300.62eps.h5    | 4300.62 | 384.000 KiB |  96.000 KiB
        # hdf5_3dshapes_resave(image_chunks=32, image_block_size=64, image_channels=3, label_chunks=4096, stop_early=False, dry_run=False, test=True)    # %[202.748 MiB] 3dshapes_gzip4_32-64-3_4096__4784.69eps.h5    | 4784.69 | 384.000 KiB | 384.000 KiB

        # settings
        image_chunks, image_block_size, image_channels = 64, 32, 1
        label_chunks = 4096
        compression, compression_lvl = 'gzip', 4

        # resave datasets
        with h5py.File(path_src, 'r') as inp_data:
            with h5py.File(path_dst, 'w') as out_data:
                hdf5_resave_dataset(inp_data, out_data, 'images', (image_chunks, image_block_size, image_block_size, image_channels), compression, compression_lvl)
                hdf5_resave_dataset(inp_data, out_data, 'labels', (label_chunks, 6), compression, compression_lvl)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    import numpy as np
    from disent.dataset.ground_truth.base import PairedVariationDataset

    dataset = Shapes3dDataset()
    pair_dataset = PairedVariationDataset(dataset, k='uniform')

    # test that dimensions are resampled correctly, and only differ by a certain number of factors, not all.
    for i in range(10):
        idx = np.random.randint(len(dataset))
        a, b = pair_dataset.sample_pair_factors(idx)
        print(all(dataset.idx_to_pos(idx) == a), '|', a, '&', b, ':', [int(v) for v in (a == b)])
        a, b = dataset.pos_to_idx([a, b])
        print(a, b)
        dataset[a], dataset[b]
