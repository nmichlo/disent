import logging
import os
import shutil
import numpy as np
from scipy.io import loadmat
from disent.data.groundtruth.base import DownloadableGroundTruthData

log = logging.getLogger(__name__)


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
    factor_sizes = (4, 24, 183)  # TOTAL: 17568
    observation_shape = (128, 128, 3)

    dataset_urls = ['http://www.scottreed.info/files/nips2015-analogy-data.tar.gz']

    def __init__(self, data_dir='data/dataset/cars3d', force_download=False):
        super().__init__(data_dir=data_dir, force_download=force_download)

        converted_file = self._make_converted_file(data_dir, force_download)

        if not hasattr(self.__class__, '_DATA'):
            # store data on class
            self.__class__._DATA = np.load(converted_file)['images']

    def __getitem__(self, idx):
        return self.__class__._DATA[idx]

    def _make_converted_file(self, data_dir, force_download):
        # get files & folders
        zip_path = self.dataset_paths[0]
        dataset_dir = os.path.splitext(os.path.splitext(zip_path)[0])[0]  # remove .tar & .gz, name of directory after renaming
        images_dir = os.path.join(dataset_dir, 'cars')  # mesh folder inside renamed folder
        converted_file = os.path.join(dataset_dir, 'cars.npz')

        if not os.path.exists(converted_file) or force_download:
            # extract data if required
            if (not os.path.exists(images_dir)) or force_download:
                extract_dir = os.path.join(data_dir, 'data')  # directory after extracting, before renaming
                # make sure the extract directory doesnt exist
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
                if os.path.exists(dataset_dir):
                    shutil.rmtree(dataset_dir)
                # extract the files
                log.info(f'[UNZIPPING]: {zip_path} to {dataset_dir}')
                shutil.unpack_archive(zip_path, data_dir)
                # rename dir
                shutil.move(extract_dir, dataset_dir)

            images = self._load_cars3d_images(images_dir)
            log.info(f'[CONVERTING]: {converted_file}')
            np.savez(os.path.splitext(converted_file)[0], images=images)

        return converted_file

    @staticmethod
    def _load_cars3d_images(images_dir):
        images = []
        log.info(f'[LOADING]: {images_dir}')
        with open(os.path.join(images_dir, 'list.txt'), 'r') as img_names:
            for i, img_name in enumerate(img_names):
                img_path = os.path.join(images_dir, f'{img_name.strip()}.mat')
                img = loadmat(img_path)['im']
                img = img[..., None].transpose([4, 3, 5, 0, 1, 2])  # (128, 128, 3, 24, 4, 1) -> (4, 24, 1, 128, 128, 3)
                images.append(img)
        return np.concatenate(images, axis=2).reshape([-1, 128, 128, 3])  # (4, 24, 183, 128, 128, 3) -> (17568, 1, 128, 128, 3)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

if __name__ == '__main__':
    Cars3dData()
