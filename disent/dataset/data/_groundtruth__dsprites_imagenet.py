#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import logging
import os
import shutil
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import psutil
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from disent.dataset.data import GroundTruthData
from disent.dataset.data._groundtruth import _DiskDataMixin
from disent.dataset.data._groundtruth import _Hdf5DataMixin
from disent.dataset.data._groundtruth__dsprites import DSpritesData
from disent.dataset.transform import ToImgTensorF32
from disent.dataset.util.datafile import DataFileHashedDlGen
from disent.dataset.util.hdf5 import H5Builder
from disent.dataset.util.stats import compute_data_mean_std
from disent.util.inout.files import AtomicSaveFile
from disent.util.iters import LengthIter
from disent.util.math.random import random_choice_prng


log = logging.getLogger(__name__)


# ========================================================================= #
# load imagenet-tiny data                                                   #
# ========================================================================= #


class NumpyFolder(ImageFolder):
    def __getitem__(self, idx):
        img, cls = super().__getitem__(idx)
        return np.array(img)


def _noop(x):
    return x


def load_imagenet_tiny_data(raw_data_dir):
    data = NumpyFolder(os.path.join(raw_data_dir, 'train'))
    data = DataLoader(data, batch_size=64, num_workers=min(16, psutil.cpu_count(logical=False)), shuffle=False, drop_last=False, collate_fn=_noop)
    # load data - this is a bit memory inefficient doing it like this instead of with a loop into a pre-allocated array
    imgs = np.concatenate(list(tqdm(data, 'loading')), axis=0)
    assert imgs.shape == (100_000, 64, 64, 3)
    return imgs


def resave_imagenet_tiny_archive(orig_zipped_file, new_save_file, overwrite=False, h5_dataset_name: str = 'data'):
    """
    Convert a imagenet tiny archive to an hdf5 or numpy file depending on the file extension.
    Uncompressing the contents of the archive into a temporary directory in the same folder,
    loading the images, then converting.
    """
    _, ext = os.path.splitext(new_save_file)
    assert ext in {'.npz', '.h5'}, f'unsupported save extension: {repr(ext)}, must be one of: {[".npz", ".h5"]}'
    # extract zipfile into temp dir
    with TemporaryDirectory(prefix='unzip_imagenet_tiny_', dir=os.path.dirname(orig_zipped_file)) as temp_dir:
        log.info(f"Extracting into temporary directory: {temp_dir}")
        shutil.unpack_archive(filename=orig_zipped_file, extract_dir=temp_dir)
        images = load_imagenet_tiny_data(raw_data_dir=os.path.join(temp_dir, 'tiny-imagenet-200'))
    # save the data
    with AtomicSaveFile(new_save_file, overwrite=overwrite) as temp_file:
        # check the mode
        with H5Builder(temp_file, 'atomic_w') as builder:
            builder.add_dataset_from_array(
                name=h5_dataset_name,
                array=images,
                chunk_shape='batch',
                compression_lvl=4,
                attrs=None,
                show_progress=True,
            )


# ========================================================================= #
# cars3d data object                                                        #
# ========================================================================= #


class ImageNetTinyDataFile(DataFileHashedDlGen):
    """
    download the cars3d dataset and convert it to a hdf5 file.
    """

    dataset_name: str = 'data'

    def _generate(self, inp_file: str, out_file: str):
        resave_imagenet_tiny_archive(orig_zipped_file=inp_file, new_save_file=out_file, overwrite=True, h5_dataset_name=self.dataset_name)


class ImageNetTinyData(_Hdf5DataMixin, _DiskDataMixin, Dataset, LengthIter):

    name = 'imagenet_tiny'

    datafile_imagenet_h5 = ImageNetTinyDataFile(
        uri='http://cs231n.stanford.edu/tiny-imagenet-200.zip',
        uri_hash={'fast': '4d97ff8efe3745a3bba9917d6d536559', 'full': '90528d7ca1a48142e341f4ef8d21d0de'},
        file_hash={'fast': '9c23e8ec658b1ec9f3a86afafbdbae51', 'full': '4c32b0b53f257ac04a3afb37e3a4204e'},
        uri_name='tiny-imagenet-200.zip',
        file_name='tiny-imagenet-200.h5',
        hash_mode='full'
    )

    datafiles = (datafile_imagenet_h5,)

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, in_memory=False, transform=None):
        super().__init__()
        self._transform = transform
        # initialize mixin
        self._mixin_disk_init(
            data_root=data_root,
            prepare=prepare,
        )
        self._mixin_hdf5_init(
            h5_path=os.path.join(self.data_dir, self.datafile_imagenet_h5.out_name),
            h5_dataset_name=self.datafile_imagenet_h5.dataset_name,
            in_memory=in_memory,
        )

    def __getitem__(self, idx: int):
        obs = self._data[idx]
        if self._transform is not None:
            obs = self._transform(obs)
        return obs


# ========================================================================= #
# dataset_dsprites                                                          #
# ========================================================================= #


class DSpritesImagenetData(GroundTruthData):
    """
    DSprites that has imagenet images in the background.
    """

    name = 'dsprites_imagenet'

    # original dsprites it only (64, 64, 1) imagenet adds the colour channel
    img_shape = (64, 64, 3)
    factor_names = DSpritesData.factor_names
    factor_sizes = DSpritesData.factor_sizes

    def __init__(self, visibility: int = 100, mode: str = 'bg', data_root: Optional[str] = None, prepare: bool = False, in_memory=False, transform=None):
        super().__init__(transform=transform)
        # check visibility and convert to ratio
        assert isinstance(visibility, int), f'incorrect visibility percentage type, expected int, got: {type(visibility)}'
        assert 0 <= visibility <= 100, f'incorrect visibility percentage: {repr(visibility)}, must be in range [0, 100]. '
        self._visibility = visibility / 100
        # check mode and convert to foreground boolean
        assert mode in {'bg', 'fg'}, f'incorrect mode: {repr(mode)}, must be one of: ["bg", "fg"]'
        self._foreground = (mode == 'fg')
        # handle the datasets
        self._dsprites = DSpritesData(data_root=data_root, prepare=prepare, in_memory=in_memory, transform=None)
        self._imagenet = ImageNetTinyData(data_root=data_root, prepare=prepare, in_memory=in_memory, transform=None)
        # deterministic randomization of the imagenet order
        self._imagenet_order = random_choice_prng(
            len(self._imagenet),
            size=len(self),
            seed=42,
        )

    def _get_observation(self, idx):
        # we need to combine the two dataset images
        # dsprites contains only {0, 255} for values
        # we can directly use these values to mask the imagenet image
        bg = self._imagenet[self._imagenet_order[idx]]
        fg = self._dsprites[idx].repeat(3, axis=-1)
        # compute background
        # set foreground
        r = self._visibility
        if self._foreground:
            # lerp content to white, and then insert into fg regions
            # r*bg + (1-r)*255
            obs = (r*bg + ((1-r)*255)).astype('uint8')
            obs[fg <= 127] = 0
        else:
            # lerp content to black, and then insert into bg regions
            # r*bg + (1-r)*000
            obs = (r*bg).astype('uint8')
            obs[fg > 127] = 255
        # checks
        return obs


# ========================================================================= #
# STATS                                                                     #
# ========================================================================= #


"""
dsprites_fg_1.0
    vis_mean: [0.02067051643494642, 0.018688392816012946, 0.01632900510079384]
    vis_std: [0.10271307751834059, 0.09390213983525653, 0.08377594259970281]
dsprites_fg_0.8
    vis_mean: [0.024956427531012196, 0.02336780403840578, 0.021475119672280243]
    vis_std: [0.11864125016313823, 0.11137998105649799, 0.10281424917834255]
dsprites_fg_0.6
    vis_mean: [0.029335176871153983, 0.028145355435322966, 0.026731731769287146]
    vis_std: [0.13663242436043319, 0.13114320478634894, 0.1246542727733097]
dsprites_fg_0.4
    vis_mean: [0.03369999506331255, 0.03290657349801835, 0.03196482946320608]
    vis_std: [0.155514074438101, 0.1518464537731621, 0.14750944591836743]
dsprites_fg_0.2
    vis_mean: [0.038064750024334834, 0.03766780505193579, 0.03719798677641122]
    vis_std: [0.17498878664096565, 0.17315570657628318, 0.1709923319496426]
dsprites_bg_1.0
    vis_mean: [0.5020433619489952, 0.47206398913310593, 0.42380018909780404]
    vis_std: [0.2505510666843685, 0.25007259803668697, 0.2562415603123114]
dsprites_bg_0.8
    vis_mean: [0.40867981393820857, 0.38468564002021527, 0.34611573047508204]
    vis_std: [0.22048328737091344, 0.22102216869942384, 0.22692977053753477]
dsprites_bg_0.6
    vis_mean: [0.31676960943447674, 0.29877166834408025, 0.2698556821388113]
    vis_std: [0.19745897110349003, 0.1986606891520453, 0.203808842880044]
dsprites_bg_0.4
    vis_mean: [0.2248598986983768, 0.21285772298967615, 0.19359577132944206]
    vis_std: [0.1841631708032332, 0.18554895825833284, 0.1893568926398198]
dsprites_bg_0.2
    vis_mean: [0.13294969414492142, 0.12694375140936273, 0.11733572285575933]
    vis_std: [0.18311250427586276, 0.1840916474752131, 0.18607373519458442]
"""


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)


    def compute_all_stats():
        from disent.util.visualize.plot import plt_subplots_imshow

        def compute_stats(visibility, mode):
            # plot images
            data = DSpritesImagenetData(prepare=True, visibility=visibility, mode=mode)
            grid = np.array([data[i*24733] for i in np.arange(16)]).reshape([4, 4, *data.img_shape])
            plt_subplots_imshow(grid, show=True, title=f'{DSpritesImagenetData.name} visibility={repr(visibility)} mode={repr(mode)}')
            # compute stats
            name = f'dsprites_{mode}_{visibility}'
            data = DSpritesImagenetData(prepare=True, visibility=visibility, mode=mode, transform=ToImgTensorF32())
            mean, std = compute_data_mean_std(data, batch_size=256, num_workers=min(psutil.cpu_count(logical=False), 64), progress=True)
            print(f'{name}\n    vis_mean: {mean.tolist()}\n    vis_std: {std.tolist()}')
            # return stats
            return name, mean, std

        # compute common stats
        stats = []
        for mode in ['fg', 'bg']:
            for vis in [100, 80, 60, 40, 20]:
                stats.append(compute_stats(vis, mode))

        # print once at end
        for name, mean, std in stats:
            print(f'{name}\n    vis_mean: {mean.tolist()}\n    vis_std: {std.tolist()}')

    compute_all_stats()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
