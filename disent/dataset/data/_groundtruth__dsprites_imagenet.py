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

import csv
import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import psutil
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from disent.dataset.data._groundtruth import _DiskDataMixin
from disent.dataset.data._groundtruth import _Hdf5DataMixin
from disent.dataset.data._groundtruth__dsprites import DSpritesData
from disent.dataset.util.datafile import DataFileHashedDlGen
from disent.dataset.util.hdf5 import H5Builder
from disent.util.inout.files import AtomicSaveFile
from disent.util.iters import LengthIter
from disent.util.math.random import random_choice_prng


log = logging.getLogger(__name__)


# ========================================================================= #
# load imagenet-tiny meta data                                              #
# ========================================================================= #


def _read_csv(path, col_types=None, n_rows: int = None, n_cols: int = None, flatten: bool = False, delimiter='\t') -> list:
    assert (n_rows is None) or n_rows >= 1
    assert (n_cols is None) or n_cols >= 1
    assert (not flatten) or (n_cols is None) or (n_cols == 1)
    # read the CVS entries
    rows = []
    with open(path, 'r') as fp:
        reader = csv.reader(fp, delimiter=delimiter)
        # read each row of the CSV and process it
        for row in reader:
            if n_cols is not None:
                assert len(row) == n_cols, f'invalid number of columns, got: {len(row)}, required: {n_cols}'
            if col_types is not None:
                assert len(col_types) == len(row), f'invalid number of col_types entries, got: {len(col_types)}, required: {len(row)}'
                row = [(v if t is None else t(v)) for t, v in zip(col_types, row)]
            if flatten:
                rows.extend(row)
            else:
                rows.append(row)
    # check we have the right number of rows
    if n_rows is not None:
        assert len(rows) == n_rows, f'invalid number of rows, got: {len(rows)}, required: {n_rows}'
    return rows


def load_imagenet_tiny_meta(raw_data_dir):
    """
    tiny-imagenet-200.zip contains:
        1. /tiny-imagenet-200/wnids.txt                    # <class>
        2. /tiny-imagenet-200/words.txt                    # <class> <description>
        3. /tiny-imagenet-200/train/n#/n#_boxes.txt        # <img_name>         <i> <j> <k> <l>
           /tiny-imagenet-200/train/n#/images/n#_#.JPEG
        4. /tiny-imagenet-200/val/images/val_#.JPEG
           /tiny-imagenet-200/val/val_annotations.txt      # <img_name> <class> <i> <j> <k> <l>
        5. /tiny-imagenet-200/test/images/test_#.JPEG
    """
    root = Path(raw_data_dir)
    # 1. read the classes
    wnids = _read_csv(root.joinpath('wnids.txt'), col_types=(str,), n_rows=200, flatten=True)
    assert len(wnids) == 200
    # 2. read the class descriptions
    cls_descs = {k: v for k, v in _read_csv(root.joinpath('words.txt'), col_types=(str, str), n_rows=82115)}
    cls_descs = {k: cls_descs[k] for k in wnids}
    assert len(cls_descs) == 200
    # 3. load the training data
    train_meta = []
    for cls_name in wnids:
        cls_folder = root.joinpath('train', cls_name)
        cls_meta = _read_csv(cls_folder.joinpath(f'{cls_name}_boxes.txt'), col_types=(str, int, int, int, int), n_rows=500)
        cls_meta = [(os.path.join('train', cls_name, name), cls_name, (i, j, k, l)) for name, i, j, k, l in cls_meta]
        train_meta.extend(cls_meta)
    assert len(train_meta) == 100_000
    # 4. read the validation data
    val_meta = _read_csv(root.joinpath('val', 'val_annotations.txt'), col_types=(str, str, int, int, int, int), n_rows=10000)
    val_meta = [(os.path.join('val', 'images', name), cls, (i, j, k, l)) for name, cls, i, j, k, l in val_meta]
    assert len(val_meta) == 10_000
    # 5. load the test data
    test_meta = [os.path.join('test', 'images', path.name) for path in root.joinpath('test', 'images').glob('test_*.JPEG')]
    assert len(test_meta) == 10_000
    # return data
    return train_meta, val_meta, test_meta, cls_descs


# ========================================================================= #
# load imagenet-tiny data                                                   #
# ========================================================================= #


class NumpyFolder(ImageFolder):
    def __getitem__(self, idx):
        img, cls = super().__getitem__(idx)
        return np.array(img)


def load_imagenet_tiny_data(raw_data_dir):
    data = NumpyFolder(os.path.join(raw_data_dir, 'train'))
    data = DataLoader(data, batch_size=64, num_workers=min(16, psutil.cpu_count(logical=False)), shuffle=False, drop_last=False, collate_fn=lambda x: x)
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


class DSpritesImagenetData(DSpritesData):
    """
    DSprites that has imagenet images in the background.
    """

    # keep the dataset name as dsprites so we don't have to download and reprocess it...
    name = 'dsprites'

    # original dsprites it only (64, 64, 1) imagenet adds the colour channel
    img_shape = (64, 64, 3)

    def __init__(self, brightness: float = 1.0, invert: bool = False, data_root: Optional[str] = None, prepare: bool = False, in_memory=False, transform=None):
        super().__init__(data_root=data_root, prepare=prepare, in_memory=in_memory, transform=transform)
        # checks
        assert 0 <= brightness <= 1, f'incorrect brightness value: {repr(brightness)}, must be in range [0, 1]'
        self._brightness = brightness
        self._invert = invert
        # handle the imagenet data
        self._imagenet_tiny = ImageNetTinyData(
            data_root=data_root,
            prepare=prepare,
            in_memory=in_memory,
            transform=None,
        )
        # deterministic randomization of the imagenet order
        self._imagenet_order = random_choice_prng(
            len(self._imagenet_tiny),
            size=len(self),
            seed=42,
        )

    # we need to combine the two dataset images
    def _get_observation(self, idx):
        # dsprites contains only {0, 255} for values
        # we can directly use these values to mask the imagenet image
        bg = self._imagenet_tiny[self._imagenet_order[idx]]
        fg = self._data[idx].repeat(3, axis=-1)
        # compute background
        obs = (bg * self._brightness).astype('uint8')
        # set foreground
        if self._invert:
            obs[fg <= 127] = 0
        else:
            obs[fg > 127] = 255
        # checks
        return obs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    data = DSpritesImagenetData(prepare=True)

    grid = np.array([data[i*24733] for i in np.arange(16)]).reshape([4, 4, *data.img_shape])

    plt_subplots_imshow(grid, show=True)
