#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2022 Nathan Juraj Michlo
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

from typing import Optional

import numpy as np
from torch.utils.data.dataset import Dataset

from disent.dataset.util.datafile import DataFileHashedDl
from disent.util.iters import LengthIter


# ========================================================================= #
# ident 3d data processing                                                #
# ========================================================================= #


def np_load_from_tar(tar_file: str, extract_path: str):
    import tarfile
    import io
    with tarfile.open(tar_file, 'r') as f:
        # TODO: this might actually use quite a lot of intermediate memory!
        array_file = io.BytesIO()
        array_file.write(f.extractfile(extract_path).read())
        array_file.seek(0)
        return np.load(array_file)


def tar_list_files(tar_file: str):
    import tarfile
    with tarfile.open(tar_file, 'r') as f:
        return f.list()


# ========================================================================= #
# ident 3d dataset                                                          #
# ========================================================================= #


class CasualIdent3dData(LengthIter, Dataset):

    DATAFILE_TRAIN = DataFileHashedDl(
        uri='https://zenodo.org/record/4784282/files/trainset.tar.gz',
        uri_hash={'full': 'acd98fda30eee75856dbbc7c54a27e45', 'fast': 'c04066aed920ecdc576a3169d0d6a68f'},
    )

    DATAFILE_TEST = DataFileHashedDl(
        uri='https://zenodo.org/record/4784282/files/testset.tar.gz',
        uri_hash={'full': 'c5d9d32d3737e241a2b12b968275fcb8', 'fast': '69dab8c27d7e09bc8866991b43107a5e'},
    )

    name = 'casualident3d'


class Ident3dData(LengthIter, Dataset):

    DATAFILE_TRAIN = DataFileHashedDl(
        uri='https://zenodo.org/record/4502485/files/3dident_train.tar',
        uri_hash={'full': 'df132b4dacf04fa28e1c8ca9d2168634', 'fast': '3f90ddf961370683f4cff9df6cf9ed98'},
    )

    DATAFILE_TEST = DataFileHashedDl(
        uri='https://zenodo.org/record/4502485/files/3dident_test.tar',
        uri_hash={'full': '35020bca16c9faa80cf0c926d265d678', 'fast': 'e65caa5c930b1d9bc7550a1741ebbf43'},
    )

    name = 'ident3d'

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, test: bool = False, transform=None):
        self._data_root = data_root
        self._test = test
        self._transform = transform
        # load everything
        if prepare:
            path = (self.DATAFILE_TEST if test else self.DATAFILE_TRAIN).prepare(out_dir)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        obs = self._data[idx]
        if self._transform is not None:
            obs = self._transform(obs)
        return obs


# ========================================================================= #
# main                                                                      #
# ========================================================================= #


if __name__ == '__main__':

    def main():
        import logging
        import gzip
        from disent.dataset.util.datafile import DataFileHashedDl

        logging.basicConfig(level=logging.DEBUG)

        print('loading data')
        path = CasualIdent3dData.DATAFILE_TRAIN.prepare('data/dataset/casual3dident')
        path = CasualIdent3dData.DATAFILE_TEST.prepare('data/dataset/casual3dident')

        path = Ident3dData.DATAFILE_TRAIN.prepare('data/dataset/casual3dident')
        path = Ident3dData.DATAFILE_TEST.prepare('data/dataset/casual3dident')


        # dat = np_load_from_tar(path, '3dident/test/raw_latents.npy')
        # idxs = np.lexsort(dat.T[::-1, :])
        # for i in range(10):
        #     item = (dat[idxs[i]] - dat[idxs[i+1]]).tolist()
        #     print(item)
        # print()
        #
        # dat = np_load_from_tar(path, '3dident/test/latents.npy')
        # idxs = np.lexsort(dat.T[::-1, :])
        # for i in range(10):
        #     item = (dat[idxs[i]] - dat[idxs[i+1]]).tolist()
        #     print(item)
        # print()

    main()
