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
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from tqdm import tqdm

from disent.dataset.data import DiskGroundTruthData
from disent.dataset.data import Hdf5GroundTruthData
from disent.dataset.util.datafile import DataFile
from disent.dataset.util.datafile import DataFileHashed
from disent.dataset.util.datafile import DataFileHashedDl
from disent.dataset.data._groundtruth import NumpyFileGroundTruthData
from disent.util.inout.paths import modify_ext


log = logging.getLogger(__name__)

# ========================================================================= #
# mpi3d - conversion                                                        #
# ========================================================================= #


def resave_mpi3d_array(in_npz_path: str, out_h5_path: str, overwrite: bool = False):
    from disent.util.profiling import Timer
    from disent.dataset.util.formats.hdf5 import H5Builder
    # load the array
    with Timer('loading images into memory'):
        imgs = np.load(in_npz_path)['images']
        assert imgs.dtype == 'uint8'
        assert imgs.shape == (1036800, 64, 64, 3)
    # resave the array as hdf5
    with Timer('resaving images'):
        with H5Builder(out_h5_path, mode='atomic_w' if overwrite else 'atomic_x') as builder:
            builder.add_dataset_from_array(name='images', array=imgs, show_progress=True)


class DataFileMpi3dResaved(DataFileHashed):

    def __init__(
        self,
        mpi3d_datafile: DataFileHashedDl,
        # # - convert file name
        out_hash: Optional[Union[str, Dict[str, str]]],
        out_name: Optional[str] = None,
        # # - hash settings
        hash_type: str = 'md5',
        hash_mode: str = 'fast',
    ):
        self._mpi3d_datafile = mpi3d_datafile
        super().__init__(
            file_name=modify_ext(self._mpi3d_datafile.out_name, 'h5') if (out_name is None) else out_name,
            file_hash=out_hash,
            hash_type=hash_type,
            hash_mode=hash_mode,
        )

    def _prepare(self, out_dir: str, out_file: str):
        log.debug('Preparing Orig Mpi3d Data:')
        mpi3d_path = self._mpi3d_datafile.prepare(out_dir)
        log.debug('Generating hdf5 Mpi3d Data:')
        resave_mpi3d_array(in_npz_path=mpi3d_path, out_h5_path=out_file, overwrite=True)

    # match the functionality of DataFileHashedDlH5
    dataset_name = 'images'


# ========================================================================= #
# mpi3d                                                                     #
# ========================================================================= #

class _Mpi3dMixin:

    # override

    name = 'mpi3d'
    factor_names = ('object_color', 'object_shape', 'object_size', 'camera_height', 'background_color', 'first_dof', 'second_dof')
    factor_sizes = (6, 6, 2, 3, 3, 40, 40)  # TOTAL: 1 036 800
    img_shape = (64, 64, 3)

    @property
    def datafile(self) -> DataFile:
        assert self.subset in self.MPI3D_DATAFILES, f'Invalid MPI3D subset: {repr(self.subset)} must be one of: {set(self.MPI3D_DATAFILES.keys())}'
        return self.MPI3D_DATAFILES[self.subset]

    # not implemented
    _subset: str

    @property
    def MPI3D_DATAFILES(self) -> Dict[str, DataFile]:
        raise NotImplementedError

    @property
    def subset(self) -> str:
        return self._subset


class Mpi3dNumpyData(_Mpi3dMixin, NumpyFileGroundTruthData):
    """
    MPI3D Dataset
    - https://github.com/rr-learning/disentanglement_dataset

    reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/mpi3d.py
    """

    # override NumpyFileGroundTruthData
    data_key = 'images'

    # override _Mpi3dMixin
    MPI3D_DATAFILES = {
        'toy':        DataFileHashedDl(uri='https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz',       uri_hash={'fast': '146138e36ff495e77ceacdc8cf14c37e', 'full': '55889cb7c7dfc655d6e0277beee88868'}),
        'realistic':  DataFileHashedDl(uri='https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_realistic.npz', uri_hash={'fast': '96c8ff1155dd61f79d3493edef9f19e9', 'full': '59a6225b88b635365f70c91b3e52f70f'}),
        'real':       DataFileHashedDl(uri='https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz',      uri_hash={'fast': 'e2941bba6f4a2b130edc5f364637b39e', 'full': '0f33f609918fb5c97996692f91129802'}),
    }

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, subset: str = 'realistic', transform=None):
        self._subset = subset
        log.warning('[WARNING]: mpi3d files are extremely large (over 11GB), you are trying to load these into memory.')
        super().__init__(data_root=data_root, prepare=prepare, transform=transform)


class Mpi3dHdf5Data(_Mpi3dMixin, Hdf5GroundTruthData):
    """
    MPI3D Dataset
    - https://github.com/rr-learning/disentanglement_dataset

    reference implementation: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/mpi3d.py
    """

    MPI3D_DATAFILES = {
        'toy':        DataFileMpi3dResaved(mpi3d_datafile=Mpi3dNumpyData.MPI3D_DATAFILES['toy'],       out_hash={'fast': '32af615e306d336449a83c53ee897c3f', 'full': '???'}),
        'realistic':  DataFileMpi3dResaved(mpi3d_datafile=Mpi3dNumpyData.MPI3D_DATAFILES['realistic'], out_hash={'fast': 'c6bfa8ecec2549dbf7f0c29f853a6beb', 'full': '???'}),
        'real':       DataFileMpi3dResaved(mpi3d_datafile=Mpi3dNumpyData.MPI3D_DATAFILES['real'],      out_hash={'fast': '81b9c54c75e921a2465971911b096ac3', 'full': '???'}),
    }

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, subset: str = 'realistic', in_memory: bool = False, transform=None):
        self._subset = subset
        if in_memory:
            log.warning('[WARNING]: mpi3d files are extremely large (over 11GB), you are trying to load these into memory.')
        super().__init__(data_root=data_root, prepare=prepare, in_memory=in_memory, transform=transform)


class Mpi3dData(DiskGroundTruthData):

    # override
    name         = _Mpi3dMixin.name
    factor_names = _Mpi3dMixin.factor_names
    factor_sizes = _Mpi3dMixin.factor_sizes
    img_shape    = _Mpi3dMixin.img_shape

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, subset: str = 'realistic', in_memory: bool = False, transform=None):
        if in_memory:
            self._wrapped_mpi3d = Mpi3dNumpyData(data_root=data_root, prepare=False, subset=subset, transform=None)
        else:
            self._wrapped_mpi3d = Mpi3dHdf5Data(data_root=data_root, prepare=False, subset=subset, transform=None, in_memory=False)
        # initialize
        super().__init__(data_root=data_root, prepare=prepare, transform=transform)

    @property
    def datafiles(self) -> Sequence[DataFile]:
        return self._wrapped_mpi3d.datafiles

    def _get_observation(self, idx):
        return self._wrapped_mpi3d._get_observation(idx)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    def main():
        from disent.util.profiling import Timer

        # NAME:     | both-disk | both-mem | numpy-mem | hdf5-disk | hdf5-mem
        # ----------+-----------+----------+-----------+-----------+---------
        # real      | 908.512µs | 10.905s  | 10.901s   | 753.156µs | 1m:11s
        # realistic | 12.397ms  | 11.039s  | 11.024s   | 764.293µs | 54.281s
        # toy       | 12.770ms  | 10.992s  | 10.984s   | 781.021µs | 38.165s

        for subset in ['real', 'realistic', 'toy']:
            print('='*100)
            print(subset)
            print('='*100)

            with Timer('both-disk'): both_dist = Mpi3dData(subset=subset, prepare=True, in_memory=False)
            with Timer('hdf5-disk'): hdf5_disk = Mpi3dHdf5Data(subset=subset, prepare=True, in_memory=False)
            with Timer('both-mem'):  both_mem  = Mpi3dData(subset=subset, prepare=True, in_memory=True)
            with Timer('numpy-mem'): numpy_mem = Mpi3dNumpyData(subset=subset, prepare=True)
            with Timer('hdf5-mem'):  hdf5_mem  = Mpi3dHdf5Data(subset=subset, prepare=True, in_memory=True)  # this is slow!

            assert len(both_dist) == len(hdf5_disk)
            assert len(hdf5_disk)  == len(both_mem)
            assert len(both_mem) == len(numpy_mem)
            assert len(numpy_mem) == len(hdf5_mem)

            # check equivalence
            for (bd, hd, bm, nm, hm) in tqdm(zip(both_dist, hdf5_disk, both_mem, numpy_mem, hdf5_mem)):
                assert np.allclose(bd, hd)
                assert np.allclose(hd, bm)
                assert np.allclose(bm, nm)
                assert np.allclose(nm, hm)

    main()
