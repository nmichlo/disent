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

# custom episodes -- base
from disent.dataset.data._episodes import BaseEpisodesData
from disent.dataset.data._episodes__custom import EpisodesPickledData
from disent.dataset.data._episodes__custom import EpisodesDownloadZippedPickledData

# raw -- groundtruth
from disent.dataset.data._groundtruth import ArrayGroundTruthData
from disent.dataset.data._groundtruth import SelfContainedHdf5GroundTruthData

# raw
from disent.dataset.data._raw import ArrayDataset
from disent.dataset.data._raw import Hdf5Dataset

# groundtruth -- base
from disent.dataset.data._groundtruth import GroundTruthData
from disent.dataset.data._groundtruth import DiskGroundTruthData
from disent.dataset.data._groundtruth import NumpyFileGroundTruthData
from disent.dataset.data._groundtruth import Hdf5GroundTruthData

# groundtruth -- impl
from disent.dataset.data._groundtruth__cars3d import Cars3dData
from disent.dataset.data._groundtruth__dsprites import DSpritesData
from disent.dataset.data._groundtruth__mpi3d import Mpi3dData
from disent.dataset.data._groundtruth__norb import SmallNorbData
from disent.dataset.data._groundtruth__shapes3d import Shapes3dData

# groundtruth -- impl synthetic
from disent.dataset.data._groundtruth__xyobject import XYObjectData
from disent.dataset.data._groundtruth__xyobject import XYObjectShadedData
