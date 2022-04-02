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
from tempfile import TemporaryDirectory
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from disent.dataset.data import DiskGroundTruthData
from disent.dataset.util.datafile import DataFileHashed

log = logging.getLogger(__name__)

# ========================================================================= #
# helper                                                                    #
# ========================================================================= #


def fetch_sprite_components() -> Tuple[np.array, np.array]:
    try:
        import git
    except ImportError:
        logging.error('GitPython not found! Please install it: `pip install GitPython`')
        exit(1)
    # store files in a temporary directory
    with TemporaryDirectory(suffix='sprites') as temp_dir:
        # clone the files into the temp dir
        git.Repo.clone_from('https://github.com/YingzhenLi/Sprites', temp_dir)
        # get all the components!
        component_sheets: List[np.ndarray] = []
        component_names = ['bottomwear', 'topwear', 'hair', 'eyes', 'shoes', 'body']
        for sprites_folder in component_names:
            # append all sprite sheets for the current component
            sheets = []
            for sheet_name in sorted(os.listdir(os.path.join(temp_dir, sprites_folder))):
                img_path = os.path.join(temp_dir, sprites_folder, sheet_name)
                img = np.array(
                    Image.open(img_path).convert('RGBA')
                )  # imageio sometimes doesnt load these all in the same way, we explicitly set the format as RGBA!
                sheets.append(img)
            # append all the sheets
            sheets = np.array(sheets)
            component_sheets.append(sheets)
            # print information
            log.debug(f'{sprites_folder} {sheets.shape} {sheets.dtype}')
    # done!
    return np.array(component_sheets, dtype=object), np.array(component_names, dtype=object)


def save_sprite_components(out_file: str, sheets, names) -> NoReturn:
    # get the path and make the parant dirs
    out_file = os.path.abspath(out_file)
    log.debug(f'saving: {out_file}')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # save the data
    np.savez_compressed(out_file, sheets=sheets, names=names)


def load_sprite_components(in_file: str) -> Tuple[Tuple[np.ndarray, ...], Tuple[str, ...]]:
    dat = np.load(in_file, allow_pickle=True)
    return dat['sheets'], dat['names']


# ========================================================================= #
# datafiles                                                                 #
# ========================================================================= #


class DataFileSprites(DataFileHashed):
    """
    Download the Sprites GitHub repo and convert it
    to lists of arrays of images that are pickled.
    """

    def _prepare(self, out_dir: str, out_file: str) -> NoReturn:
        sheets, names = fetch_sprite_components()
        save_sprite_components(out_file, sheets=sheets, names=names)


# ========================================================================= #
# dataset_sprites                                                           #
# ========================================================================= #


class SpritesData(DiskGroundTruthData):
    name = 'sprites'
    factor_names = ('bottomwear', 'topwear', 'hair', 'eyes', 'shoes', 'body', 'action', 'rotation', 'frame')
    factor_sizes = (7, 7, 10, 5, 3, 7, 5, 4, 6)
    img_shape = (64, 64, 3)

    datafile = DataFileSprites(
        file_name='sprites.npz',
        file_hash={'fast': '5c739a7c8e59a20ec34439213036993a', 'full': '189baaca306cd305f51291c5decad18d'},
    )
    datafiles = (datafile,)

    # 21x13 tiles in size @ 64x64 px gives 1344x832 px total or 273 tiles total
    SPRITE_SHEET_ACTIONS = {
        'spellcard': {'back': range(0, 7), 'left': range(13, 20), 'front': range(26, 33), 'right': range(39, 46)},
        'thrust': {'back': range(52, 60), 'left': range(65, 73), 'front': range(78, 86), 'right': range(91, 99)},
        'walk': {'back': range(104, 113), 'left': range(117, 126), 'front': range(130, 139), 'right': range(143, 152)},
        'slash': {'back': range(156, 162), 'left': range(169, 175), 'front': range(182, 188), 'right': range(195, 201)},
        'shoot': {'back': range(208, 221), 'left': range(221, 234), 'front': range(234, 247), 'right': range(247, 260)},
        'hurt': {'front': range(260, 266)}
    }

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, transform=None):
        super().__init__(data_root=data_root, prepare=prepare, transform=transform)
        # load the data
        dat = np.load(os.path.join(self._data_dir, self.datafile.out_name), allow_pickle=True)
        self._names = dat['names']
        self._sheets = dat['sheets']

    def _get_observation(self, idx):
        # TODO: something here is wrong!
        # TODO: something here is wrong!
        # TODO: something here is wrong!
        *sheet_idxs, act, rot, frame = self.idx_to_pos(idx)
        print(self.idx_to_pos(idx), idx, sheet_idxs, (act, rot, frame))
        # extract the individual sheets
        sheets = (sheets[i] for i, sheets in zip(sheet_idxs, self._sheets))
        # extract the individual tiles
        y, x = act * 4 + rot, frame
        frames = [sheet[y*64:(y+1)*64, x*64:(x+1)*64] for sheet in sheets]
        # combine the frames in order
        img = frames[5][:, :, 0:3]  # body
        for i in [4, 0, 1, 3, 2]:  # shoes, bottomwear, topwear, eyes, hair
            mask = frames[i][:, :, 3] > 128
            img[:, :, 0][mask] = frames[i][:, :, 0][mask]
            img[:, :, 1][mask] = frames[i][:, :, 1][mask]
            img[:, :, 2][mask] = frames[i][:, :, 2][mask]
        # done
        return img


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    data = SpritesData(prepare=True)

    for i in tqdm(data):
        pass
