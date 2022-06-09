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

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from disent.dataset.data import DiskGroundTruthData
from disent.dataset.data import GroundTruthData
from disent.dataset.util.datafile import DataFileHashed


log = logging.getLogger(__name__)

# ========================================================================= #
# helper                                                                    #
# ========================================================================= #


SPRITES_REPO = 'https://github.com/YingzhenLi/Sprites'
SPRITES_REPO_COMMIT_SHA = '3ce4048c5227802bd8f1888e293fd3afdba91c0c'


def fetch_sprite_components() -> Tuple[np.array, np.array]:
    try:
        import git
    except ImportError:
        log.error('GitPython not found! Please install it: `pip install GitPython`')
        exit(1)
    # store files in a temporary directory
    with TemporaryDirectory(suffix='sprites') as temp_dir:
        # clone the files into the temp dir
        log.info(f'Generating sprites data, temporarily cloning: {SPRITES_REPO} to {temp_dir}`')
        repo = git.Repo.clone_from(SPRITES_REPO, temp_dir, no_checkout=True)
        repo.git.checkout(SPRITES_REPO_COMMIT_SHA)
        # get all the components!
        component_sheets: List[np.ndarray] = []
        component_names = ['bottomwear', 'topwear', 'hair', 'eyes', 'shoes', 'body']
        for sprites_folder in component_names:
            # append all sprite sheets for the current component
            sheets = []
            for sheet_name in sorted(os.listdir(os.path.join(temp_dir, sprites_folder))):
                img_path = os.path.join(temp_dir, sprites_folder, sheet_name)
                img = np.array(Image.open(img_path).convert('RGBA'))  # imageio sometimes doesnt load these all in the same way, we explicitly set the format as RGBA!
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


class SpritesAllData(DiskGroundTruthData):
    """
    Custom version of sprites, with the data obtained from:
    https://github.com/YingzhenLi/Sprites
    """

    name = 'sprites'
    factor_names = ('bottomwear', 'topwear', 'hair', 'eyes', 'shoes', 'body', 'action', 'rotation', 'frame')
    factor_sizes = (7, 7, 10, 5, 3, 7, 5, 4, 6)  # 6_174_000
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

    def sample_random_frames(self, alpha: bool = True, combined: bool = False) -> List[np.ndarray]:
        return self.get_frames(idx=self.sample_indices(), alpha=alpha, combined=combined)

    def get_frames(self, idx: Optional[int], alpha: bool = True, combined: bool = False) -> List[np.ndarray]:
        *sheet_idxs, act, rot, frame = self.idx_to_pos(idx)
        # extract the individual sheets
        sheets = (sheets[i] for i, sheets in zip(sheet_idxs, self._sheets))
        # extract the individual tiles
        y, x = act * 4 + rot, frame
        # return individual frames
        frames = [sheet[y*64:(y+1)*64, x*64:(x+1)*64, :] for sheet in sheets]
        # return the combined frame
        if combined:
            frames.append(self.combine_frames(frames, alpha=True))
        # return the final result
        if not alpha:
            return [frame[:, :, :3] for frame in frames]
        return frames

    def combine_frames(self, frames_rgba, alpha: bool = True):
        # combine the frames in order
        img = np.zeros((64, 64, 4) if alpha else (64, 64, 3), dtype='uint8')
        for i in [5, 4, 0, 1, 3, 2]:  # body, shoes, bottomwear, topwear, eyes, hair
            mask = frames_rgba[i][:, :, 3] > 128
            img[:, :, 0][mask] = frames_rgba[i][:, :, 0][mask]
            img[:, :, 1][mask] = frames_rgba[i][:, :, 1][mask]
            img[:, :, 2][mask] = frames_rgba[i][:, :, 2][mask]
            # union for the mask
            if alpha:
                img[:, :, 3][mask] = 255
        # return the result
        return img

    def _get_observation(self, idx):
        frames = self.get_frames(idx, alpha=True, combined=False)
        img = self.combine_frames(frames, alpha=False)
        return img


class SpritesPartialData(GroundTruthData):
    """
    Same as `SpritesOrigData` but the backwards facing `rotation` is removed because it
    hides some of the other ground-truth factors. Similarely, the 5th `bottomwear` and `topwear`
    are removed too because they are missing entries.
    """

    img_shape = SpritesAllData.img_shape
    factor_names = SpritesAllData.factor_names
    factor_sizes = (6, 6, 10, 5, 3, 7, 5, 3, 6)  # 3_402_000

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, transform=None):
        super().__init__(transform=transform)
        self._sprites = SpritesAllData(data_root=data_root, prepare=prepare, transform=None)

    def _offset_idx(self, idx: int) -> int:
        pos = self.idx_to_pos(idx)
        # convert to orig state space
        if pos[0] >= 5: pos[0] += 1  # no missing pants
        if pos[1] >= 5: pos[1] += 1  # no missing shirt
        if pos[7] >= 0: pos[7] += 1  # no backwards facing man
        idx = self._sprites.pos_to_idx(pos)
        # index in orig state space
        return idx

    def sample_random_frames(self, alpha: bool = True, combined: bool = False) -> List[np.ndarray]:
        return self._sprites.sample_random_frames(alpha=alpha, combined=combined)

    def get_frames(self, idx: int, alpha: bool = True, combined: bool = False) -> List[np.ndarray]:
        return self._sprites.get_frames(idx=self._offset_idx(idx), alpha=alpha, combined=combined)

    def combine_frames(self, frames_rgba, alpha: bool = True):
        return self._sprites.combine_frames(frames_rgba=frames_rgba, alpha=alpha)

    def _get_observation(self, idx):
        return self._sprites._get_observation(self._offset_idx(idx))


class SpritesData(SpritesPartialData):
    """
    Same as `SpritesOrigData` but the backwards facing `rotation` is removed because it
    hides some of the other ground-truth factors.
    """

    factor_sizes = (7, 7, 10, 5, 3, 7, 5, 3, 6)  # 4_630_500

    def _offset_idx(self, idx: int) -> int:
        pos = self.idx_to_pos(idx)
        # convert to orig state space
        if pos[7] >= 0: pos[7] += 1  # no backwards facing man
        idx = self._sprites.pos_to_idx(pos)
        # index in orig state space
        return idx


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    def main():
        from tqdm import tqdm
        from disent.util.visualize.plot import plt_imshow
        from disent.util.visualize.vis_util import make_image_grid

        data = SpritesAllData(prepare=True)

        # show frames and combined frames
        frames = data.sample_random_frames(True, True)
        plt_imshow(make_image_grid(frames, num_cols=-1), show=True)
        plt_imshow(make_image_grid([f[:, :, 0:3] for f in frames], num_cols=-1), show=True)
        plt_imshow(make_image_grid([f[:, :, 3:4] for f in frames], num_cols=-1), show=True)

        # plot factor traversals
        factors, indices = data.sample_random_factor_traversal_grid(num=10, return_indices=True)
        plt_imshow(make_image_grid([data[i] for i in indices.flatten()], num_cols=10, pad=4))
        plt.show()

        # check speeds
        for i, _ in enumerate(tqdm(SpritesAllData(prepare=True))):  # ~7443 it/s
            if i > 25000: break
        for i, _ in enumerate(tqdm(SpritesData(prepare=True))):      # ~6355 it/s
            if i > 25000: break

    main()
