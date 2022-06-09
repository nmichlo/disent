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

import numpy as np
from tqdm import tqdm
from disent.util.inout.files import AtomicSaveFile


# ========================================================================= #
# Save Numpy Files                                                          #
# ========================================================================= #


def save_dataset_array(array: np.ndarray, out_file: str, overwrite: bool = False, save_key: str = 'images'):
    assert array.ndim == 4, f'invalid array shape, got: {array.shape}, must be: (N, H, W, C)'
    assert array.dtype == 'uint8', f'invalid array dtype, got: {array.dtype}, must be: "uint8"'
    # save the data
    with AtomicSaveFile(out_file, overwrite=overwrite) as temp_file:
        np.savez_compressed(temp_file, **{save_key: array})


def save_resized_dataset_array(array: np.ndarray, out_file: str, size: int = 64, overwrite: bool = False, save_key: str = 'images', progress: bool = True):
    import torchvision.transforms.functional as F_tv
    from disent.dataset.data import ArrayDataset
    # checks
    assert out_file.endswith('.npz'), f'The output file must end with the extension: ".npz", got: {repr(out_file)}'
    # Get the transform -- copied from: ToImgTensorF32 / ToImgTensorU8
    def transform(obs):
        H, W, C = obs.shape
        obs = F_tv.to_pil_image(obs)
        obs = F_tv.resize(obs, size=[size, size])
        obs = np.array(obs)
        # add removed dimension!
        if obs.ndim == 2:
            obs = obs[:, :, None]
            assert obs.shape == (size, size, C)
        return obs
    # load the converted cars3d data ?x128x128x3
    assert array.ndim == 4, f'invalid array shape, got: {array.shape}, must be: (N, H, W, C)'
    assert array.dtype == 'uint8', f'invalid array dtype, got: {array.dtype}, must be: "uint8"'
    N, H, W, C = array.shape
    data = ArrayDataset(array, transform=transform)
    # save the data
    with AtomicSaveFile(out_file, overwrite=overwrite) as temp_file:
        # resize the cars3d data
        idxs = tqdm(range(N), desc = 'converting') if progress else range(N)
        converted = np.zeros([N, size, size, C], dtype='uint8')
        for i in idxs:
            converted[i, ...] = data[i]
        # save the data
        np.savez_compressed(temp_file, **{save_key: converted})


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
