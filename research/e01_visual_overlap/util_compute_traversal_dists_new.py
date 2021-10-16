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
from typing import Optional

import ray

import logging
import os
from typing import Tuple

import numpy as np
import torch
from ray.util.queue import Queue
from tqdm import tqdm

import research.util as H
from disent.dataset.data import GroundTruthData
from disent.util.profiling import Timer


log = logging.getLogger(__name__)


# ========================================================================= #
# Dataset Distances                                                         #
# ========================================================================= #


def get_as_completed(obj_ids):
    # https://github.com/ray-project/ray/issues/5554
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


def _compute_dists(gt_data: GroundTruthData, obs_idx: int, pair_idxs: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Compute the distances between a single observation
    and all the other observations in a list.
    """
    obs = gt_data[obs_idx].flatten()
    batch = torch.stack([gt_data[i].flatten() for i in pair_idxs], dim=0)
    # compute distances
    dists = torch.mean((batch - obs[None, :]) ** 2, dim=-1, dtype=torch.float32).numpy()
    # done!
    return dists


def __compute_batch_dists(gt_data: GroundTruthData, start_idx: int, obs_pair_idxs: np.ndarray):
    """
    Compute the distances between observations (B,) and their corresponding array of pairs (B, N).
    - The observations are obtained from a starting point in the gt_data and the batch size
    - The obs_pair_idxs is a 2D array, the first dim is the batch size, the second dim is the number of pairs per observation.
    """
    assert obs_pair_idxs.ndim == 2
    # compute dists
    obs_pair_dists = np.stack([
        _compute_dists(gt_data, start_idx + i, pair_idxs)
        for i, pair_idxs in enumerate(obs_pair_idxs)
    ])
    # checks
    assert obs_pair_dists.shape == obs_pair_idxs.shape
    return start_idx, obs_pair_dists


@ray.remote
def _compute_batch_dists(gt_data: GroundTruthData, start_idx: int, obs_pair_idxs: np.ndarray):
    return __compute_batch_dists(gt_data=gt_data, start_idx=start_idx, obs_pair_idxs=obs_pair_idxs)


def compute_dists(gt_data: GroundTruthData, obs_pair_idxs: np.ndarray, batch_size: int = 256):
    """
    Compute all the distances for ground truth data.
    - obs_pair_idxs is a 2D array (len(gt_dat), N) that is a list
      of paired indices to each element in the dataset.
    """
    # checks
    assert obs_pair_idxs.ndim == 2
    assert obs_pair_idxs.shape[0] == len(gt_data)
    # store distances
    obs_pair_dists = np.zeros(obs_pair_idxs.shape, dtype='float32')
    ref_gt_data = ray.put(gt_data)
    # compute distances
    # TODO: this should assign a portion of the dataset to each worker, rather than split it like this.
    #       this is very inefficient for large datasets and small batch sizes.
    futures = [
        _compute_batch_dists.remote(ref_gt_data, start_idx, obs_pair_idxs[start_idx:start_idx+batch_size])
        for start_idx in range(0, len(gt_data), batch_size)
    ]
    # wait for dists
    for start_idx, pair_dists in tqdm(get_as_completed(futures), total=len(futures)):
        obs_pair_dists[start_idx:start_idx+len(pair_dists), :] = pair_dists
    # done!
    return obs_pair_dists


# ========================================================================= #
# Distance Types                                                            #
# ========================================================================= #


def dataset_pair_idxs__random(gt_data: GroundTruthData, num_pairs: int = 25) -> np.ndarray:
    # purely random pairs...
    return np.random.randint(0, len(gt_data), size=[len(gt_data), num_pairs])


def dataset_pair_idxs__nearby(gt_data: GroundTruthData, num_pairs: int = 10, radius: int = 5) -> np.ndarray:
    radius = np.array(radius)
    assert radius.ndim in (0, 1)
    if radius.ndim == 1:
        assert radius.shape == (gt_data.num_factors,)
    # get all positions
    pos = gt_data.idx_to_pos(np.arange(len(gt_data)))
    # generate random offsets
    offsets = np.random.randint(-radius, radius + 1, size=[len(gt_data), num_pairs, gt_data.num_factors])
    # broadcast random offsets & wrap around
    nearby_pos = (pos[:, None, :] + offsets) % gt_data.factor_sizes
    # convert back to indices
    nearby_idxs = gt_data.pos_to_idx(nearby_pos)
    # done!
    return nearby_idxs


def dataset_pair_idxs__scaled_nearby(gt_data: GroundTruthData, num_pairs: int = 10, min_radius: int = 2, radius_ratio: float = 0.2) -> np.ndarray:
    return dataset_pair_idxs__nearby(
        gt_data=gt_data,
        num_pairs=num_pairs,
        radius=np.maximum((np.array(gt_data.factor_sizes) * radius_ratio).astype('int'), min_radius),
    )


_PAIR_IDXS_FNS = {
    'random': dataset_pair_idxs__random,
    'nearby': dataset_pair_idxs__nearby,
    'scaled_nearby': dataset_pair_idxs__scaled_nearby,
}


def dataset_pair_idxs(mode: str, gt_data: GroundTruthData, num_pairs: int = 10, seed: int = 7777, **kwargs):
    if mode not in _PAIR_IDXS_FNS:
        raise KeyError('invalid mode: {repr()}')


# ========================================================================= #
# Distance Types                                                            #
# ========================================================================= #


if __name__ == '__main__':


    ray.init(num_cpus=min(os.cpu_count(), 64))
    # generate_common_cache()

    def main():
        dataset_name = 'cars3d'

        # load data
        gt_data = H.make_data(dataset_name, transform_mode='float32')

        obs_pair_idxs = dataset_pair_idxs__scaled_nearby(gt_data)

        results = compute_dists(gt_data, obs_pair_idxs=obs_pair_idxs, batch_size=256)

    main()
