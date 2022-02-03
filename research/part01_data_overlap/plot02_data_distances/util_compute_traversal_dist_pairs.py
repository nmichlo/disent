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
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import ray
import torch
from ray.util.queue import Queue
from tqdm import tqdm

import research.code.util as H
from disent.dataset.data import GroundTruthData
from disent.util.inout.files import AtomicSaveFile
from disent.util.profiling import Timer
from disent.util.seeds import TempNumpySeed


log = logging.getLogger(__name__)


# ========================================================================= #
# Dataset Distances                                                         #
# ========================================================================= #


@ray.remote
def _compute_given_dists(gt_data, idxs, obs_pair_idxs, progress_queue=None):
    # checks
    assert idxs.ndim == 1
    assert obs_pair_idxs.ndim == 2
    assert len(obs_pair_idxs) == len(idxs)
    # storage
    with torch.no_grad(), Timer() as timer:
        obs_pair_dists = torch.zeros(*obs_pair_idxs.shape, dtype=torch.float32)
        # progress
        done = 0
        # for each observation
        for i, (obs_idx, pair_idxs) in enumerate(zip(idxs, obs_pair_idxs)):
            # load data
            obs = gt_data[obs_idx].flatten()
            batch = torch.stack([gt_data[i].flatten() for i in pair_idxs], dim=0)
            # compute distances
            obs_pair_dists[i, :] = torch.mean((batch - obs[None, :])**2, dim=-1, dtype=torch.float32)
            # add progress
            done += 1
            if progress_queue is not None:
                if timer.elapsed > 0.2:
                    timer.restart()
                    progress_queue.put(done)
                    done = 0
        # final update
        if progress_queue is not None:
            if done > 0:
                progress_queue.put(done)
        # done!
        return obs_pair_dists.numpy()


def compute_dists(gt_data: GroundTruthData, obs_pair_idxs: np.ndarray, jobs_per_cpu: int = 1):
    """
    Compute all the distances for ground truth data.
    - obs_pair_idxs is a 2D array (len(gt_dat), N) that is a list
      of paired indices to each element in the dataset.
    """
    # checks
    assert obs_pair_idxs.ndim == 2
    assert obs_pair_idxs.shape[0] == len(gt_data)
    assert jobs_per_cpu > 0
    # get workers
    num_cpus = int(ray.available_resources().get('CPU', 1))
    num_workers = int(num_cpus * jobs_per_cpu)
    # get chunks
    pair_idxs_chunks = np.array_split(obs_pair_idxs, num_workers)
    start_idxs = [0] + np.cumsum([len(c) for c in pair_idxs_chunks]).tolist()
    # progress queue
    progress_queue = Queue()
    ref_gt_data = ray.put(gt_data)
    # make workers
    futures = [
        _compute_given_dists.remote(ref_gt_data, np.arange(i, i+len(chunk)), chunk, progress_queue)
        for i, chunk in zip(start_idxs, pair_idxs_chunks)
    ]
    # check progress
    with tqdm(desc=gt_data.name, total=len(gt_data)) as progress:
        completed = 0
        while completed < len(gt_data):
            done = progress_queue.get()
            completed += done
            progress.update(done)
    # done
    obs_pair_dists = np.concatenate(ray.get(futures), axis=0)
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


def dataset_pair_idxs__nearby_scaled(gt_data: GroundTruthData, num_pairs: int = 10, min_radius: int = 2, radius_ratio: float = 0.2) -> np.ndarray:
    return dataset_pair_idxs__nearby(
        gt_data=gt_data,
        num_pairs=num_pairs,
        radius=np.maximum((np.array(gt_data.factor_sizes) * radius_ratio).astype('int'), min_radius),
    )


_PAIR_IDXS_FNS = {
    'random': dataset_pair_idxs__random,
    'nearby': dataset_pair_idxs__nearby,
    'nearby_scaled': dataset_pair_idxs__nearby_scaled,
}


def dataset_pair_idxs(mode: str, gt_data: GroundTruthData, num_pairs: int = 10, **kwargs):
    if mode not in _PAIR_IDXS_FNS:
        raise KeyError(f'invalid mode: {repr(mode)}, must be one of: {sorted(_PAIR_IDXS_FNS.keys())}')
    return _PAIR_IDXS_FNS[mode](gt_data, num_pairs=num_pairs, **kwargs)


# ========================================================================= #
# Cache Distances                                                           #
# ========================================================================= #

def _get_default_seed(
    pairs_per_obs: int,
    pair_mode: str,
    dataset_name: str,
):
    import hashlib
    seed_key = (pairs_per_obs, pair_mode, dataset_name)
    seed_hash = hashlib.md5(str(seed_key).encode())
    seed = int(seed_hash.hexdigest()[:8], base=16) % (2**32)  # [0, 2**32-1]
    return seed


def cached_compute_dataset_pair_dists(
    dataset_name: str = 'smallnorb',
    pair_mode: str = 'nearby_scaled',  # random, nearby, nearby_scaled
    pairs_per_obs: int = 64,
    seed: Optional[int] = None,
    # cache settings
    cache_dir: str = 'data/cache',
    force: bool = False,
    # normalize
    scaled: bool = True,
):
    # checks
    assert (seed is None) or isinstance(seed, int), f'seed must be an int or None, got: {type(seed)}'
    assert isinstance(pairs_per_obs, int), f'pairs_per_obs must be an int, got: {type(pairs_per_obs)}'
    assert pair_mode in _PAIR_IDXS_FNS, f'pair_mode is invalid, got: {repr(pair_mode)}, must be one of: {sorted(_PAIR_IDXS_FNS.keys())}'
    # get default seed
    if seed is None:
        seed = _get_default_seed(pairs_per_obs=pairs_per_obs, pair_mode=pair_mode, dataset_name=dataset_name)
    # cache path
    cache_path = Path(cache_dir, f'dist-pairs_{dataset_name}_{pairs_per_obs}_{pair_mode}_{seed}.npz')
    # generate if it does not exist
    if force or not cache_path.exists():
        log.info(f'generating cached distances for: {dataset_name} to: {cache_path}')
        # load data
        gt_data = H.make_data(dataset_name, transform_mode='float32')
        # generate idxs
        with TempNumpySeed(seed=seed):
            obs_pair_idxs = dataset_pair_idxs(pair_mode, gt_data, num_pairs=pairs_per_obs)
        obs_pair_dists = compute_dists(gt_data, obs_pair_idxs)
        # generate & save
        with AtomicSaveFile(file=cache_path, overwrite=force) as path:
            np.savez(path, **{
                'dataset_name': dataset_name,
                'seed': seed,
                'obs_pair_idxs': obs_pair_idxs,
                'obs_pair_dists': obs_pair_dists,
            })
    # load cached data
    else:
        log.info(f'loading cached distances for: {dataset_name} from: {cache_path}')
        data = np.load(cache_path)
        obs_pair_idxs = data['obs_pair_idxs']
        obs_pair_dists = data['obs_pair_dists']
    # normalize the max distance to 1.0
    if scaled:
        obs_pair_dists /= np.max(obs_pair_dists)
    # done!
    return obs_pair_idxs, obs_pair_dists


# ========================================================================= #
# TEST!                                                                     #
# ========================================================================= #


def generate_common_cache(force=False, force_seed=None):
    import itertools
    # settings
    sweep_pairs_per_obs = [128, 32, 256, 64, 16]
    sweep_pair_modes = ['nearby_scaled', 'random', 'nearby']
    sweep_dataset_names = ['cars3d', 'smallnorb', 'shapes3d', 'dsprites', 'xysquares']
    # info
    log.info(f'Computing distances for sweep of size: {len(sweep_pairs_per_obs)*len(sweep_pair_modes)*len(sweep_dataset_names)}')
    # sweep
    for i, (pairs_per_obs, pair_mode, dataset_name) in enumerate(itertools.product(sweep_pairs_per_obs, sweep_pair_modes, sweep_dataset_names)):
        # deterministic seed based on settings
        if force_seed is None:
            seed = _get_default_seed(pairs_per_obs=pairs_per_obs, pair_mode=pair_mode, dataset_name=dataset_name)
        else:
            seed = force_seed
        # info
        log.info(f'[{i}] Computing distances for: {repr(dataset_name)} {repr(pair_mode)} {repr(pairs_per_obs)} {repr(seed)}')
        # get the dataset and delete the transform
        cached_compute_dataset_pair_dists(
            dataset_name=dataset_name,
            pair_mode=pair_mode,
            pairs_per_obs=pairs_per_obs,
            seed=seed,
            force=force,
            scaled=True
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ray.init(num_cpus=psutil.cpu_count(logical=False))
    generate_common_cache()


# ========================================================================= #
# DONE                                                                      #
# ========================================================================= #
