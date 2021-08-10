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

import multiprocessing.synchronize
from concurrent.futures import Executor
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
from typing import Sequence

import h5py
import logging

import numpy as np
import os
import psutil
import torch
from tqdm import tqdm

import experiment.exp.util as H
from disent.util.inout.paths import ensure_parent_dir_exists
from disent.util.seeds import seed
from disent.util.seeds import TempNumpySeed
from disent.util.profiling import Timer


log = logging.getLogger(__name__)


# ========================================================================= #
# losses                                                                    #
# ========================================================================= #


def stochastic_const_loss(pred: torch.Tensor, mask: torch.Tensor, num_pairs: int, num_samples: int, loss='mse', reg_out_of_bounds=True, top_k: int = None, constant_targ: float = None) -> torch.Tensor:
    ia, ib = torch.randint(0, len(pred), size=(2, num_samples), device=pred.device)
    # constant dist loss
    x_ds = (H.unreduced_loss(pred[ia], pred[ib], mode=loss) * mask[None, ...]).mean(dim=(-3, -2, -1))
    # compute constant loss
    if constant_targ is None:
        iA, iB = torch.randint(0, len(x_ds), size=(2, num_pairs), device=pred.device)
        lcst = H.unreduced_loss(x_ds[iA], x_ds[iB], mode=loss)
    else:
        lcst = H.unreduced_loss(x_ds, torch.full_like(x_ds, constant_targ), mode=loss)
    # aggregate constant loss
    if top_k is None:
        lcst = lcst.mean()
    else:
        lcst = torch.topk(lcst, k=top_k, largest=True).values.mean()
    # values over the required range
    if reg_out_of_bounds:
        m = torch.nan_to_num((0 - pred[pred < 0]) ** 2, nan=0).mean()
        M = torch.nan_to_num((pred[pred > 1] - 1) ** 2, nan=0).mean()
        mM = m + M
    else:
        mM = 0.
    # done!
    return mM + lcst


# ========================================================================= #
# h5py dataset helper                                                       #
# ========================================================================= #


NAME_DATA = 'data'
NAME_VISITS = 'visits'
NAME_OBS = 'x_targ'

_SAVE_TYPE_LOOKUP = {
    'uint8': torch.uint8,
    'float16': torch.float16,
    'float32': torch.float32,
}

SAVE_TYPE = 'float16'
assert SAVE_TYPE in _SAVE_TYPE_LOOKUP


def _make_hdf5_dataset(path, dataset, overwrite_mode: str = 'continue') -> str:
    path = ensure_parent_dir_exists(path)
    # get read/write mode
    if overwrite_mode == 'overwrite':
        rw_mode = 'w'  # create new file, overwrite if exists
    elif overwrite_mode == 'fail':
        rw_mode = 'x'  # create new file, fail if exists
    elif overwrite_mode == 'continue':
        rw_mode = 'a'  # create if missing, append if exists
        # clear file consistency flags
        # if clear_consistency_flags:
        #     if os.path.isfile(path):
        #         cmd = ["h5clear", "-s", "'{path}'"]
        #         print(f'clearing file consistency flags: {" ".join(cmd)}')
        #         try:
        #             subprocess.check_output(cmd)
        #         except FileNotFoundError:
        #             raise FileNotFoundError('h5clear utility is not installed!')
    else:
        raise KeyError(f'invalid overwrite_mode={repr(overwrite_mode)}')
    # open in read write mode
    log.info(f'Opening hdf5 dataset: overwrite_mode={repr(overwrite_mode)} exists={repr(os.path.exists(path))} path={repr(path)}')
    with h5py.File(path, rw_mode, libver='earliest') as f:
        # get data
        num_obs = len(dataset)
        obs_shape = dataset[0][NAME_OBS][0].shape
        # make dset
        if NAME_DATA not in f:
            f.create_dataset(
                NAME_DATA,
                shape=(num_obs, *obs_shape),
                dtype=SAVE_TYPE,
                chunks=(1, *obs_shape),
                track_times=False,
            )
        # make set_dset
        if NAME_VISITS not in f:
            f.create_dataset(
                NAME_VISITS,
                shape=(num_obs,),
                dtype='int64',
                chunks=(1,),
                track_times=False,
            )
    return path


# def _read_hdf5_batch(h5py_path: str, idxs, return_visits=False):
#     batch, visits = [], []
#     with h5py.File(h5py_path, 'r', swmr=True) as f:
#         for i in idxs:
#             visits.append(f[NAME_VISITS][i])
#             obs = torch.as_tensor(f[NAME_DATA][i], dtype=torch.float32)
#             if SAVE_TYPE == 'uint8':
#                 obs /= 255
#             batch.append(obs)
#     # return values
#     if return_visits:
#         return torch.stack(batch, dim=0), np.array(visits, dtype=np.int64)
#     else:
#         return torch.stack(batch, dim=0)


def _load_hdf5_batch(dataset, h5py_path: str, idxs, initial_noise: Optional[float] = None, return_visits=True):
    """
    Load a batch from the disk -- always return float32
    - Can be used by multiple threads at a time.
    - returns an item from the original dataset if an
      observation has not been saved into the hdf5 dataset yet.
    """
    batch, visits = [], []
    with h5py.File(h5py_path, 'r', swmr=True) as f:
        for i in idxs:
            v = f[NAME_VISITS][i]
            if v > 0:
                obs = torch.as_tensor(f[NAME_DATA][i], dtype=torch.float32)
                if SAVE_TYPE == 'uint8':
                    obs /= 255
            else:
                (obs,) = dataset[i][NAME_OBS]
                obs = obs.to(torch.float32)
                if initial_noise is not None:
                    obs += (torch.randn_like(obs) * initial_noise)
            batch.append(obs)
            visits.append(v)
    # stack and check values
    batch = torch.stack(batch, dim=0)
    assert batch.dtype == torch.float32
    # return values
    if return_visits:
        return batch, np.array(visits, dtype=np.int64)
    else:
        return batch


def _save_hdf5_batch(h5py_path: str, batch, idxs):
    """
    Save a float32 batch to disk.
    - Can only be used by one thread at a time!
    """
    assert batch.dtype == torch.float32
    with h5py.File(h5py_path, 'r+', libver='earliest') as f:
        for obs, idx in zip(batch, idxs):
            if SAVE_TYPE == 'uint8':
                f[NAME_DATA][idx] = torch.clamp(torch.round(obs * 255), 0, 255).to(torch.uint8)
            else:
                f[NAME_DATA][idx] = obs.to(_SAVE_TYPE_LOOKUP[SAVE_TYPE])
            f[NAME_VISITS][idx] += 1


# ========================================================================= #
# multiproc h5py dataset helper                                             #
# ========================================================================= #


class FutureList(object):
    def __init__(self, futures: Sequence[Future]):
        self._futures = futures

    def result(self):
        return [future.result() for future in self._futures]


# ========================================================================= #
# multiproc h5py dataset helper                                             #
# ========================================================================= #


# SUBMIT:


def _submit_load_batch_futures(executor: Executor, num_splits: int, dataset, h5py_path: str, idxs, initial_noise: Optional[float] = None) -> FutureList:
    return FutureList([
        executor.submit(__inner__load_batch, dataset=dataset, h5py_path=h5py_path, idxs=idxs, initial_noise=initial_noise)
        for idxs in np.array_split(idxs, num_splits)
    ])


def _submit_save_batch(executor: Executor, h5py_path: str, batch, idxs) -> Future:
    return executor.submit(__inner__save_batch, h5py_path=h5py_path, batch=batch, idxs=idxs)


NUM_WORKERS = psutil.cpu_count()
_BARRIER = None


def __inner__load_batch(dataset, h5py_path: str, idxs, initial_noise: Optional[float] = None):
    _BARRIER.wait()
    result = _load_hdf5_batch(dataset=dataset, h5py_path=h5py_path, idxs=idxs, initial_noise=initial_noise)
    _BARRIER.wait()
    return result


def __inner__save_batch(h5py_path, batch, idxs):
    _save_hdf5_batch(h5py_path=h5py_path, batch=batch, idxs=idxs)


# WAIT:


def _wait_for_load_future(future: FutureList):
    with Timer() as t:
        xs, visits = zip(*future.result())
    xs = torch.cat(xs, dim=0)
    visits = np.concatenate(visits, axis=0).mean(dtype=np.float32)
    return (xs, visits), t


def _wait_for_save_future(future: Future):
    with Timer() as t:
        future.result()
    return t


# ========================================================================= #
# adversarial dataset generator                                             #
# ========================================================================= #


def run_generate_and_save_adversarial_dataset_mp(
    dataset_name: str = 'shapes3d',
    dataset_load_into_memory: bool = False,
    optimizer: str = 'adam',
    lr: float = 1e-2,
    obs_masked: bool = True,
    obs_initial_noise: Optional[float] = None,
    loss_fn: str = 'mse',
    batch_size: int = 1024*12,                # approx
    batch_sample_mode: str = 'shuffle',       # range, shuffle, random
    loss_num_pairs: int = 1024*4,
    loss_num_samples: int = 1024*4*2,         # only applies if loss_const_targ=None
    loss_top_k: Optional[int] = None,
    loss_const_targ: Optional[float] = 0.1,   # replace stochastic pairwise constant loss with deterministic loss target
    loss_reg_out_of_bounds: bool = False,
    train_epochs: int = 8,
    train_optim_steps: int = 125,
    # skipped params
    save_folder: str = 'out/overlap',
    save_prefix: str = '',
    overwrite_mode: str = 'fail',             # continue, overwrite, fail
    seed_: Optional[int] = 777,
) -> str:
    # checks
    if obs_initial_noise is not None:
        assert not obs_masked, '`obs_masked` cannot be `True`, if using initial noise, ie. `obs_initial_noise is not None`'

    # deterministic!
    seed(seed_)

    # ↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓ #
    # make dataset
    dataset = H.make_dataset(dataset_name, load_into_memory=dataset_load_into_memory, load_memory_dtype=torch.float16)
    # get save path
    assert not ('/' in save_prefix or '\\' in save_prefix)
    name = H.params_as_string(H.get_caller_params(exclude=["save_folder", "save_prefix", "overwrite_mode", "seed_"]))
    path = _make_hdf5_dataset(os.path.join(save_folder, f'{save_prefix}{name}.hdf5'), dataset=dataset, overwrite_mode=overwrite_mode)
    # ↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑ #

    train_batches = (len(dataset) + batch_size - 1) // batch_size
    # loop vars & progress bar
    save_time = Timer()
    prog = tqdm(total=train_epochs * train_batches * train_optim_steps, postfix={'loss': 0.0, '💯': 0.0, '🔍': 'N/A', '💾': 'N/A'}, ncols=100)
    # multiprocessing pool
    global _BARRIER  # TODO: this is a hack and should be unique to each run
    _BARRIER = multiprocessing.Barrier(NUM_WORKERS)
    executor = ProcessPoolExecutor(NUM_WORKERS)

    # EPOCHS:
    for e in range(train_epochs):
        # generate batches
        batch_idxs = H.generate_epoch_batch_idxs(num_obs=len(dataset), num_batches=train_batches, mode=batch_sample_mode)
        # first data load
        load_future = _submit_load_batch_futures(executor, num_splits=NUM_WORKERS, dataset=dataset, h5py_path=path, idxs=batch_idxs[0], initial_noise=obs_initial_noise)

        # TODO: log to WANDB
        # TODO: SAMPLING STRATEGY MIGHT NEED TO CHANGE!
        #       - currently random pairs are generated, but the pairs that matter are the nearby ones.
        #       - sample pairs that increase and decrease along an axis
        #       - sample pairs that are nearby according to the factor distance metric

        # BATCHES:
        for n in range(len(batch_idxs)):
            # ↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓ #
            # get batch -- transfer to gpu is the bottleneck
            (x, visits), load_time = _wait_for_load_future(load_future)
            x = x.cuda().requires_grad_(True)
            # ↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑ #

            # queue loading an extra batch
            if (n+1) < len(batch_idxs):
                load_future = _submit_load_batch_futures(executor, num_splits=NUM_WORKERS, dataset=dataset, h5py_path=path, idxs=batch_idxs[n + 1], initial_noise=obs_initial_noise)

            # ↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓ #
            # make optimizers
            mask = H.make_changed_mask(x, masked=obs_masked)
            optim = H.make_optimizer(x, name=optimizer, lr=lr)
            # ↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑ #

            # OPTIMIZE:
            for _ in range(train_optim_steps):
                # final loss & update
                # ↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓=↓ #
                loss = stochastic_const_loss(x, mask, num_pairs=loss_num_pairs, num_samples=loss_num_samples, loss=loss_fn, reg_out_of_bounds=loss_reg_out_of_bounds, top_k=loss_top_k, constant_targ=loss_const_targ)
                H.step_optimizer(optim, loss)
                # ↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑=↑ #

                # update progress bar
                logs = {'loss': float(loss), '💯': visits, '🔍': load_time.pretty, '💾': save_time.pretty}
                prog.update()
                prog.set_postfix(logs)

            # save optimized minibatch
            if n > 0:
                save_time = _wait_for_save_future(save_future)
            save_future = _submit_save_batch(executor, h5py_path=path, batch=x.detach().cpu(), idxs=batch_idxs[n])

        # final save
        save_time = _wait_for_save_future(save_future)

    # cleanup all
    executor.shutdown()
    # return the path to the dataset
    return path


# ========================================================================= #
# test adversarial dataset generator                                      #
# ========================================================================= #

def run_generate_adversarial_data(
    dataset: str ='shapes3d',
    factor: str ='wall_hue',
    factor_mode: str = 'sample_random',
    optimizer: str ='radam',
    lr: float = 1e-2,
    obs_num: int = 1024 * 10,
    obs_noise_weight: float = 0,
    obs_masked: bool = True,
    loss_fn: str = 'mse',
    loss_num_pairs: int = 4096,
    loss_num_samples: int = 4096*2,  # only applies if loss_const_targ=None
    loss_top_k: int = None,
    loss_const_targ: float = None,  # replace stochastic pairwise constant loss with deterministic loss target
    loss_reg_out_of_bounds: bool = False,
    train_steps: int = 2000,
    display_period: int = 500,
):
    seed(777)
    # make dataset
    dataset = H.make_dataset(dataset)
    # make batches
    factors = H.sample_factors(dataset, num_obs=obs_num, factor_mode=factor_mode, factor=factor)
    x = dataset.dataset_batch_from_factors(factors, 'target')
    # make tensors to optimize
    if torch.cuda.is_available():
        x = x.cuda()
    x = torch.tensor(x + torch.randn_like(x) * obs_noise_weight, requires_grad=True)
    # generate mask
    mask = H.make_changed_mask(x, masked=obs_masked)
    H.plt_imshow(H.to_img(mask.to(torch.float32)), show=True)
    # make optimizer
    optimizer = H.make_optimizer(x, name=optimizer, lr=lr)

    # optimize differences according to loss
    prog = tqdm(range(train_steps+1), postfix={'loss': 0.0})
    for i in prog:
        # final loss
        loss = stochastic_const_loss(x, mask, num_pairs=loss_num_pairs, num_samples=loss_num_samples, loss=loss_fn, reg_out_of_bounds=loss_reg_out_of_bounds, top_k=loss_top_k, constant_targ=loss_const_targ)
        # update variables
        H.step_optimizer(optimizer, loss)
        if i % display_period == 0:
            log.warning(f'visualisation of `x[:9]` was disabled')
        prog.set_postfix({'loss': float(loss)})


# ========================================================================= #
# entrypoint                                                                #
# ========================================================================= #

# TODO: add WANDB support for visualisation of dataset
# TODO: add graphing of visual overlap like exp 01

def main():
    logging.basicConfig(level=logging.INFO, format='(%(asctime)s) %(name)s:%(lineno)d [%(levelname)s]: %(message)s')

    paths = []
    for i, kwargs in enumerate([
        # dict(save_prefix='e128__fixed_unmask_const_', obs_masked=False, loss_const_targ=0.1,  obs_initial_noise=None, optimizer='adam', dataset_name='cars3d'),
        # dict(save_prefix='e128__fixed_unmask_const_', obs_masked=False, loss_const_targ=0.1,  obs_initial_noise=None, optimizer='adam', dataset_name='smallnorb'),
        # dict(save_prefix='e128__fixed_unmask_randm_', obs_masked=False, loss_const_targ=None, obs_initial_noise=None, optimizer='adam', dataset_name='cars3d'),
        # dict(save_prefix='e128__fixed_unmask_randm_', obs_masked=False, loss_const_targ=None, obs_initial_noise=None, optimizer='adam', dataset_name='smallnorb'),
    ]):
        # generate dataset
        try:
            path = run_generate_and_save_adversarial_dataset_mp(
                train_epochs=128,
                train_optim_steps=175,
                seed_=777,
                overwrite_mode='overwrite',
                dataset_load_into_memory=True,
                lr=5e-3,
                # batch_sample_mode='range',
                **kwargs
            )
            paths.append(path)
        except Exception as e:
            log.error(f'[{i}] FAILED RUN: {e} -- {repr(kwargs)}', exc_info=True)
        # load some samples and display them
        try:
            log.warning(f'visualisation of `_read_hdf5_batch(paths[-1], display_idxs)` was disabled')
        except Exception as e:
            log.warning(f'[{i}] FAILED SHOW: {e} -- {repr(kwargs)}')

    for path in paths:
        print(path)


# ========================================================================= #
# main                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    main()
