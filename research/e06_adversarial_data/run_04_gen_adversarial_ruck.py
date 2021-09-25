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
import random
from datetime import datetime
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import ray
import ruck
from ruck import R
from ruck.util import ray_map
from ruck.util import ray_remote_put
from ruck.util import ray_remote_puts

import research.util as H
from disent.util.function import wrapped_partial
from disent.util.inout.paths import ensure_parent_dir_exists
from disent.util.profiling import Timer
from disent.util.seeds import seed
from research.e01_visual_overlap.util_compute_traversal_dists import cached_compute_all_factor_dist_matrices
from research.e06_adversarial_data.run_04_gen_adversarial_genetic import individual_ave
from research.e06_adversarial_data.util_eval_adversarial import eval_individual, eval_factor_fitness_numba


log = logging.getLogger(__name__)


# ========================================================================= #
# Evaluation                                                                #
# ========================================================================= #

@ray.remote
def evaluate_member(
    value: np.ndarray,
    gt_dist_matrices: np.ndarray,
    factor_sizes: Tuple[int, ...],
    factor_score_offset: float,
    factor_score_weight: float,
    kept_ratio_offset: float,
    kept_ratio_weight: float,
    factor_score_mode: str,
    factor_score_agg: str,
) -> float:
    factor_score, kept_ratio = eval_individual(
        eval_factor_fitness_fn=eval_factor_fitness_numba,
        individual=value,
        gt_dist_matrices=gt_dist_matrices,
        factor_sizes=factor_sizes,
        fitness_mode=factor_score_mode,
        obj_mode_aggregate=factor_score_agg,
        exclude_diag=True,
    )
    # weight scores
    w_factor_score = factor_score_offset + factor_score_weight * factor_score
    w_kept_ratio = kept_ratio_offset + kept_ratio_weight * kept_ratio
    # get minimum, which we want to maximize
    return min(w_factor_score, w_kept_ratio)


# ========================================================================= #
# Type Hints                                                                #
# ========================================================================= #


Values = List[ray.ObjectRef]
Population = List[ruck.Member[ray.ObjectRef]]


def mutate_oneof(*mutate_fns):
    def mutate_fn(value):
        fn = random.choice(mutate_fns)
        return fn(value)
    return mutate_fn


# ========================================================================= #
# Evolutionary System                                                       #
# ========================================================================= #


class DatasetMaskModule(ruck.EaModule):

    # STATISTICS

    def get_stats_groups(self):
        return {
            'fit': ruck.StatsGroup(lambda pop: [m.fitness for m in pop], min=np.min, max=np.max, mean=np.mean),
            'mask': ruck.StatsGroup(lambda pop: [np.mean(ray.get(m.value)) for m in pop], min=np.min, max=np.max, mean=np.mean),
        }

    def get_progress_stats(self):
        return ('evals', 'fit:max', 'fit:mean', 'mask:mean')

    # POPULATION

    def gen_starting_values(self) -> Values:
        return [
            ray.put(np.random.random(np.prod(self.hparams.factor_sizes)) < (0.1 + np.random.random() * 0.8))
            for _ in range(self.hparams.population_size)
        ]

    def select_population(self, population: Population, offspring: Population) -> Population:
        return R.select_tournament(population + offspring, num=len(population), k=3)

    def evaluate_values(self, values: Values) -> List[float]:
        return ray.get([self._evaluate_value_fn(v) for v in values])

    # INITIALISE

    def __init__(
        self,
        dataset_name: str = 'cars3d',
        population_size: int = 128,
        # optim settings
        factor_score_weight: float = -1000.0,
        factor_score_offset: float = 1.0,
        kept_ratio_weight: float = 1.0,
        kept_ratio_offset: float = 0.0,
        # other settings
        factor_score_mode: str = 'std',
        factor_score_agg: str = 'mean',
        # ea settings
        p_mate: float = 0.5,
        p_mutate: float = 0.5,
        p_mutate_flip: float = 0.05,
    ):
        # load the dataset
        gt_data = H.make_data(dataset_name)
        factor_sizes = gt_data.factor_sizes
        # save hyper parameters to .hparams
        self.save_hyperparameters(include=['factor_sizes'])
        # get offspring function
        self.generate_offspring = wrapped_partial(
            R.apply_mate_and_mutate,
            mate_fn=ray_remote_puts(R.mate_crossover_nd).remote,
            mutate_fn=ray_remote_put(mutate_oneof(
                wrapped_partial(R.mutate_flip_bits, p=p_mutate_flip),
                wrapped_partial(R.mutate_flip_bit_groups, p=p_mutate_flip),
            )).remote,
            p_mate=p_mate,
            p_mutate=p_mutate,
            map_fn=ray_map  # parallelize
        )
        # get evaluation function
        self._evaluate_value_fn = wrapped_partial(
            evaluate_member.remote,
            gt_dist_matrices=ray.put(cached_compute_all_factor_dist_matrices(dataset_name)),
            factor_sizes=factor_sizes,
            factor_score_offset=factor_score_offset,
            factor_score_weight=factor_score_weight,
            kept_ratio_offset=kept_ratio_offset,
            kept_ratio_weight=kept_ratio_weight,
            factor_score_mode=factor_score_mode,
            factor_score_agg=factor_score_agg,
        )


# ========================================================================= #
# RUNNER                                                                    #
# ========================================================================= #


def run(
    dataset_name: str = 'shapes3d',  # xysquares_8x8_toy_s4, xcolumns_8x_toy_s1
    generations: int = 250,
    population_size: int = 128,
    factor_score_mode: str = 'std',
    factor_score_agg: str = 'mean',
    # optim settings
    factor_score_weight: float = -1000.0,
    kept_ratio_weight: float = 1.0,
    # save settings
    save: bool = False,
    save_prefix: str = '',
    seed_: Optional[int] = None,
    # plot settings
    plot: bool = False,
):
    # save the starting time for the save path
    time_string = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    log.info(f'Starting run at time: {time_string}')

    # deterministic? This might not actually work with ray
    seed_ = seed_ if (seed_ is not None) else int(np.random.randint(1, 2**31-1))
    seed(seed_)

    # run!
    with Timer('ruck:onemax'):
        problem = DatasetMaskModule(
            dataset_name=dataset_name,
            population_size=population_size,
            factor_score_mode=factor_score_mode,
            factor_score_agg=factor_score_agg,
            factor_score_weight=factor_score_weight,
            kept_ratio_weight=kept_ratio_weight,
        )
        # train
        population, logbook, halloffame = ruck.Trainer(generations=generations, progress=True).fit(problem)
        # retrieve stats
        log.info(f'start population: {logbook[0]}')
        log.info(f'end population: {logbook[-1]}')
        values = [ray.get(m.value) for m in halloffame]

    # plot average images
    if plot:
        H.plt_subplots_imshow(
            [[individual_ave(dataset_name, v) for v in values]],
            col_labels=[f'{np.sum(v)} / {np.prod(v.shape)} |' for v in values],
            show=True, vmin=0.0, vmax=1.0,
            title=f'{dataset_name}: g{generations} p{population_size} [{factor_score_mode}, {factor_score_agg}]',
        )

    if save:
        # get totals
        use_elems = np.sum(values[0])
        num_elems = np.prod(values[0].shape)
        use_ratio = (use_elems / num_elems)
        # get save path, make parent dir & save!
        job_name = f'{(save_prefix + "_" if save_prefix else "")}{dataset_name}_{use_ratio:.2f}x{num_elems}_{generations}x{population_size}_{factor_score_mode}_{factor_score_agg}_{factor_score_weight}_{kept_ratio_weight}'
        save_path = ensure_parent_dir_exists(ROOT_DIR, 'out/adversarial_mask', f'{time_string}_{job_name}_mask.npz')
        log.info(f'saving mask data to: {save_path}')
        np.savez(save_path, mask=values[0], params=problem.hparams, seed=seed_)
        # return the path
        return save_path
    else:
        # return the value
        return values[0]

# ========================================================================= #
# ENTRYPOINT                                                                #
# ========================================================================= #


ROOT_DIR = os.path.abspath(__file__ + '/../../..')


def main():
    from itertools import product

    # values for ~50%
    # cars3d -15
    # smallnorb -35

    # for (factor_score_agg, factor_score_mode, (dataset_name, factor_score_weight)) in product(
    #     ['mean'],
    #     ['range'],
    #     [('cars3d', -15), ('smallnorb', -34)]
    # ):

    for (factor_score_agg, factor_score_mode, dataset_name, factor_score_weight) in product(
        ['mean'], #, 'max', 'gmean'],
        ['range'], #, 'std'],
        ['xysquares_8x8_toy_s2'], # , 'cars3d', 'smallnorb', 'shapes3d', 'dsprites'],
        [-1], # , -5, -15, -34],
    ):
        # print('='*100)
        # print(f'[STARTING]: dataset_name={repr(dataset_name)} factor_score_mode={repr(factor_score_mode)} factor_score_agg={repr(factor_score_agg)} factor_score_weight={repr(factor_score_weight)}')
        # try:
        run(
            dataset_name=dataset_name,
            factor_score_mode=factor_score_mode,
            factor_score_agg=factor_score_agg,
            factor_score_weight=factor_score_weight,
            generations=100,
            seed_=42,
            save=False,
            save_prefix='DELETE',
            plot=True,
        )
        # except KeyboardInterrupt:
        #     warnings.warn('Exiting early')
        # except:
        #     warnings.warn(f'[FAILED]: dataset_name={repr(dataset_name)} factor_score_mode={repr(factor_score_mode)} factor_score_agg={repr(factor_score_agg)} factor_score_weight={repr(factor_score_weight)}')
        # print('='*100)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ray.init(num_cpus=64)
    main()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
