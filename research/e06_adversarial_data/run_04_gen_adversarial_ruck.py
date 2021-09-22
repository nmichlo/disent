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
import warnings
from datetime import datetime
from typing import Optional
from typing import Tuple

import numpy as np
import ray

from disent.util.function import wrapped_partial
from disent.util.inout.paths import ensure_parent_dir_exists
from disent.util.profiling import Timer
from disent.util.seeds import seed
from research.e01_visual_overlap.util_compute_traversal_dists import cached_compute_all_factor_dist_matrices
from research.e06_adversarial_data.run_04_gen_adversarial_genetic import _toolbox_eval_individual
from research.e06_adversarial_data.run_04_gen_adversarial_genetic import individual_ave
from research.ruck import *
from research.ruck import EaModule
from research.ruck import PopulationHint

import research.util as H


log = logging.getLogger(__name__)


# ========================================================================= #
# EXPERIMENT                                                                #
# ========================================================================= #


def evaluate_member(
    value: np.ndarray,
    gt_dist_matrices_id,
    factor_sizes: Tuple[int, ...],
    factor_score_offset: float,
    factor_score_weight: float,
    kept_ratio_offset: float,
    kept_ratio_weight: float,
    factor_score_mode: str,
    factor_score_agg: str,
) -> float:
    factor_score, kept_ratio = _toolbox_eval_individual(
        individual=value,
        gt_dist_matrices=ray.get(gt_dist_matrices_id),
        factor_sizes=factor_sizes,
        fitness_mode=factor_score_mode,
        obj_mode_aggregate=factor_score_agg,
        exclude_diag=True,
    )
    # check values just in case something goes wrong!
    factor_score = np.nan_to_num(factor_score, nan=float('-inf'))
    kept_ratio = np.nan_to_num(kept_ratio, nan=float('-inf'))
    # weight scores
    w_factor_score = factor_score_offset + factor_score_weight * factor_score
    w_kept_ratio = kept_ratio_offset + kept_ratio_weight * kept_ratio
    # get minimum, which we want to maximize
    return min(w_factor_score, w_kept_ratio)


def generate_offspring(population: PopulationHint, p_mate: float = 0.5, p_mutate: float = 0.5) -> PopulationHint:
    # SEE: R.factory_ea_alg -- TODO: make it easier to swap!
    return R.apply_mate_and_mutate(
        population=R.select_tournament(population, len(population)),  # tools.selNSGA2
        mate=R.mate_crossover_1d,
        mutate=R.mutate_flip_bit_types,
        p_mate=p_mate,
        p_mutate=p_mutate,
    )


def select_population(population: PopulationHint, offspring: PopulationHint) -> PopulationHint:
    return offspring


class DatasetMaskModule(EaModule):

    def __init__(
        self,
        dataset_name: str = 'cars3d',
        population_size: int = 128,
        generations: int = 200,
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
    ):
        super(EaModule, self).__init__()  # skip calling OneMaxProblem __init__
        self.save_hyperparameters()
        # load dataset
        gt_data = H.make_data(dataset_name)
        self._num_obs = len(gt_data)
        # OVERRIDES
        self.select_population = wrapped_partial(
            select_population,
        )
        self.generate_offspring = wrapped_partial(
            generate_offspring,
            p_mate=self.hparams.p_mate,
            p_mutate=self.hparams.p_mutate,
        )
        self.evaluate_member = wrapped_partial(
            evaluate_member,
            gt_dist_matrices_id=ray.put(cached_compute_all_factor_dist_matrices(dataset_name)),
            factor_sizes=gt_data.factor_sizes,
            factor_score_offset=self.hparams.factor_score_offset,
            factor_score_weight=self.hparams.factor_score_weight,
            kept_ratio_offset=self.hparams.kept_ratio_offset,
            kept_ratio_weight=self.hparams.kept_ratio_weight,
            factor_score_mode=self.hparams.factor_score_mode,
            factor_score_agg=self.hparams.factor_score_agg,
        )

    def gen_starting_population(self) -> PopulationHint:
        return [
            Member(np.random.random(self._num_obs) < (0.1 + np.random.random() * 0.8))
            for _ in range(self.hparams.population_size)
        ]

    @property
    def num_generations(self) -> int:
        return self.hparams.generations



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
):
    # save the starting time for the save path
    time_string = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    log.info(f'Starting run at time: {time_string}')

    # determinism
    seed_ = seed_ if (seed_ is not None) else int(np.random.randint(1, 2**31-1))
    seed(seed_)

    # RUN
    with Timer('ruck:onemax'):
        problem = DatasetMaskModule(
            dataset_name=dataset_name,
            population_size=population_size,
            factor_score_mode=factor_score_mode,
            factor_score_agg=factor_score_agg,
            generations=generations,
            factor_score_weight=factor_score_weight,
            kept_ratio_weight=kept_ratio_weight,
        )
        population, logbook, halloffame = Trainer(multiproc=True).fit(problem)

    # plot average images
    H.plt_subplots_imshow(
        [[individual_ave(dataset_name, v) for v in halloffame.values]],
        col_labels=[f'{np.sum(v)} / {np.prod(v.shape)} |' for v in halloffame.values],
        show=True, vmin=0.0, vmax=1.0,
        title=f'{dataset_name}: g{generations} p{population_size} [{factor_score_mode}, {factor_score_agg}]',
    )

    # get save path, make parent dir & save!
    if save:
        job_name = f'{(save_prefix + "_" if save_prefix else "")}{dataset_name}_{generations}x{population_size}_{factor_score_mode}_{factor_score_agg}_{factor_score_weight}_{kept_ratio_weight}'
        save_path = ensure_parent_dir_exists(ROOT_DIR, 'out/adversarial_mask', f'{time_string}_{job_name}_mask.npz')
        log.info(f'saving mask data to: {save_path}')
        np.savez(save_path, mask=halloffame.values[0], params=problem.hparams, seed=seed_)


# ========================================================================= #
# ENTRYPOINT                                                                #
# ========================================================================= #


ROOT_DIR = os.path.abspath(__file__ + '/../../..')


def main():
    from itertools import product

    for (factor_score_agg, factor_score_mode, dataset_name, factor_score_weight) in product(
        ['mean'], #, 'max', 'gmean'],
        ['range'], #, 'std'],
        ['cars3d', 'smallnorb', 'shapes3d', 'dsprites'],
        [-1.0, -30.0, -1000.0],
    ):
        print('='*100)
        print(f'[STARTING]: dataset_name={repr(dataset_name)} factor_score_mode={repr(factor_score_mode)} factor_score_agg={repr(factor_score_agg)} factor_score_weight={repr(factor_score_weight)}')
        try:
            run(
                dataset_name=dataset_name,
                factor_score_mode=factor_score_mode,
                factor_score_agg=factor_score_agg,
                factor_score_weight=factor_score_weight,
                generations=1000,
                seed_=42,
                save=True,
                save_prefix='EXPERIMENT',
            )
        except KeyboardInterrupt:
            warnings.warn('Exiting early')
        except:
            warnings.warn(f'[FAILED]: dataset_name={repr(dataset_name)} factor_score_mode={repr(factor_score_mode)} factor_score_agg={repr(factor_score_agg)} factor_score_weight={repr(factor_score_weight)}')
        print('='*100)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ray.init(num_cpus=24)
    main()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
