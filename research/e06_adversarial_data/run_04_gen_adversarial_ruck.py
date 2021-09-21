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
import numpy as np

from disent.util.profiling import Timer
from research.e01_visual_overlap.util_compute_traversal_dists import cached_compute_all_factor_dist_matrices
from research.e06_adversarial_data.run_04_gen_adversarial_genetic import _toolbox_eval_individual
from research.e06_adversarial_data.run_04_gen_adversarial_genetic import individual_ave
from research.ruck import *
from research.ruck.examples.onemax import OneMaxProblem

import research.util as H


class DatasetMaskProblem(OneMaxProblem):

    def __init__(
        self,
        dataset_name: str = 'cars3d',
        population_size: int = 128,
        # optim settings
        factor_score_weight: float = -1000.0,
        factor_score_offset: float = 1.0,
        kept_ratio_weight: float = 1.0,
        kept_ratio_offset: float = 0.0,
        # ea settings
        p_mate: float = 0.5,
        p_mutate: float = 0.5,
    ):
        super(EaProblem, self).__init__()  # skip calling OneMaxProblem __init__
        self.save_hyperparameters()
        # load and compute dataset
        self.gt_data          = H.make_data(dataset_name)
        self.gt_dist_matrices = cached_compute_all_factor_dist_matrices(dataset_name)

    def get_starting_population_values(self) -> PopulationHint:
        return [
            np.random.random(len(self.gt_data)) < (0.1 + np.random.random() * 0.8)
            for _ in range(self.hparams.population_size)
        ]

    def evaluate_member(self, value: np.ndarray) -> float:
        factor_score, kept_ratio = _toolbox_eval_individual(
            individual=value,
            gt_dist_matrices=self.gt_dist_matrices,
            factor_sizes=self.gt_data.factor_sizes,
            fitness_mode='range',
            obj_mode_aggregate='mean',
            exclude_diag=True,
        )
        return min(
            self.hparams.factor_score_offset + self.hparams.factor_score_weight * factor_score,
            self.hparams.kept_ratio_offset + self.hparams.kept_ratio_weight * kept_ratio,
        )


def run(
    dataset_name: str = 'xysquares_8x8_toy_s2',  # xysquares_8x8_toy_s4, xcolumns_8x_toy_s1
):
    with Timer('ruck:onemax'):
        problem = DatasetMaskProblem(dataset_name=dataset_name)
        population, logbook, halloffame = run_ea(problem, generations=250)

    # plot average images
    H.plt_subplots_imshow(
        [[individual_ave(dataset_name, v) for v in halloffame.values]],
        col_labels=[f'{np.sum(v)} / {np.prod(v.shape)} |' for v in halloffame.values],
        show=True, vmin=0.0, vmax=1.0
        # title=f'{dataset_name}: g{num_generations} p{population_size} [{fitness_mode}, {objective_mode_aggregate}]',
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # ray.init(num_cpus=16)

    run()
