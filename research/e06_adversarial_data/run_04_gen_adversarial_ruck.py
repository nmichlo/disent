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

"""
This file generates pareto-optimal solutions to the multi-objective
optimisation problem of masking a dataset as to minimize some metric
for overlap, while maximizing the amount of data kept.

- We solve this problem using the NSGA2 algorithm and save all the results
  to disk to be loaded with `get_closest_mask` from `util_load_adversarial_mask.py`
"""

import gzip
import logging
import os
import pickle
import random
import warnings
from datetime import datetime
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import ray
import ruck
from matplotlib import pyplot as plt
from ruck import R
from ruck.external.ray import ray_map
from ruck.external.ray import ray_remote_put
from ruck.external.ray import ray_remote_puts

import research.util as H
from disent.dataset.wrapper import MaskedDataset
from disent.util.function import wrapped_partial
from disent.util.inout.paths import ensure_parent_dir_exists
from disent.util.profiling import Timer
from disent.util.seeds import seed
from disent.util.visualize.vis_util import get_idx_traversal
from research.e01_visual_overlap.util_compute_traversal_dists import cached_compute_all_factor_dist_matrices
from research.e06_adversarial_data.util_eval_adversarial import eval_factor_fitness_numba
from research.e06_adversarial_data.util_eval_adversarial import eval_individual


log = logging.getLogger(__name__)


'''
NOTES ON MULTI-OBJECTIVE OPTIMIZATION:
    https://en.wikipedia.org/wiki/Pareto_efficiency
    https://en.wikipedia.org/wiki/Multi-objective_optimization
    https://www.youtube.com/watch?v=SL-u_7hIqjA

    IDEAL MULTI-OBJECTIVE OPTIMIZATION
        1. generate set of pareto-optimal solutions (solutions lying along optimal boundary) -- (A posteriori methods)
           - converge to pareto optimal front
           - maintain as diverse a population as possible along the front (nsga-ii?)
        2. choose one from set using higher level information

    NOTE:
        most multi-objective problems are just
        converted into single objective functions.

    WEIGHTED SUMS
        -- need to know weights
        -- non-uniform in pareto-optimal solutions
        -- cannot find some pareto-optimal solutions in non-convex regions
        `return w0 * score0 + w1 * score1 + ...`

    ε-CONSTRAINT: constrain all but one objective
        -- need to know ε vectors
        -- non-uniform in pareto-optimal solutions
        -- any pareto-optimal solution can be found
        * EMO is a generalisation?
'''


# ========================================================================= #
# Ruck Helper                                                               #
# ========================================================================= #


def select_nsga2(population, num_offspring: int, weights: Tuple[float, ...]):
    # TODO: move this into ruck
    """
    This is hacky... ruck doesn't yet have NSGA2
    support, but we want to use it for this!
    """
    from deap import creator, tools, base
    # initialize creator
    creator.create('IdxFitness', base.Fitness, weights=weights)
    creator.create('IdxIndividual', int, fitness=creator.IdxFitness)
    # convert to deap population
    idx_individuals = []
    for i, m in enumerate(population):
        ind = creator.IdxIndividual(i)
        ind.fitness.values = m.fitness
        idx_individuals.append(ind)
    # run nsga2
    chosen_idx = tools.selNSGA2(individuals=idx_individuals, k=num_offspring)
    # return values
    return [population[i] for i in chosen_idx]


def mutate_oneof(*mutate_fns):
    # TODO: move this into ruck
    def mutate_fn(value):
        fn = random.choice(mutate_fns)
        return fn(value)
    return mutate_fn


def plt_pareto_solutions(
    population,
    label_fitness_0: str,
    label_fitness_1: str,
    title: str = None,
    plot: bool = True,
    chosen_idxs_f0=None,
    chosen_idxs_f1=None,
    random_points=None,
    **fig_kw,
):
    # fitness values must be of type Tuple[float, float] for this function to work!
    fig, axs = H.plt_subplots(1, 1, title=title if title else 'Pareto-Optimal Solutions', **fig_kw)
    # plot fitness values
    xs, ys = zip(*(m.fitness for m in population))
    axs[0, 0].set_xlabel(label_fitness_0)
    axs[0, 0].set_ylabel(label_fitness_1)
    # plot random
    if random_points is not None:
        axs[0, 0].scatter(*np.array(random_points).T, c='orange')
    # plot normal
    axs[0, 0].scatter(xs, ys)
    # plot chosen
    if chosen_idxs_f0 is not None:
        axs[0, 0].scatter(*np.array([population[i].fitness for i in chosen_idxs_f0]).T, c='purple')
    if chosen_idxs_f1 is not None:
        axs[0, 0].scatter(*np.array([population[i].fitness for i in chosen_idxs_f1]).T, c='green')
    # label axes
    # layout
    fig.tight_layout()
    # plot
    if plot:
        plt.show()
    # done!
    return fig, axs


def individual_ave(dataset, individual, print_=False):
    if isinstance(dataset, str):
        dataset = H.make_data(dataset, transform_mode='none')
    # masked
    sub_data = MaskedDataset(data=dataset, mask=individual.flatten())
    if print_:
        print(', '.join(f'{individual.reshape(sub_data._data.factor_sizes).sum(axis=f_idx).mean():2f}' for f_idx in range(sub_data._data.num_factors)))
    # make obs
    ave_obs = np.zeros_like(sub_data[0], dtype='float64')
    for obs in sub_data:
        ave_obs += obs
    return ave_obs / ave_obs.max()


def plot_averages(dataset_name: str, values: list, subtitle: str, title_prefix: str = None, titles=None, show: bool = False):
    data = H.make_data(dataset_name, transform_mode='none')
    # average individuals
    ave_imgs = [individual_ave(data, v) for v in values]
    col_lbls = [f'{np.sum(v)} / {np.prod(v.shape)}' for v in values]
    # make plots
    fig_ave_imgs, _ = H.plt_subplots_imshow(
        [ave_imgs],
        col_labels=col_lbls,
        titles=titles,
        show=show,
        vmin=0.0,
        vmax=1.0,
        figsize=(10, 3),
        title=f'{f"{title_prefix} " if title_prefix else ""}Average Datasets\n{subtitle}',
    )
    return fig_ave_imgs


def get_spaced(array, num: int):
    return [array[i] for i in get_idx_traversal(len(array), num)]


# ========================================================================= #
# Evaluation                                                                #
# ========================================================================= #


@ray.remote
def evaluate_member(
    value: np.ndarray,
    gt_dist_matrices: np.ndarray,
    factor_sizes: Tuple[int, ...],
    fitness_overlap_mode: str,
    fitness_overlap_aggregate: str,
) -> Tuple[float, float]:
    overlap_score, usage_ratio = eval_individual(
        eval_factor_fitness_fn=eval_factor_fitness_numba,
        individual=value,
        gt_dist_matrices=gt_dist_matrices,
        factor_sizes=factor_sizes,
        fitness_overlap_mode=fitness_overlap_mode,
        fitness_overlap_aggregate=fitness_overlap_aggregate,
        exclude_diag=True,
    )

    # weight components
    # assert fitness_overlap_weight >= 0
    # assert fitness_usage_weight >= 0
    # w_ovrlp = fitness_overlap_weight * overlap_score
    # w_usage = fitness_usage_weight * usage_ratio

    # GOALS: minimize overlap, maximize usage
    # [min, max] objective    -> target
    # [  0,   1] factor_score -> 0
    # [  0,   1] kept_ratio   -> 1

    # linear scalarization
    # loss = w_ovrlp - w_usage

    # No-preference method
    # -- norm(f(x) - z_ideal)
    # -- preferably scale variables
    # z_ovrlp = fitness_overlap_weight * (overlap_score - 0.0)
    # z_usage = fitness_usage_weight * (usage_ratio - 1.0)
    # loss = np.linalg.norm([z_ovrlp, z_usage], ord=2)

    # convert minimization problem into maximization
    # return - loss

    return (-overlap_score, usage_ratio)


# ========================================================================= #
# Type Hints                                                                #
# ========================================================================= #


Values = List[ray.ObjectRef]
Population = List[ruck.Member[ray.ObjectRef]]



# ========================================================================= #
# Evolutionary System                                                       #
# ========================================================================= #


class DatasetMaskModule(ruck.EaModule):

    # STATISTICS

    def get_stats_groups(self):
        remote_sum = ray.remote(np.mean).remote
        return {
            **super().get_stats_groups(),
            'mask': ruck.StatsGroup(lambda pop: ray.get([remote_sum(m.value) for m in pop]), min=np.min, max=np.max, mean=np.mean),
        }

    def get_progress_stats(self):
        return ('evals', 'fit:mean', 'mask:mean')

    # POPULATION

    def gen_starting_values(self) -> Values:
        return [
            ray.put(np.random.random(np.prod(self.hparams.factor_sizes)) < (0.1 + np.random.random() * 0.8))
            for _ in range(self.hparams.population_size)
        ]

    def select_population(self, population: Population, offspring: Population) -> Population:
        return select_nsga2(population + offspring, len(population), weights=(1.0, 1.0))

    def evaluate_values(self, values: Values) -> List[float]:
        return ray.get([self._evaluate_value_fn(v) for v in values])

    # INITIALISE

    def __init__(
        self,
        dataset_name: str = 'cars3d',
        dist_normalize_mode: str = 'all',
        population_size: int = 128,
        # fitness settings
        fitness_overlap_aggregate: str = 'mean',
        fitness_overlap_mode: str = 'std',
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
            gt_dist_matrices=ray.put(cached_compute_all_factor_dist_matrices(dataset_name, normalize_mode=dist_normalize_mode)),
            factor_sizes=factor_sizes,
            fitness_overlap_mode=fitness_overlap_mode,
            fitness_overlap_aggregate=fitness_overlap_aggregate,
        )


# ========================================================================= #
# RUNNER                                                                    #
# ========================================================================= #


def run(
    dataset_name: str = 'shapes3d',  # xysquares_8x8_toy_s4, xcolumns_8x_toy_s1
    dist_normalize_mode: str = 'all',  # all, each, none
    generations: int = 250,
    population_size: int = 128,
    # fitness settings
    fitness_overlap_mode: str = 'std',
    fitness_overlap_aggregate: str = 'mean',
    # save settings
    save: bool = False,
    save_prefix: str = '',
    seed_: Optional[int] = None,
    # plot settings
    plot: bool = False,
    # wandb_settings
    wandb_enabled: bool = True,
    wandb_init: bool = True,
    wandb_project: str = 'exp-adversarial-mask',
    wandb_user: str = 'n_michlo',
    wandb_job_name: str = None,
    wandb_tags: Optional[List[str]] = None,
    wandb_finish: bool = True,
):
    # save the starting time for the save path
    time_string = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    log.info(f'Starting run at time: {time_string}')

    # get hparams
    hparams = dict(dataset_name=dataset_name, dist_normalize_mode=dist_normalize_mode, generations=generations, population_size=population_size, fitness_overlap_mode=fitness_overlap_mode, fitness_overlap_aggregate=fitness_overlap_aggregate, save=save, save_prefix=save_prefix, seed_=seed_, plot=plot, wandb_enabled=wandb_enabled, wandb_init=wandb_init, wandb_project=wandb_project, wandb_user=wandb_user, wandb_job_name=wandb_job_name)

    # enable wandb
    wandb = None
    if wandb_enabled:
        import wandb
        # cleanup from old runs:
        if wandb_init:
            if wandb_finish:
                try:
                    wandb.finish()
                except:
                    pass
            # initialize
            wandb.init(
                entity=wandb_user,
                project=wandb_project,
                name=wandb_job_name if (wandb_job_name is not None) else f'{(save_prefix + "_" if save_prefix else "")}{dataset_name}_{generations}x{population_size}_{fitness_overlap_mode}_{fitness_overlap_aggregate}',
                group=None,
                tags=wandb_tags,
            )
        # track hparams
        wandb.config.update({f'adv/{k}': v for k, v in hparams.items()})

    # This is not completely deterministic with ray
    # although the starting population will always be the same!
    seed_ = seed_ if (seed_ is not None) else int(np.random.randint(1, 2**31-1))
    seed(seed_)

    # run!
    with Timer('ruck:onemax'):
        problem = DatasetMaskModule(
            dataset_name=dataset_name,
            dist_normalize_mode=dist_normalize_mode,
            population_size=population_size,
            fitness_overlap_mode=fitness_overlap_mode,
            fitness_overlap_aggregate=fitness_overlap_aggregate,
        )
        # train
        population, logbook, halloffame = ruck.Trainer(generations=generations, progress=True).fit(problem)
        # retrieve stats
        log.info(f'start population: {logbook[0]}')
        log.info(f'end population: {logbook[-1]}')
        values = [ray.get(m.value) for m in halloffame]

    # log to wandb as steps
    if wandb_enabled:
        for i, stats in enumerate(logbook):
            stats = {f'stats/{k}': v for k, v in stats.items()}
            stats['current_step'] = i
            wandb.log(stats, step=i)

    # generate average images
    if plot or wandb_enabled:
        title = f'{dataset_name}: g{generations} p{population_size} [{dist_normalize_mode}, {fitness_overlap_mode}, {fitness_overlap_aggregate}]'
        # plot average
        fig_ave_imgs_hof = plot_averages(dataset_name, values, title_prefix='HOF', subtitle=title, show=plot)
        # get individuals -- this is not ideal because not evenly spaced
        idxs_chosen_f0 = get_spaced(np.argsort([m.fitness[0] for m in population])[::-1], 5)  # overlap
        idxs_chosen_f1 = get_spaced(np.argsort([m.fitness[1] for m in population]),       5)  # usage
        chosen_values_f0    = [ray.get(population[i].value) for i in idxs_chosen_f0]
        chosen_values_f1    = [ray.get(population[i].value) for i in idxs_chosen_f1]
        random_fitnesses = problem.evaluate_values([ray.put(np.random.random(values[0].shape) < p) for p in np.linspace(0.025, 1, num=population_size+2)[1:-1]])
        # plot averages
        fig_ave_imgs_f0 = plot_averages(dataset_name, chosen_values_f0, subtitle=title, titles=[f'{population[i].fitness[0]:2f}' for i in idxs_chosen_f0], title_prefix='Overlap -', show=plot)
        fig_ave_imgs_f1 = plot_averages(dataset_name, chosen_values_f1, subtitle=title, titles=[f'{population[i].fitness[1]:2f}' for i in idxs_chosen_f1], title_prefix='Usage -', show=plot)
        # plot parento optimal solutions
        fig_pareto_sol, axs = plt_pareto_solutions(
            population,
            label_fitness_0='Overlap Score',
            label_fitness_1='Usage Score',
            title=f'Pareto-Optimal Solutions\n{title}',
            plot=plot,
            chosen_idxs_f0=idxs_chosen_f0,
            chosen_idxs_f1=idxs_chosen_f1,
            random_points=random_fitnesses,
            figsize=(7, 7),
        )
        # log average
        if wandb_enabled:
            wandb.log({
                'ave_images_hof':     wandb.Image(fig_ave_imgs_hof),
                'ave_images_overlap': wandb.Image(fig_ave_imgs_f0),
                'ave_images_usage':   wandb.Image(fig_ave_imgs_f1),
                'pareto_solutions':   wandb.Image(fig_pareto_sol),
            })

    # get summary
    use_elems = np.sum(values[0])
    num_elems = np.prod(values[0].shape)
    use_ratio = (use_elems / num_elems)

    # log summary
    if wandb_enabled:
        wandb.summary['num_elements'] = num_elems
        wandb.summary['used_elements'] = use_elems
        wandb.summary['used_elements_ratio'] = use_ratio
        for k, v in logbook[0].items(): wandb.summary[f'log:start:{k}'] = v
        for k, v in logbook[-1].items(): wandb.summary[f'log:end:{k}'] = v

    if save:
        # get save path, make parent dir & save!
        job_name = f'{time_string}_{(save_prefix + "_" if save_prefix else "")}{dataset_name}_{generations}x{population_size}_{dist_normalize_mode}_{fitness_overlap_mode}_{fitness_overlap_aggregate}'
        save_path = ensure_parent_dir_exists(ROOT_DIR, 'out/adversarial_mask', job_name, 'data.pkl.gz')
        log.info(f'saving data to: {save_path}')

        # NONE : 122943493 ~= 118M (100.%) : 103.420ms
        # lvl=1 : 23566691 ~=  23M (19.1%) : 1.223s
        # lvl=2 : 21913595 ~=  21M (17.8%) : 1.463s
        # lvl=3 : 20688319 ~=  20M (16.8%) : 2.504s
        # lvl=4 : 18325859 ~=  18M (14.9%) : 1.856s  # good
        # lvl=5 : 17467772 ~=  17M (14.2%) : 3.332s  # good
        # lvl=6 : 16594660 ~=  16M (13.5%) : 7.163s  # starting to slow
        # lvl=7 : 16242279 ~=  16M (13.2%) : 12.407s
        # lvl=8 : 15586416 ~=  15M (12.7%) : 1m:4s   # far too slow
        # lvl=9 : 15023324 ~=  15M (12.2%) : 3m:11s  # far too slow

        with gzip.open(save_path, 'wb', compresslevel=5) as fp:
            pickle.dump({
                'hparams': hparams,
                'job_name': job_name,
                'time_string': time_string,
                'values': [ray.get(m.value) for m in population],
                'scores': [m.fitness for m in population],
                # score components
                'scores_overlap': [m.fitness[0] for m in population],
                'scores_usage': [m.fitness[1] for m in population],
                # history data
                'logbook_history': logbook.history,
                # we don't want these because they store object refs, and
                # it means we need ray to unpickle them.
                # 'population': population,
                # 'halloffame_members': halloffame.members,
            }, fp)
        # done!
        log.info(f'saved data to: {save_path}')
        # return
        results = save_path
    else:
        # return the population
        results = (population, logbook, halloffame)
    # cleanup wandb
    if wandb_enabled:
        if wandb_finish:
            try:
                wandb.finish()
            except:
                pass
    # done!
    return results

# ========================================================================= #
# ENTRYPOINT                                                                #
# ========================================================================= #


ROOT_DIR = os.path.abspath(__file__ + '/../../..')


def main():
    from itertools import product

    # (3 * 2 * 2 * 5)
    for (dist_normalize_mode, fitness_overlap_aggregate, fitness_overlap_mode, dataset_name) in product(
        ['all', 'each', 'none'],
        ['gmean', 'mean'],
        ['std', 'range'],
        ['xysquares_8x8_toy_s2', 'cars3d', 'smallnorb', 'shapes3d', 'dsprites'],
    ):
        print('='*100)
        print(f'[STARTING]: dataset_name={repr(dataset_name)} dist_normalize_mode={repr(dist_normalize_mode)} fitness_overlap_mode={repr(fitness_overlap_mode)} fitness_overlap_aggregate={repr(fitness_overlap_aggregate)}')
        try:
            run(
                dataset_name=dataset_name,
                dist_normalize_mode=dist_normalize_mode,
                # fitness
                fitness_overlap_aggregate=fitness_overlap_aggregate,
                fitness_overlap_mode=fitness_overlap_mode,
                # population
                generations=1000,
                population_size=256,
                seed_=42,
                save=True,
                save_prefix='EXP',
                plot=True,
                wandb_enabled=True,
                wandb_tags=['exp']
            )
        except KeyboardInterrupt:
            warnings.warn('Exiting early')
            exit(1)
        # except:
        #     warnings.warn(f'[FAILED]: dataset_name={repr(dataset_name)} dist_normalize_mode={repr(dist_normalize_mode)} fitness_overlap_mode={repr(fitness_overlap_mode)} fitness_overlap_aggregate={repr(fitness_overlap_aggregate)}')
        print('='*100)


if __name__ == '__main__':
    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../gadfly.mplstyle'))

    # run
    logging.basicConfig(level=logging.INFO)
    ray.init(num_cpus=64)
    main()

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
