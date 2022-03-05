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
import itertools
import os
from pprint import pprint
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from cachier import cachier as _cachier
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import research.code.util as H
from disent.util.function import wrapped_partial


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #
from research.code.util._wandb_plots import clear_runs_cache


cachier = wrapped_partial(_cachier, cache_dir='./cache')
DF = pd.DataFrame


from research.code.util._wandb_plots import drop_non_unique_cols
from research.code.util._wandb_plots import drop_unhashable_cols
from research.code.util._wandb_plots import load_runs


# ========================================================================= #
# Prepare Data                                                              #
# ========================================================================= #


# common keys
K_GROUP     = 'Run Group'
K_DATASET   = 'Dataset'
K_FRAMEWORK = 'Framework'
K_BETA      = 'Beta'
K_LOSS      = 'Recon. Loss'
K_Z_SIZE    = 'Latent Dims.'
K_REPEAT    = 'Repeat'
K_STATE     = 'State'
K_LR        = 'Learning Rate'
K_SCHEDULE  = 'Schedule'
K_SAMPLER   = 'Sampler'
K_ADA_MODE  = 'Threshold Mode'

K_MIG       = 'MIG Score'
K_DCI       = 'DCI Score'
K_LCORR_GT_F   = 'Linear Corr. (factors)'
K_RCORR_GT_F   = 'Rank Corr. (factors)'
K_LCORR_GT_G   = 'Global Linear Corr. (factors)'
K_RCORR_GT_G   = 'Global Rank Corr. (factors)'
K_LCORR_DATA_F = 'Linear Corr. (data)'
K_RCORR_DATA_F = 'Rank Corr. (data)'
K_LCORR_DATA_G = 'Global Linear Corr. (data)'
K_RCORR_DATA_G = 'Global Rank Corr. (data)'
K_AXIS      = 'Axis Ratio'
K_LINE      = 'Linear Ratio'

# ALL_K_SCORES = [
#     K_MIG,
#     K_DCI,
#     # K_LCORR_GT_F,
#     K_RCORR_GT_F,
#     # K_LCORR_GT_G,
#     # K_RCORR_GT_G,
#     # K_LCORR_DATA_F,
#     K_RCORR_DATA_F,
#     # K_LCORR_DATA_G,
#     # K_RCORR_DATA_G,
#     # K_LINE,
#     # K_AXIS,
# ]

def load_general_data(project: str):
    # load data
    df = load_runs(project)
    # filter out unneeded columns
    df, dropped_hash = drop_unhashable_cols(df)
    df, dropped_diverse = drop_non_unique_cols(df)
    # rename columns
    return df.rename(columns={
        'settings/optimizer/lr':                K_LR,
        'EXTRA/tags':                           K_GROUP,
        'dataset/name':                         K_DATASET,
        'framework/name':                       K_FRAMEWORK,
        'settings/framework/beta':              K_BETA,
        'settings/framework/recon_loss':        K_LOSS,
        'settings/model/z_size':                K_Z_SIZE,
        'DUMMY/repeat':                         K_REPEAT,
        'state':                                K_STATE,
        'final_metric/mig.discrete_score.max':  K_MIG,
        'final_metric/dci.disentanglement.max': K_DCI,
        # scores
        'epoch_metric/distances.lcorr_ground_latent.l1.factor.max': K_LCORR_GT_F,
        'epoch_metric/distances.rcorr_ground_latent.l1.factor.max': K_RCORR_GT_F,
        'epoch_metric/distances.lcorr_ground_latent.l1.global.max': K_LCORR_GT_G,
        'epoch_metric/distances.rcorr_ground_latent.l1.global.max': K_RCORR_GT_G,
        'epoch_metric/distances.lcorr_latent_data.l2.factor.max':   K_LCORR_DATA_F,
        'epoch_metric/distances.rcorr_latent_data.l2.factor.max':   K_RCORR_DATA_F,
        'epoch_metric/distances.lcorr_latent_data.l2.global.max':   K_LCORR_DATA_G,
        'epoch_metric/distances.rcorr_latent_data.l2.global.max':   K_RCORR_DATA_G,
        'epoch_metric/linearity.axis_ratio.var.max':                K_AXIS,
        'epoch_metric/linearity.linear_ratio.var.max':              K_LINE,
        # adaptive methods
        'schedule/name': K_SCHEDULE,
        'sampling/name': K_SAMPLER,
        'framework/cfg/ada_thresh_mode': K_ADA_MODE,
    })


# ========================================================================= #
# Plot Experiments                                                          #
# ========================================================================= #


PINK = '#FE375F'     # usually: Beta-VAE
PURPLE = '#5E5BE5'   # maybe:   Ada-TVAE
BLUE = '#1A93FE'     # maybe:   TVAE
LBLUE = '#63D2FE'
ORANGE = '#FE9F0A'   # usually: Ada-VAE
GREEN = '#2FD157'

LGREEN = '#9FD911'   # usually: MSE
LBLUE2 = '#36CFC8'   # usually: MSE-Overlap


# ========================================================================= #
# Experiment p02e00                                                         #
# ========================================================================= #


def plot_e00_beta_metric_correlation(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    color_betavae: str = PINK,
    color_adavae: str = ORANGE,
    grid_size_v: float = 2.25,
    grid_size_h: float = 2.75,
    metrics: Sequence[str] = (K_MIG, K_RCORR_GT_F, K_RCORR_DATA_F),  # (K_MIG, K_DCI, K_RCORR_GT_F, K_RCORR_DATA_F, K_AXIS, K_LINE)
):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df: pd.DataFrame = load_general_data(f'{os.environ["WANDB_USER"]}/MSC-p02e00_beta-data-latent-corr')
    # select run groups
    df = df[df[K_GROUP].isin(['sweep_beta_corr'])]
    df = df[~df[K_DATASET].isin(['xyobject'])]
    df = df[df[K_STATE].isin(['finished', 'running'])]
    # df = df[df[K_LR].isin([0.0001])]  # 0.0001, 0.001
    # sort everything
    df = df.sort_values([K_FRAMEWORK, K_DATASET, K_BETA, K_LR])
    # print common key values
    print('K_GROUP:    ',   list(df[K_GROUP].unique()))
    print('K_FRAMEWORK:',   list(df[K_FRAMEWORK].unique()))
    print('K_BETA:     ',   list(df[K_BETA].unique()))
    print('K_REPEAT:   ',   list(df[K_REPEAT].unique()))
    print('K_STATE:    ',   list(df[K_STATE].unique()))
    print('K_DATASET:  ',   list(df[K_DATASET].unique()))
    print('K_LR:       ',   list(df[K_LR].unique()))
    # number of runs
    print(f'total={len(df)}')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # plot vars
    x_m = 10**-4 / (10**0.5)
    x_M = 10**0 * (10**0.5)
    y_m = -0.05
    y_M = 1.05
    ax_m = x_m * 1.05
    ax_M = x_M / 1.05
    ay_m = y_m + 0.01
    ay_M = y_M - 0.01

    # rename entries
    df.loc[df[K_DATASET] == 'xysquares_minimal', K_DATASET] = 'xysquares'

    # manual remove invalid values
    df = df[df[K_LINE].notna()]
    df = df[df[K_AXIS].notna()]
    # make sure we do not have invalid values!
    for k in metrics:
        df = df[df[k].notna()]
    #     df = df[df[k] >= 0]
    #     df = df[df[k] <= 1]
    # for k in metrics:
    #     df[k] = df[k].replace(np.nan, 0)

    ALL_DATASETS = list(df[K_DATASET].unique())
    num_datasets = len(df[K_DATASET].unique())
    num_scores = len(metrics)

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # make plot
    fig, axs = plt.subplots(num_scores, num_datasets, figsize=(grid_size_h*num_datasets, num_scores*grid_size_v))
    # fill plot
    for i, score_key in enumerate(metrics):
        for j, dataset in enumerate(ALL_DATASETS):
            # filter data
            ada_df  = df[(df[K_FRAMEWORK] == 'adavae_os') & (df[K_DATASET] == dataset)]
            beta_df = df[(df[K_FRAMEWORK] == 'betavae') & (df[K_DATASET] == dataset)]
            # plot!
            # sns.regplot(ax=axs[i, j], x=K_BETA, y=score_key, data=ada_df,  seed=777, order=2, robust=False, color=color_adavae,  marker='o')
            # sns.regplot(ax=axs[i, j], x=K_BETA, y=score_key, data=beta_df, seed=777, order=2, robust=False, color=color_betavae, marker='x', line_kws=dict(linestyle='dashed'))
            sns.lineplot(ax=axs[i, j], x=K_BETA, y=score_key, data=ada_df,  ci=None, color=color_adavae)
            sns.lineplot(ax=axs[i, j], x=K_BETA, y=score_key, data=beta_df, ci=None, color=color_betavae, linestyle='dashed')
            sns.scatterplot(ax=axs[i, j], x=K_BETA, y=score_key, data=ada_df, color=color_adavae,  marker='o')
            sns.scatterplot(ax=axs[i, j], x=K_BETA, y=score_key, data=beta_df, color=color_betavae, marker='X')
            # set the axis limits
            axs[i, j].set(xscale='log', ylim=(y_m, y_M), xlim=(x_m, x_M))
            # hide labels
            if i == 0:
                axs[i, j].set_title(dataset)
            if i < num_scores-1:
                axs[i, j].set_xlabel(None)
                axs[i, j].set_xticklabels([])
            if j > 0:
                axs[i, j].set_ylabel(None)
                axs[i, j].set_yticklabels([])
            # draw border
            axs[i, j].plot([ax_m, ax_m], [ay_m, ay_M], color='#cccccc', linewidth=1)  # left
            axs[i, j].plot([ax_m, ax_M], [ay_m, ay_m], color='#cccccc', linewidth=1)  # bottom
            axs[i, j].plot([ax_M, ax_M], [ay_m, ay_M], color='#cccccc', linewidth=1)  # right
            axs[i, j].plot([ax_m, ax_M], [ay_M, ay_M], color='#cccccc', linewidth=1)  # top
            # add axis labels
            axs[i, j].set_xticks([10**i for i in [-4, -3, -2, -1, 0]])
    # add the legend to the top right plot
    marker_ada = mlines.Line2D([], [], color=color_adavae, marker='o', markersize=12, label='Ada-GVAE')
    marker_beta = mlines.Line2D([], [], color=color_betavae, marker='X', markersize=12, label='Beta-VAE')  # why does 'x' not work? only 'X'?
    axs[0, -1].legend(handles=[marker_beta, marker_ada], fontsize=14)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=150, ext='.png')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #


# ========================================================================= #
# Experiment p02e02                                                         #
# ========================================================================= #


def plot_e02_axis_triplet(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    grid_size_v: float = 2.10,
    grid_size_h: float = 2.75,
    metrics: Sequence[str] = (K_MIG, K_RCORR_GT_F, K_RCORR_DATA_F),  # (K_MIG, K_DCI, K_RCORR_GT_F, K_RCORR_DATA_F, K_AXIS, K_LINE)
    adaptive_modes: Sequence[str] = ('symmetric_kl',),
    title: str = None,
):
    if (title is None) and len(adaptive_modes) == 1:
        title = adaptive_modes[0]
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df: pd.DataFrame = load_general_data(f'{os.environ["WANDB_USER"]}/MSC-p02e02_axis-aligned-triplet')
    # select run groups
    df = df[df[K_GROUP].isin(['sweep_adanegtvae_params_longmed'])]
    df = df[df[K_ADA_MODE].isin(adaptive_modes)]
    # sort everything
    df = df.sort_values([K_FRAMEWORK, K_DATASET, K_ADA_MODE, K_SCHEDULE, K_SAMPLER])
    # print common key values
    print('K_GROUP:    ',   list(df[K_GROUP].unique()))
    print('K_FRAMEWORK:',   list(df[K_FRAMEWORK].unique()))
    print('K_BETA:     ',   list(df[K_BETA].unique()))
    print('K_REPEAT:   ',   list(df[K_REPEAT].unique()))
    print('K_STATE:    ',   list(df[K_STATE].unique()))
    print('K_DATASET:  ',   list(df[K_DATASET].unique()))
    print('K_LR:       ',   list(df[K_LR].unique()))
    print('K_SCHEDULE: ',   list(df[K_SCHEDULE].unique()))
    print('K_SAMPLER:  ',   list(df[K_SAMPLER].unique()))
    print('K_ADA_MODE: ',   list(df[K_ADA_MODE].unique()))
    # number of runs
    print(f'total={len(df)}')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # replace values in the df
    for key, value, new_value in [
        (K_DATASET,  'xysquares_minimal',        'xysquares'),
        (K_SCHEDULE, 'adanegtvae_up_all_full',   'Incr. All (full)'),
        (K_SCHEDULE, 'adanegtvae_up_all',        'Incr. All (semi)'),
        (K_SCHEDULE, 'adanegtvae_up_ratio_full', 'Incr. Ratio (full)'),
        (K_SCHEDULE, 'adanegtvae_up_ratio',      'Incr. Ratio (semi)'),
        (K_SCHEDULE, 'adanegtvae_up_thresh',     'Incr. Thresh (semi)'),
    ]:
        df.loc[df[key] == value, key] = new_value

    hatches = [None, '..']

    # col_x = K_SCHEDULE
    # col_bars = K_DATASET
    col_x = K_DATASET
    col_bars = K_SCHEDULE
    col_hue = K_SAMPLER

    # get rows and columns
    all_bars = list(df[col_bars].unique())
    all_x = list(df[col_x].unique())
    all_y = metrics
    all_hue = list(df[col_hue].unique())

    num_bars = len(all_bars)
    num_x = len(all_x)
    num_y = len(all_y)
    num_hue = len(all_hue)

    colors = [
        PINK,
        ORANGE,
        BLUE,
        LBLUE,
        # GREEN,
        # LGREEN,
        PURPLE,
    ]

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # make plot
    fig, axs = plt.subplots(num_y, num_x, figsize=(grid_size_h*num_x, num_y*grid_size_v))
    if title:
        fig.suptitle(title, fontsize=19)
    # fill plot
    for y, key_y in enumerate(all_y):  # metric
        for x, key_x in enumerate(all_x):  # schedule
            ax = axs[y, x]
            # filter items
            df_filtered = df[(df[col_x] == key_x)]
            # plot!
            sns.barplot(ax=ax, x=col_bars, y=key_y, hue=col_hue, data=df_filtered)
            ax.set(ylim=(0, 1), xlim=(-0.5, num_bars - 0.5))
            # set colors
            for i, (bar, color) in enumerate(zip(ax.patches, itertools.cycle(colors))):
                bar.set_facecolor(color)
                bar.set_edgecolor('white')
                bar.set_linewidth(2)
                if hatches[i // num_bars]:
                    bar.set_facecolor(color + '65')
                    bar.set_hatch(hatches[i // num_bars])
            # remove labels
            if ax.get_legend():
                ax.get_legend().remove()
            if y == 0:
                ax.set_title(key_x)
            # if y < num_y-1:
            ax.set_xlabel(None)
            ax.set_xticklabels([])
            if x > 0:
                ax.set_ylabel(None)
                ax.set_yticklabels([])
    # add the legend to the top right plot
    handles = [mpl.patches.Patch(label=label, color=color) for label, color in zip(all_bars, colors)]
    axs[0, -1].legend(handles=handles, fontsize=14)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=150, ext='.png')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #


# ========================================================================= #
# Entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':

    assert 'WANDB_USER' in os.environ, 'specify "WANDB_USER" environment variable'

    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../../code/util/gadfly.mplstyle'))

    # clear_runs_cache()

    def main():
        # plot_e00_beta_metric_correlation(rel_path='plots/p02e00_metrics_some', show=True, metrics=(K_MIG, K_RCORR_GT_F, K_RCORR_DATA_F))
        # plot_e00_beta_metric_correlation(rel_path='plots/p02e00_metrics_all',  show=True, metrics=(K_MIG, K_RCORR_GT_F, K_RCORR_DATA_F, K_AXIS, K_LINE))
        # plot_e00_beta_metric_correlation(rel_path='plots/p02e00_metrics_alt',  show=True, metrics=(K_MIG, K_RCORR_GT_F, K_RCORR_DATA_F, K_LINE))
        plot_e02_axis_triplet(rel_path='plots/p02e02_axis_triplet', show=True)

    main()


# ========================================================================= #
# DONE                                                                      #
# ========================================================================= #
