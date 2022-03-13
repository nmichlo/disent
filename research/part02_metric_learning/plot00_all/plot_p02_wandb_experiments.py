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
import logging
import os
from pprint import pprint
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import research.code.util as H
from disent.util.profiling import Timer
from research.code.util._wandb_plots import drop_non_unique_cols
from research.code.util._wandb_plots import drop_unhashable_cols
from research.code.util._wandb_plots import load_runs


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def clear_wandb_cache():
    from research.code.util._wandb_plots import clear_runs_cache
    clear_runs_cache()

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

K_MIG_END = 'MIG Score\n(End)'
K_DCI_END = 'DCI Score\n(End)'
K_MIG_MAX = 'MIG Score'
K_DCI_MAX = 'DCI Score'
K_LCORR_GT_F   = 'Linear Corr.\n(factors)'
K_RCORR_GT_F   = 'Rank Corr.\n(factors)'
K_LCORR_GT_G   = 'Global Linear Corr.\n(factors)'
K_RCORR_GT_G   = 'Global Rank Corr.\n(factors)'
K_LCORR_DATA_F = 'Linear Corr.\n(data)'
K_RCORR_DATA_F = 'Rank Corr.\n(data)'
K_LCORR_DATA_G = 'Global Linear Corr.\n(data)'
K_RCORR_DATA_G = 'Global Rank Corr.\n(data)'
K_AXIS      = 'Axis Ratio'
K_LINE      = 'Linear Ratio'

K_TRIPLET_SCALE  = 'Triplet Scale'   # framework.cfg.triplet_margin_max
K_TRIPLET_MARGIN = 'Triplet Margin'  # framework.cfg.triplet_scale
K_TRIPLET_P      = 'Triplet P'       # framework.cfg.triplet_p
K_DETACH         = 'Detached'        # framework.cfg.detach_decoder
K_TRIPLET_MODE   = 'Triplet Mode'    # framework.cfg.triplet_loss

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


def load_general_data(project: str, include_history: bool = False, keep_cols: Sequence[str] = None):
    # keep columns
    if keep_cols is None:
        keep_cols = []
    keep_cols = list(keep_cols)
    # load data
    with Timer('loading data'):
        df = load_runs(project, include_history=include_history)
    # process data
    with Timer('processing data'):
        # filter out unneeded columns
        df, dropped_hash = drop_unhashable_cols(df, skip=['history'] + keep_cols)
        df, dropped_diverse = drop_non_unique_cols(df, skip=['history'] + keep_cols)
        # rename columns
        df = df.rename(columns={
            'settings/optimizer/lr':                K_LR,
            'EXTRA/tags':                           K_GROUP,
            'dataset/name':                         K_DATASET,
            'framework/name':                       K_FRAMEWORK,
            'settings/framework/beta':              K_BETA,
            'settings/framework/recon_loss':        K_LOSS,
            'settings/model/z_size':                K_Z_SIZE,
            'DUMMY/repeat':                         K_REPEAT,
            'state':                                K_STATE,
            'epoch_metric/mig.discrete_score.max':  K_MIG_MAX,
            'epoch_metric/dci.disentanglement.max': K_DCI_MAX,
            'final_metric/mig.discrete_score.max':  K_MIG_END,
            'final_metric/dci.disentanglement.max': K_DCI_END,
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
            # triplet experiments
            'framework/cfg/triplet_margin_max': K_TRIPLET_MARGIN,
            'framework/cfg/triplet_scale':      K_TRIPLET_SCALE,
            'framework/cfg/triplet_p':          K_TRIPLET_P,
            'framework/cfg/detach_decoder':     K_DETACH,
            'framework/cfg/triplet_loss':       K_TRIPLET_MODE,
        })
    return df


def rename_entries(df: pd.DataFrame):
    df = df.copy()
    # replace values in the df
    for key, value, new_value in [
        (K_DATASET,  'xysquares_minimal',        'xysquares'),
        (K_SCHEDULE, 'adanegtvae_up_all_full',   'Schedule: Both (strong)'),
        (K_SCHEDULE, 'adanegtvae_up_all',        'Schedule: Both (weak)'),
        (K_SCHEDULE, 'adanegtvae_up_ratio_full', 'Schedule: Weight (strong)'),
        (K_SCHEDULE, 'adanegtvae_up_ratio',      'Schedule: Weight (weak)'),
        (K_SCHEDULE, 'adanegtvae_up_thresh',     'Schedule: Threshold'),
        (K_TRIPLET_MODE, 'triplet', 'Triplet Loss (Hard Margin)'),
        (K_TRIPLET_MODE, 'triplet_soft', 'Triplet Loss (Soft Margin)'),
        (K_TRIPLET_P, 1, 'L1 Distance'),
        (K_TRIPLET_P, 2, 'L2 Distance'),
        # (K_SAMPLER, 'gt_dist__manhat',        'Ground-Truth Dist Sampling'),
        # (K_SAMPLER, 'gt_dist__manhat_scaled', 'Ground-Truth Dist Sampling (Scaled)'),
    ]:
        if key in df.columns:
            df[key].replace(value, new_value, inplace=True)
            # df.loc[df[key] == value, key] = new_value
    return df


def drop_and_copy_old_invalid_xysquares(df: pd.DataFrame) -> pd.DataFrame:
    # hack to replace the invalid sampling with the correct sampling!
    df = df.copy()
    # drop invalid items
    sel_drop = (df[K_DATASET] == 'xysquares') & (df[K_SAMPLER] == 'gt_dist__manhat_scaled')
    df = df.drop(df[sel_drop].index)
    # copy invalid items & rename entries
    sel_copy = (df[K_DATASET] == 'xysquares') & (df[K_SAMPLER] == 'gt_dist__manhat')
    assert np.sum(sel_drop) == np.sum(sel_copy), f'drop: {np.sum(sel_drop)}, copy: {np.sum(sel_copy)}'
    df_append = df[sel_copy].copy()
    df_append.loc[df_append[K_SAMPLER] == 'gt_dist__manhat', K_SAMPLER] = 'gt_dist__manhat_scaled'
    # append the rows!
    df = df.append(df_append)
    # done!
    return df


# ========================================================================= #
# Plot Experiments                                                          #
# ========================================================================= #


PINK = '#FE375F'     # usually: Beta-VAE
PURPLE = '#5E5BE5'   # maybe:   Ada-TVAE
LPURPLE = '#b0b6ff'  # maybe:   Ada-TVAE (alt)
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
    metrics: Sequence[str] = (K_MIG_MAX, K_RCORR_GT_F, K_RCORR_DATA_F),  # (K_MIG, K_DCI, K_RCORR_GT_F, K_RCORR_DATA_F, K_AXIS, K_LINE)
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
    df = rename_entries(df)
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
    # PLOT
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=150, ext='.png')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #


# ========================================================================= #
# Experiment p02e02                                                         #
# ========================================================================= #


# TODO: this should be replaced with hard triplet loss experiments!
def plot_e02_axis_triplet_schedules(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    grid_size_v: float = 1.4,
    grid_size_h: float = 2.75,
    metrics: Sequence[str] = (K_MIG_END, K_DCI_END, K_RCORR_GT_F, K_RCORR_DATA_F, K_AXIS, K_LINE),
    adaptive_modes: Sequence[str] = ('dist',),
    title: Optional[Union[str, bool]] = True,
):
    if (title is True) and len(adaptive_modes) == 1:
        title = adaptive_modes[0]
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    with Timer('getting data'):
        df: pd.DataFrame = load_general_data(f'{os.environ["WANDB_USER"]}/MSC-p02e02_axis-aligned-triplet')
    # select run groups
    df = df[df[K_GROUP].isin(['sweep_adanegtvae_params_longmed'])]
    df = df[df[K_ADA_MODE].isin(adaptive_modes)]
    # sort everything
    df = df.sort_values([K_FRAMEWORK, K_DATASET, K_SCHEDULE, K_SAMPLER, K_ADA_MODE])
    # print common key values
    print('K_GROUP:        ', list(df[K_GROUP].unique()))
    print('K_FRAMEWORK:    ', list(df[K_FRAMEWORK].unique()))
    print('K_BETA:         ', list(df[K_BETA].unique()))
    print('K_REPEAT:       ', list(df[K_REPEAT].unique()))
    print('K_STATE:        ', list(df[K_STATE].unique()))
    print('K_DATASET:      ', list(df[K_DATASET].unique()))
    print('K_LR:           ', list(df[K_LR].unique()))
    print('K_TRIPLET_MODE: ', list(df[K_TRIPLET_MODE].unique()))
    print('K_SCHEDULE:     ', list(df[K_SCHEDULE].unique()))
    print('K_SAMPLER:      ', list(df[K_SAMPLER].unique()))
    print('K_ADA_MODE:     ', list(df[K_ADA_MODE].unique()))
    # number of runs
    print(f'total={len(df)}')
    df = rename_entries(df)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # hack to replace the invalid sampling with the correct sampling!
    df = drop_and_copy_old_invalid_xysquares(df)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # axis keys
    col_x = K_DATASET
    col_bars = K_SCHEDULE
    col_hue = K_SAMPLER
    # get rows and columns
    all_bars = list(df[col_bars].unique())
    all_x = list(df[col_x].unique())
    all_y = metrics
    all_hue = list(df[col_hue].unique())
    # num rows and cols
    num_bars = len(all_bars)
    num_x = len(all_x)
    num_y = len(all_y)
    num_hue = len(all_hue)
    # palettes
    colors = [PINK, ORANGE, BLUE, LBLUE, PURPLE]
    hatches = [None, '..']
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # make plot
    fig, axs = plt.subplots(num_y, num_x, figsize=(grid_size_h*num_x, num_y*grid_size_v))
    if isinstance(title, str):
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
    handles_a = [mpl.patches.Patch(label=label, color=color) for label, color in zip(all_bars, colors)]
    fig.legend(handles=handles_a, fontsize=12, bbox_to_anchor=(0.986, 0.03), loc='lower right', ncol=2, labelspacing=0.1)
    # add the legend to the top right plot
    assert np.all(df[col_hue].unique() == ['gt_dist__manhat', 'gt_dist__manhat_scaled']), f'{list(df[col_hue].unique())}'
    handles_b = [mpl.patches.Patch(label='Ground-Truth Dist Sampling',          facecolor='#505050ff', edgecolor='white', hatch=hatches[0]),  # gt_dist__manhat
                 mpl.patches.Patch(label='Ground-Truth Dist Sampling (Scaled)', facecolor='#50505065', edgecolor='white', hatch=hatches[1])]  # gt_dist__manhat_scaled
    fig.legend(handles=handles_b, fontsize=12, bbox_to_anchor=(0.073, 0.03), loc='lower left',  ncol=1, labelspacing=0.1)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # plot!
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=150, ext='.png')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #


def plot_e02_axis_triplet_kl_vs_dist(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    grid_size_v: float = 1.4,
    grid_size_h: float = 2.75,
    metrics: Sequence[str] = (K_MIG_END, K_DCI_END, K_RCORR_GT_F, K_RCORR_DATA_F, K_AXIS, K_LINE),
    sampling_modes: Sequence[str] = ('gt_dist__manhat_scaled',),
    vals_schedules: Optional[Sequence[str]] = ('adanegtvae_up_all', 'adanegtvae_up_ratio', 'adanegtvae_up_thresh',),  # ('adanegtvae_up_all', 'adanegtvae_up_all_full', 'adanegtvae_up_ratio', 'adanegtvae_up_ratio_full', 'adanegtvae_up_thresh')
    title: Optional[Union[str, bool]] = True,
):
    all_schedules = ('adanegtvae_up_all', 'adanegtvae_up_all_full', 'adanegtvae_up_ratio', 'adanegtvae_up_ratio_full', 'adanegtvae_up_thresh')
    if not vals_schedules:
        vals_schedules = all_schedules
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    if (title is True) and len(sampling_modes) == 1:
        title = sampling_modes[0]
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    with Timer('getting data'):
        df: pd.DataFrame = load_general_data(f'{os.environ["WANDB_USER"]}/MSC-p02e02_axis-aligned-triplet')
    # select run groups
    df = df[df[K_GROUP].isin(['sweep_adanegtvae_params_longmed'])]
    df = df[df[K_SCHEDULE].isin(vals_schedules)]
    # sort everything
    df = df.sort_values([K_FRAMEWORK, K_DATASET, K_SCHEDULE, K_SAMPLER, K_ADA_MODE])
    # print common key values
    print('K_GROUP:        ', list(df[K_GROUP].unique()))
    print('K_FRAMEWORK:    ', list(df[K_FRAMEWORK].unique()))
    print('K_BETA:         ', list(df[K_BETA].unique()))
    print('K_REPEAT:       ', list(df[K_REPEAT].unique()))
    print('K_STATE:        ', list(df[K_STATE].unique()))
    print('K_DATASET:      ', list(df[K_DATASET].unique()))
    print('K_LR:           ', list(df[K_LR].unique()))
    print('K_TRIPLET_MODE: ', list(df[K_TRIPLET_MODE].unique()))
    print('K_SCHEDULE:     ', list(df[K_SCHEDULE].unique()))
    print('K_ADA_MODE:     ', list(df[K_ADA_MODE].unique()))
    # number of runs
    df = rename_entries(df)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # hack to replace the invalid sampling with the correct sampling!
    df = drop_and_copy_old_invalid_xysquares(df)
    df = df[df[K_SAMPLER].isin(sampling_modes)]  # we can only filter this after the above!
    print('K_SAMPLER:      ', list(df[K_SAMPLER].unique()))
    print(f'total={len(df)}')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # axis keys
    col_x = K_DATASET
    col_bars = K_SCHEDULE
    col_hue = K_ADA_MODE
    # get rows and columns
    all_bars = list(df[col_bars].unique())
    all_x = list(df[col_x].unique())
    all_y = metrics
    all_hue = list(df[col_hue].unique())
    # num rows and cols
    num_bars = len(all_bars)
    num_x = len(all_x)
    num_y = len(all_y)
    num_hue = len(all_hue)
    # palettes
    colors = [PINK, ORANGE, BLUE, LBLUE, PURPLE]
    hatches = [None, '..']
    idxs = [all_schedules.index(v) for v in vals_schedules]
    colors = [colors[i] for i in idxs]
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # make plot
    fig, axs = plt.subplots(num_y, num_x, figsize=(grid_size_h * num_x, num_y * grid_size_v))
    if isinstance(title, str):
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
    handles_a = [mpl.patches.Patch(label=label, color=color) for label, color in zip(all_bars, colors)]
    fig.legend(
        handles=handles_a, fontsize=12, bbox_to_anchor=(0.986, 0.03), loc='lower right', ncol=2, labelspacing=0.1
    )
    # add the legend to the top right plot
    assert np.all(df[col_hue].unique() == ['dist', 'symmetric_kl']), f'{list(df[col_hue].unique())}'
    handles_b = [mpl.patches.Patch(label='Absolute Difference', facecolor='#505050ff', edgecolor='white', hatch=hatches[0]),  # gt_dist__manhat
                 mpl.patches.Patch(label='KL Divergence',       facecolor='#50505065', edgecolor='white', hatch=hatches[1])]  # gt_dist__manhat_scaled
    fig.legend(handles=handles_b, fontsize=12, bbox_to_anchor=(0.073, 0.03), loc='lower left', ncol=1, labelspacing=0.1)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # plot!
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=150, ext='.png')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #


def plot_e02_axis_triplet_schedule_recon_loss(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    # filtering
    vals_schedules: Optional[Sequence[str]] = ('adanegtvae_up_all', 'adanegtvae_up_ratio', 'adanegtvae_up_thresh',),  # ('adanegtvae_up_all', 'adanegtvae_up_all_full', 'adanegtvae_up_ratio', 'adanegtvae_up_ratio_full', 'adanegtvae_up_thresh')
    vals_detach: Sequence[bool] = (True, False),
    vals_triplet_scale: Sequence[int] = (1.0, 10.0),
    # vals_triplet_mode: Sequence[str] = ('triplet_soft',),]
    color_triplet_soft: str = BLUE,
    color_triplet_hard: str = PURPLE,
):
    all_schedules = ('adanegtvae_up_all', 'adanegtvae_up_all_full', 'adanegtvae_up_ratio', 'adanegtvae_up_ratio_full', 'adanegtvae_up_thresh')
    if not vals_schedules:
        vals_schedules = all_schedules
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df: pd.DataFrame = load_general_data(f'{os.environ["WANDB_USER"]}/MSC-p02e02_axis-aligned-triplet', include_history=True)
    # select run groups
    # TODO: add sweep_adanegtvae_params_longmed:
    #     sampling=gt_dist__manhat,gt_dist__manhat_scaled \
    #     framework.cfg.ada_thresh_mode=dist,symmetric_kl \
    #     framework.cfg.detach_decoder=FALSE \
    #     schedule=adanegtvae_up_all,adanegtvae_up_all_full,adanegtvae_up_ratio,adanegtvae_up_ratio_full,adanegtvae_up_thresh \
    #     framework.cfg.triplet_loss=triplet_soft \
    #     dataset=cars3d,smallnorb,shapes3d,dsprites,X--xysquares
    # CURRENT:
    #     schedule=adanegtvae_up_ratio,adanegtvae_up_all \
    #     framework.cfg.triplet_scale=10.0,1.0 \
    #     framework.cfg.detach_decoder=FALSE,TRUE \
    #     framework.cfg.triplet_loss=triplet,triplet_soft \
    #     dataset=cars3d,smallnorb,shapes3d,dsprites,X--xysquares \
    df = df[df[K_GROUP].isin(['sweep_adanegtvae_alt_params_longmed'])]
    df = df[df[K_SCHEDULE].isin(vals_schedules)]
    df = df[df[K_TRIPLET_SCALE].isin(vals_triplet_scale)]
    df = df[df[K_DETACH].isin(vals_detach)]
    df = df[df[K_TRIPLET_MODE].isin(['triplet'])]
    # sort everything
    df = df.sort_values([K_FRAMEWORK, K_DATASET, K_SCHEDULE, K_TRIPLET_SCALE, K_TRIPLET_MODE, K_DETACH])
    # print common key values
    print('K_GROUP:         ', list(df[K_GROUP].unique()))
    print('K_FRAMEWORK:     ', list(df[K_FRAMEWORK].unique()))
    print('K_BETA:          ', list(df[K_BETA].unique()))
    print('K_REPEAT:        ', list(df[K_REPEAT].unique()))
    print('K_STATE:         ', list(df[K_STATE].unique()))
    print('K_DATASET:       ', list(df[K_DATASET].unique()))
    print('K_LR:            ', list(df[K_LR].unique()))
    print('K_TRIPLET_SCALE: ', list(df[K_TRIPLET_SCALE].unique()))
    print('K_DETACH:        ', list(df[K_DETACH].unique()))
    print('K_TRIPLET_MODE:  ', list(df[K_TRIPLET_MODE].unique()))
    print('K_SCHEDULE:      ', list(df[K_SCHEDULE].unique()))
    print('K_SAMPLER:       ', list(df[K_SAMPLER].unique()))
    print('K_ADA_MODE:      ', list(df[K_ADA_MODE].unique()))
    # number of runs
    df = rename_entries(df)
    print(f'total={len(df)}')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # extract entries from run histories
    # - we need to get the minimum reconstruction loss over the course of the runs
    # - or we need to get the point with the reconstruction loss corresponding to the maximum metric?
    df['recon_loss.min'] = [hist['recon_loss'].min() for hist in df['history']]
    metric_key = 'recon_loss.min'
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    replace = [
        ('Schedule: Both (weak)',     'Schedule:\nBoth (weak)'),
        ('Schedule: Both (strong)',   'Schedule:\nBoth (strong)'),
        ('Schedule: Weight (weak)',   'Schedule:\nWeight (weak)'),
        ('Schedule: Weight (strong)', 'Schedule:\nWeight (strong)'),
        ('Schedule: Threshold',       'Schedule:\nThreshold'),
    ]
    for k, v in replace:
        df[K_SCHEDULE].replace(k, v, inplace=True)

    PALLETTE = {
        'Triplet Loss (Soft Margin)': color_triplet_soft,
        'Triplet Loss (Hard Margin)': color_triplet_hard,
        'Schedule:\nBoth (weak)': PINK,
        'Schedule:\nBoth (strong)': ORANGE,
        'Schedule:\nWeight (weak)': BLUE,
        'Schedule:\nWeight (strong)': LBLUE,
        'Schedule:\nThreshold': PURPLE,
    }

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    dataset_vals = df[K_DATASET].unique()

    fig, axs = plt.subplots(1, len(dataset_vals), figsize=(len(dataset_vals)*3.5, 3.33), squeeze=False)
    axs = axs.reshape(-1)

    # PLOT: MIG
    for i, (ax, dataset_val) in enumerate(zip(axs, dataset_vals)):
        # filter items
        df_filtered = df[df[K_DATASET] == dataset_val]
        # plot everything
        sns.violinplot(data=df_filtered, ax=ax, x=K_SCHEDULE, y=metric_key, palette=PALLETTE, split=True, cut=0, width=0.75, scale='width', inner='quartile')
        ax.set_ylim([0, None])
        if i == 0:
            if ax.get_legend():
                ax.get_legend().remove()
            ax.set_xlabel(None)
            ax.set_ylabel('Minimum Recon. Loss')
        elif i == len(axs) - 1:
            ax.legend(bbox_to_anchor=(0, 0.1), fontsize=11.5, loc='lower left', labelspacing=0.1)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        else:
            if ax.get_legend():
                ax.get_legend().remove()
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        ax.set_title(dataset_val)

    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=300)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    return fig, axs


# ========================================================================= #
# Experiment p02e01                                                         #
# ========================================================================= #


def plot_e01_normal_triplet(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    grid_size_v: float = 1.4,
    grid_size_h: float = 2.75,
    metrics: Sequence[str] = (K_MIG_MAX, K_DCI_MAX, K_RCORR_GT_F, K_RCORR_DATA_F, K_AXIS, K_LINE),
    title: str = None,
    violin: bool = False,
):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df: pd.DataFrame = load_general_data(f'{os.environ["WANDB_USER"]}/MSC-p02e01_triplet-param-tuning')
    # select run groups
    df = df[df[K_GROUP].isin(['sweep_tvae_params_basic_RERUN', 'sweep_tvae_params_basic_RERUN_soft'])]
    df = df[df[K_DETACH].isin([False])]              # True, False
    df = df[df[K_TRIPLET_MARGIN].isin([1.0, 10.0])]  # 1.0, 10.0
    df = df[df[K_TRIPLET_SCALE].isin([0.1, 1.0])]    # 0.1, 1.0
    df = df[df[K_TRIPLET_P].isin([1, 2])]            # 1, 2
    df = df[df[K_TRIPLET_MODE].isin(['triplet', 'triplet_soft'])]  # 'triplet', 'triplet_soft'
    df = df[df[K_SAMPLER].isin(['gt_dist__manhat', 'gt_dist__manhat_scaled'])]  # 'gt_dist__manhat', 'gt_dist__manhat_scaled'
    # sort everything
    df = df.sort_values([K_DATASET, K_SAMPLER, K_TRIPLET_MODE, K_TRIPLET_P, K_TRIPLET_SCALE, K_TRIPLET_MARGIN, K_DETACH])
    # print common key values
    with Timer('values'):
        print('K_GROUP:          ', list(df[K_GROUP].unique()))
        print('K_STATE:          ', list(df[K_STATE].unique()))
        print('K_DATASET:        ', list(df[K_DATASET].unique()))
        print('K_SAMPLER:        ', list(df[K_SAMPLER].unique()))
        print('K_TRIPLET_MODE:   ', list(df[K_TRIPLET_MODE].unique()))
        print('K_TRIPLET_P:      ', list(df[K_TRIPLET_P].unique()))
        print('K_TRIPLET_SCALE:  ', list(df[K_TRIPLET_SCALE].unique()))
        print('K_TRIPLET_MARGIN: ', list(df[K_TRIPLET_MARGIN].unique()))
        print('K_DETACH:         ', list(df[K_DETACH].unique()))
    # number of runs
    print(f'total={len(df)}')
    df = rename_entries(df)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    hatches = [None, '..']

    col_x = K_DATASET
    col_bars = K_TRIPLET_MODE
    col_hue = K_TRIPLET_P

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
        # PINK,
        # ORANGE,
        PURPLE,
        BLUE,
        # LBLUE,
        # GREEN,
        # LGREEN,
    ]

    # hack to replace the invalid sampling with the correct sampling!
    df = drop_and_copy_old_invalid_xysquares(df)

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
            if violin:
                raise NotImplementedError
                # sns.violinplot(ax=ax, x=col_bars, y=key_y, hue=col_hue, data=df_filtered)
            else:
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
    handles_a = [mpl.patches.Patch(label=label, color=color) for label, color in zip(all_bars, colors)]
    fig.legend(handles=handles_a, fontsize=12, bbox_to_anchor=(0.983, 0.03), loc='lower right', ncol=1, labelspacing=0.1)
    assert len(all_hue) == len(hatches) == 2, f'{all_hue}, {hatches}'
    handles_b = [mpl.patches.Patch(label=all_hue[0], facecolor='#505050ff', edgecolor='white', hatch=hatches[0]),
               mpl.patches.Patch(label=all_hue[1], facecolor='#50505065', edgecolor='white', hatch=hatches[1])]
    fig.legend(handles=handles_b, fontsize=12, bbox_to_anchor=(0.077, 0.03), loc='lower left',  ncol=1, labelspacing=0.1)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=150, ext='.png')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #


def plot_e01_normal_triplet_recon_loss(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    color_triplet_soft: str = BLUE,
    color_triplet_hard: str = PURPLE,
    # filtering
    vals_detach: Sequence[bool] = (True,),
    vals_p: Sequence[int] = (1,),
    vals_scale: Sequence[int] = (0.1, 1.0),
    vals_margin: Sequence[int] = (1.0, 10.0),
):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df: pd.DataFrame = load_general_data(f'{os.environ["WANDB_USER"]}/MSC-p02e01_triplet-param-tuning', include_history=True)
    # select run groups
    df = df[df[K_GROUP].isin(['sweep_tvae_params_basic_RERUN', 'sweep_tvae_params_basic_RERUN_soft'])]
    df = df[df[K_DETACH].isin(vals_detach)]              # True, False
    df = df[df[K_TRIPLET_MARGIN].isin(vals_margin)]  # 1.0, 10.0
    df = df[df[K_TRIPLET_SCALE].isin(vals_scale)]    # 0.1, 1.0
    df = df[df[K_TRIPLET_P].isin(vals_p)]            # 1, 2
    df = df[df[K_TRIPLET_MODE].isin(['triplet', 'triplet_soft'])]  # 'triplet', 'triplet_soft'
    df = df[df[K_SAMPLER].isin(['gt_dist__manhat', 'gt_dist__manhat_scaled'])]  # 'gt_dist__manhat', 'gt_dist__manhat_scaled'
    # sort everything
    df = df.sort_values([K_SAMPLER, K_DATASET, K_TRIPLET_MODE, K_TRIPLET_P, K_TRIPLET_SCALE, K_TRIPLET_MARGIN, K_DETACH])
    # print common key values
    with Timer('values'):
        print('K_GROUP:          ', list(df[K_GROUP].unique()))
        print('K_STATE:          ', list(df[K_STATE].unique()))
        print('K_DATASET:        ', list(df[K_DATASET].unique()))
        print('K_SAMPLER:        ', list(df[K_SAMPLER].unique()))
        print('K_TRIPLET_MODE:   ', list(df[K_TRIPLET_MODE].unique()))
        print('K_TRIPLET_P:      ', list(df[K_TRIPLET_P].unique()))
        print('K_TRIPLET_SCALE:  ', list(df[K_TRIPLET_SCALE].unique()))
        print('K_TRIPLET_MARGIN: ', list(df[K_TRIPLET_MARGIN].unique()))
        print('K_DETACH:         ', list(df[K_DETACH].unique()))
    # number of runs
    print(f'total={len(df)}')
    df = rename_entries(df)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # print a list of columns
    # pprint(df.iloc[0]['history'].columns)

    # extract entries from run histories
    # - we need to get the minimum reconstruction loss over the course of the runs
    # - or we need to get the point with the reconstruction loss corresponding to the maximum metric?
    df['recon_loss.min'] = [hist['recon_loss'].min() for hist in df['history']]

    metric_key = 'recon_loss.min'

    # hack to replace the invalid sampling with the correct sampling!
    df = drop_and_copy_old_invalid_xysquares(df)

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # df = df[[K_DATASET, K_FRAMEWORK, K_MIG, K_DCI]]
    df[K_SAMPLER].replace('gt_dist__manhat',        'Dist. Sampling\n', inplace=True)
    df[K_SAMPLER].replace('gt_dist__manhat_scaled', 'Dist. Sampling\n(Scaled)', inplace=True)

    # rename columns
    for key, value, new_value in [
        (K_TRIPLET_MODE, 'Triplet Loss (Hard Margin)', 'Hard-Margin'),
        (K_TRIPLET_MODE, 'Triplet Loss (Soft Margin)', 'Soft-Margin'),
    ]:
        if key in df.columns:
            df[key].replace(value, new_value, inplace=True)

    PALLETTE = {
        'Soft-Margin': color_triplet_soft,
        'Hard-Margin': color_triplet_hard,
    }

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    dataset_vals = df[K_DATASET].unique()

    fig, axs = plt.subplots(1, len(dataset_vals), figsize=(len(dataset_vals)*3.33, 3.33), squeeze=False)
    axs = axs.reshape(-1)

    # PLOT: MIG
    for i, (ax, dataset_val) in enumerate(zip(axs, dataset_vals)):
        # filter items
        df_filtered = df[df[K_DATASET] == dataset_val]
        # plot everything
        sns.violinplot(data=df_filtered, ax=ax, x=K_SAMPLER, y=metric_key, hue=K_TRIPLET_MODE, palette=PALLETTE, split=True, cut=0, width=0.75, scale='width', inner='quartile')
        ax.set_ylim([0, None])
        if i == 0:
            ax.legend(bbox_to_anchor=(0, 0), fontsize=12, loc='lower left', labelspacing=0.1)
            ax.set_xlabel(None)
            ax.set_ylabel('Minimum Recon. Loss')
        else:
            ax.get_legend().remove()
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        ax.set_title(dataset_val)

    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=300)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    return fig, axs


# ========================================================================= #
# EXPERIMENT 03 -- unsupervised triplet!                                    #
# ========================================================================= #


def plot_e03_unsupervised_triplet_scores(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    grid_size_v: float = 3.33,
    grid_size_h: float = 3.33,
    metrics: Sequence[str] = (K_MIG_MAX, K_DCI_MAX, K_RCORR_GT_F, K_RCORR_DATA_F, K_AXIS, K_LINE),
):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df: pd.DataFrame = load_general_data(f'{os.environ["WANDB_USER"]}/MSC-p02e03_unsupervised-axis-triplet', include_history=False)

    # SWEEP STANDARD DATASETS
    #     - framework.cfg.overlap_num=512,1024 \
    #     - dataset=cars3d,smallnorb,shapes3d,dsprites \
    #     X settings.framework.recon_loss=mse \
    #     - framework.cfg.overlap_mine_ratio=0.1,0.2 \
    #     - framework.cfg.overlap_mine_triplet_mode=none,hard_neg,semi_hard_neg,hard_pos,easy_pos \
    # SWEEP XYSQUARES DATASET
    #     - framework.cfg.overlap_num=512,1024 \
    #     - dataset=X--xysquares \
    #     X settings.framework.recon_loss='mse_box_r31_l1.0_k3969.0' \
    #     - framework.cfg.overlap_mine_ratio=0.1,0.2 \
    #     - framework.cfg.overlap_mine_triplet_mode=none,hard_neg,semi_hard_neg,hard_pos,easy_pos \

    K_MINE_NUM   = 'framework/cfg/overlap_num'
    K_MINE_RATIO = 'framework/cfg/overlap_mine_ratio'
    K_MINE_MODE  = 'framework/cfg/overlap_mine_triplet_mode'

    # select run groups
    df = df[df[K_GROUP].isin(['sweep_dotvae_hard_params_longmed', 'sweep_dotvae_hard_params_longmed_xy', 'sweep_dotvae_hard_params_9_long'])]
    df = df[df[K_MINE_MODE].isin(['none'])]

    # sort everything
    df = df.sort_values([K_DATASET, K_MINE_MODE, K_MINE_RATIO, K_MINE_NUM])
    # print common key values
    with Timer('values'):
        print('K_GROUP:          ', list(df[K_GROUP].unique()))
        print('K_STATE:          ', list(df[K_STATE].unique()))
        print('K_DATASET:        ', list(df[K_DATASET].unique()))
        print('K_SAMPLER:        ', list(df[K_SAMPLER].unique()))
        print('K_TRIPLET_MODE:   ', list(df[K_TRIPLET_MODE].unique()))
        print('K_TRIPLET_P:      ', list(df[K_TRIPLET_P].unique()))
        print('K_TRIPLET_SCALE:  ', list(df[K_TRIPLET_SCALE].unique()))
        print('K_TRIPLET_MARGIN: ', list(df[K_TRIPLET_MARGIN].unique()))
        print('K_DETACH:         ', list(df[K_DETACH].unique()))
    # number of runs
    print(f'total={len(df)}')
    df = rename_entries(df)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # axis keys
    col_bars = K_DATASET
    # get rows and columns
    all_bars = list(df[col_bars].unique())
    all_x = metrics
    # num rows and cols
    num_bars = len(all_bars)
    num_x = len(all_x)
    # palettes
    colors = [PURPLE, BLUE, LBLUE, LGREEN, '#889299']
    # hatches = [None, '..']
    # idxs = [all_schedules.index(v) for v in vals_schedules]
    # colors = [colors[i] for i in idxs]
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # make plot
    fig, axs = plt.subplots(1, num_x, figsize=(grid_size_h * num_x, 1 * grid_size_v))
    # fill plot
    for x, key_x in enumerate(all_x):  # schedule
        ax = axs[x]
        # plot!
        sns.barplot(ax=ax, x=col_bars, y=key_x, data=df)
        ax.set(ylim=(-0.05, 1.05), xlim=(-0.5, num_bars - 0.5))
        # set colors
        for i, (bar, color) in enumerate(zip(ax.patches, itertools.cycle(colors))):
            bar.set_facecolor(color)
            bar.set_edgecolor('white')
            bar.set_linewidth(2)
        # remove labels
        # ax.set_title(key_x)
        if ax.get_legend():
            ax.get_legend().remove()
        # if y < num_y-1:
        ax.set_xlabel(None)
        ax.set_xticklabels([])
        # if x > 0:
        #     ax.set_ylabel(None)
        #     ax.set_yticklabels([])
    # add the legend to the top right plot
    handles_a = [mpl.patches.Patch(label=label, color=color) for label, color in zip(all_bars, colors)]
    fig.legend(handles=handles_a, fontsize=12, bbox_to_anchor=(0.99, 0.1), loc='lower right', ncol=1, labelspacing=0.1)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # plot!
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=150, ext='.png')
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #


# ========================================================================= #
# Entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':

    assert 'WANDB_USER' in os.environ, 'specify "WANDB_USER" environment variable'

    logging.basicConfig(level=logging.INFO)

    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../../code/util/gadfly.mplstyle'))

    # clear_wandb_cache()

    def main():
        plot_e00_beta_metric_correlation(rel_path='plots/p02e00_metrics_some', show=True, metrics=(K_MIG_MAX, K_RCORR_GT_F, K_RCORR_DATA_F))
        plot_e00_beta_metric_correlation(rel_path='plots/p02e00_metrics_all',  show=True, metrics=(K_MIG_MAX, K_RCORR_GT_F, K_RCORR_DATA_F, K_AXIS, K_LINE))
        plot_e00_beta_metric_correlation(rel_path='plots/p02e00_metrics_alt',  show=True, metrics=(K_MIG_MAX, K_RCORR_GT_F, K_RCORR_DATA_F, K_LINE))

        plot_e01_normal_triplet(rel_path='plots/p02e01_normal-triplet', show=True)
        plot_e01_normal_triplet_recon_loss(rel_path='plots/p02e01_normal-l1-triplet_recon-loss_detached', vals_detach=(True,),  vals_p=(1,), show=True)
        plot_e01_normal_triplet_recon_loss(rel_path='plots/p02e01_normal-l1-triplet_recon-loss_attached', vals_detach=(False,), vals_p=(1,), show=True)

        plot_e02_axis_triplet_kl_vs_dist(rel_path='plots/p02e02_axis__soft-triplet__kl-vs-dist', show=True, title=False)
        plot_e02_axis_triplet_schedules(rel_path='plots/p02e02_axis__soft-triplet__dist',   show=True, adaptive_modes=('dist',), title=False)

        plot_e02_axis_triplet_schedule_recon_loss(rel_path='plots/p02e02_ada-triplet_recon-loss_detached', vals_detach=(True,), show=True)
        plot_e02_axis_triplet_schedule_recon_loss(rel_path='plots/p02e02_ada-triplet_recon-loss_attached', vals_detach=(False,), show=True)

        plot_e03_unsupervised_triplet_scores(rel_path='plots/p02e03_unsupervised-triplet')

    main()


# ========================================================================= #
# DONE                                                                      #
# ========================================================================= #
