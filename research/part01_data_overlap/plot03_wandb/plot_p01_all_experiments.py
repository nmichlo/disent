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

import os
from typing import Optional
from typing import Tuple

import pandas as pd
import seaborn as sns
from cachier import cachier as _cachier
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

import research.code.util as H
from disent.util.function import wrapped_partial


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


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
K_SPACING   = 'Grid Spacing'
K_BETA      = 'Beta'
K_LOSS      = 'Recon. Loss'
K_Z_SIZE    = 'Latent Dims.'
K_REPEAT    = 'Repeat'
K_STATE     = 'State'
K_MIG_END   = 'MIG Score\n(End)'
K_DCI_END   = 'DCI Score\n(End)'
K_MIG_MAX   = 'MIG Score'
K_DCI_MAX   = 'DCI Score'


def load_general_data(project: str):
    # load data
    df = load_runs(project)
    # filter out unneeded columns
    df, dropped_hash = drop_unhashable_cols(df)
    df, dropped_diverse = drop_non_unique_cols(df)
    # rename columns
    return df.rename(columns={
        'EXTRA/tags':                           K_GROUP,
        'dataset/name':                         K_DATASET,
        'framework/name':                       K_FRAMEWORK,
        'dataset/data/grid_spacing':            K_SPACING,
        'settings/framework/beta':              K_BETA,
        'settings/framework/recon_loss':        K_LOSS,
        'settings/model/z_size':                K_Z_SIZE,
        'DUMMY/repeat':                         K_REPEAT,
        'state':                                K_STATE,
        'final_metric/mig.discrete_score.max':  K_MIG_END,
        'final_metric/dci.disentanglement.max': K_DCI_END,
        'epoch_metric/mig.discrete_score.max':  K_MIG_MAX,
        'epoch_metric/dci.disentanglement.max': K_DCI_MAX,
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
# Experiment 2                                                              #
# ========================================================================= #


def plot_e02_incr_overlap_xysquares(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    reg_order: int = 4,
    color_betavae: str = PINK,
    color_adavae: str = ORANGE,
    titles: bool = False,
    figsize: Tuple[float, float] = (10, 5),
    include: Tuple[str] = ('mig', 'dci'),
):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df = load_general_data(f'{os.environ["WANDB_USER"]}/CVPR-01__incr_overlap')
    # select run groups
    df = df[df[K_GROUP].isin(['sweep_xy_squares_overlap', 'sweep_xy_squares_overlap_small_beta'])]
    # print common key values
    print('K_GROUP:    ', list(df[K_GROUP].unique()))
    print('K_FRAMEWORK:', list(df[K_FRAMEWORK].unique()))
    print('K_SPACING:  ', list(df[K_SPACING].unique()))
    print('K_BETA:     ', list(df[K_BETA].unique()))
    print('K_REPEAT:   ', list(df[K_REPEAT].unique()))
    print('K_STATE:    ', list(df[K_STATE].unique()))
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    BETA = 0.00316   # if grid_spacing <  6
    BETA = 0.001     # if grid_spacing >= 6

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    orig = df
    # select runs
    # df = df[df[K_STATE] == 'finished']
    # df = df[df[K_REPEAT].isin([1, 2, 3])]
    # select adavae
    adavae_selector = (df[K_FRAMEWORK] == 'adavae_os') & (df[K_BETA] == 0.001)  # 0.001, 0.0001
    data_adavae = df[adavae_selector]
    # select
    betavae_selector_a = (df[K_FRAMEWORK] == 'betavae')   & (df[K_BETA] == 0.001)   & (df[K_SPACING] >= 3)
    betavae_selector_b = (df[K_FRAMEWORK] == 'betavae')   & (df[K_BETA] == 0.00316) & (df[K_SPACING] < 3)
    data_betavae = df[betavae_selector_a | betavae_selector_b]
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    print('ADAGVAE', len(orig), '->', len(data_adavae))
    print('BETAVAE', len(orig), '->', len(data_betavae))

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    fig, axs = plt.subplots(1, len(include), figsize=figsize, squeeze=False)
    axs = axs.reshape(-1)
    # Legend entries
    marker_ada  = mlines.Line2D([], [], color=color_adavae,  marker='o', markersize=12, label='Ada-GVAE')
    marker_beta = mlines.Line2D([], [], color=color_betavae, marker='X', markersize=12, label='Beta-VAE')  # why does 'x' not work? only 'X'?
    # PLOT: MIG
    if 'mig' in include:
        sns.regplot(ax=axs[0], x=K_SPACING, y=K_MIG_END, data=data_adavae,  seed=777, order=reg_order, robust=False, color=color_adavae,  marker='o')
        sns.regplot(ax=axs[0], x=K_SPACING, y=K_MIG_END, data=data_betavae, seed=777, order=reg_order, robust=False, color=color_betavae, marker='x', line_kws=dict(linestyle='dashed'))
        axs[0].legend(handles=[marker_beta, marker_ada], fontsize=14)
        axs[0].set_ylim([-0.1, 1.1])
        axs[0].set_xlim([0.8, 8.2])
        if titles: axs[0].set_title('Framework Mig Scores')
    # PLOT: DCI
    if 'dci' in include:
        sns.regplot(ax=axs[-1], x=K_SPACING, y=K_DCI_END, data=data_adavae,  seed=777, order=reg_order, robust=False, color=color_adavae,  marker='o')
        sns.regplot(ax=axs[-1], x=K_SPACING, y=K_DCI_END, data=data_betavae, seed=777, order=reg_order, robust=False, color=color_betavae, marker='x', line_kws=dict(linestyle='dashed'))
        axs[-1].legend(handles=[marker_beta, marker_ada], fontsize=14)
        axs[-1].set_ylim([-0.1, 1.1])
        axs[-1].set_xlim([0.8, 8.2])
        if titles: axs[-1].set_title('Framework DCI Scores')
    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    return fig, axs


# ========================================================================= #
# Experiment 1                                                              #
# ========================================================================= #


def plot_e01_hparam_tuning(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    color_betavae: str = PINK,
    color_adavae: str = ORANGE,
):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df = load_general_data(f'{os.environ["WANDB_USER"]}/CVPR-00__basic-hparam-tuning')
    # select run groups
    df = df[df[K_GROUP].isin(['sweep_beta'])]
    # print common key values
    print('K_GROUP:    ', list(df[K_GROUP].unique()))
    print('K_DATASET:  ', list(df[K_DATASET].unique()))
    print('K_FRAMEWORK:', list(df[K_FRAMEWORK].unique()))
    print('K_BETA:     ', list(df[K_BETA].unique()))
    print('K_Z_SIZE:   ', list(df[K_Z_SIZE].unique()))
    print('K_REPEAT:   ', list(df[K_REPEAT].unique()))
    print('K_STATE:    ', list(df[K_STATE].unique()))
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    orig = df
    # select runs
    df = df[df[K_STATE] == 'finished']
    # [1.0, 0.316, 0.1, 0.0316, 0.01, 0.00316, 0.001, 0.000316]
    # df = df[(0.000316 < df[K_BETA]) & (df[K_BETA] < 1.0)]
    print('NUM', len(orig), '->', len(df))
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    df = df[[K_DATASET, K_FRAMEWORK, K_MIG_END, K_DCI_END]]
    df[K_DATASET].replace('xysquares_minimal', 'XYSquares', inplace=True)
    df[K_DATASET].replace('smallnorb', 'NORB', inplace=True)
    df[K_DATASET].replace('cars3d', 'Cars3D', inplace=True)
    df[K_DATASET].replace('3dshapes', 'Shapes3D', inplace=True)
    df[K_DATASET].replace('dsprites', 'dSprites', inplace=True)
    df[K_FRAMEWORK].replace('adavae_os', 'Ada-GVAE', inplace=True)
    df[K_FRAMEWORK].replace('betavae', 'Beta-VAE', inplace=True)
    PALLETTE = {'Ada-GVAE': color_adavae, 'Beta-VAE': color_betavae}

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    (ax0, ax1) = axs
    # PLOT: MIG
    sns.violinplot(x=K_DATASET, y=K_MIG_END, hue=K_FRAMEWORK, palette=PALLETTE, split=True, cut=0, width=0.75, data=df, ax=ax0, scale='width', inner='quartile')
    ax0.set_ylim([-0.1, 1.1])
    ax0.legend(bbox_to_anchor=(0.425, 0.9), fontsize=13)
    sns.violinplot(x=K_DATASET, y=K_DCI_END, hue=K_FRAMEWORK, palette=PALLETTE, split=True, cut=0, width=0.75, data=df, ax=ax1, scale='width', inner='quartile')
    ax1.set_ylim([-0.1, 1.1])
    ax1.get_legend().remove()
    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=300)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    return fig, axs


# ========================================================================= #
# Experiment 3                                                              #
# ========================================================================= #


def plot_e03_modified_loss_xysquares(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    color_betavae: str = PINK,
    color_adavae: str = ORANGE,
    color_mse: str = LGREEN,
    color_mse_overlap: str = LBLUE2,
):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df = load_general_data(f'{os.environ["WANDB_USER"]}/CVPR-09__vae_overlap_loss')
    # select run groups
    df = df[df[K_GROUP].isin(['sweep_overlap_boxblur_specific', 'sweep_overlap_boxblur'])]
    # print common key values
    print('K_GROUP:    ', list(df[K_GROUP].unique()))
    print()
    print('K_DATASET:  ', list(df[K_DATASET].unique()))
    print('K_FRAMEWORK:', list(df[K_FRAMEWORK].unique()))
    print('K_Z_SIZE:   ', list(df[K_Z_SIZE].unique()))
    print('K_LOSS:     ', list(df[K_LOSS].unique()))
    print('K_BETA:     ', list(df[K_BETA].unique()))
    print()
    print('K_REPEAT:   ', list(df[K_REPEAT].unique()))
    print('K_STATE:    ', list(df[K_STATE].unique()))
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    orig = df
    # select runs
    df = df[df[K_STATE] == 'finished']  # TODO: update
    df = df[df[K_DATASET] == 'xysquares_minimal']
    df = df[df[K_BETA].isin([0.0001, 0.0316])]
    df = df[df[K_Z_SIZE] == 25]
    # df = df[df[K_FRAMEWORK] == 'betavae'] # 18
    # df = df[df[K_FRAMEWORK] == 'adavae_os'] # 21
    # df = df[df[K_LOSS] == 'mse']  # 20
    # df = df[df[K_LOSS] != 'mse']  # 19
    # # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # TEMP
    # df[K_MIG] = df['final_metric/mig.discrete_score.max']
    # df[K_DCI] = df['final_metric/dci.disentanglement.max']

    print('NUM', len(orig), '->', len(df))

    df = df[[K_DATASET, K_FRAMEWORK, K_LOSS, K_BETA, K_MIG_END, K_DCI_END]]
    df[K_DATASET].replace('xysquares_minimal', 'XYSquares', inplace=True)
    df[K_FRAMEWORK].replace('adavae_os', 'Ada-GVAE', inplace=True)
    df[K_FRAMEWORK].replace('betavae', 'Beta-VAE', inplace=True)
    df[K_LOSS].replace('mse_box_r31_l1.0_k3969.0', 'MSE-boxblur', inplace=True)
    df[K_LOSS].replace('mse', 'MSE', inplace=True)
    PALLETTE = {'Ada-GVAE': color_adavae, 'Beta-VAE': color_betavae, 'MSE': color_mse, 'MSE-boxblur': color_mse_overlap}

    print(df)

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    (ax0, ax1) = axs
    # PLOT: MIG
    sns.violinplot(x=K_FRAMEWORK, y=K_MIG_END, hue=K_LOSS, palette=PALLETTE, split=True, cut=0, width=0.5, data=df, ax=ax0, scale='width', inner='quartile')
    ax0.set_ylim([-0.1, 1.1])
    ax0.legend(fontsize=13)
    # ax0.legend(bbox_to_anchor=(0.425, 0.9), fontsize=13)
    sns.violinplot(x=K_FRAMEWORK, y=K_DCI_END, hue=K_LOSS, palette=PALLETTE, split=True, cut=0, width=0.5, data=df, ax=ax1, scale='width', inner='quartile')
    ax1.set_ylim([-0.1, 1.1])
    ax1.get_legend().remove()
    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=300)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #


# ========================================================================= #
# Entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':

    assert 'WANDB_USER' in os.environ, 'specify "WANDB_USER" environment variable'

    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../../code/util/gadfly.mplstyle'))

    # clear_cache()

    def main():
        plot_e01_hparam_tuning(rel_path='plots/p01e01_hparam-tuning', show=True)                      # was: exp_hparams-exp
        plot_e02_incr_overlap_xysquares(rel_path='plots/p01e02_incr-overlap-xysquares', show=True)    # was: exp_incr-overlap
        plot_e02_incr_overlap_xysquares(rel_path='plots/p01e02_incr-overlap-xysquares_mig', show=True, include=('mig',), figsize=(6.5, 4))    # was: exp_incr-overlap
        plot_e02_incr_overlap_xysquares(rel_path='plots/p01e02_incr-overlap-xysquares_dci', show=True, include=('dci',), figsize=(6.5, 4))    # was: exp_incr-overlap
        plot_e03_modified_loss_xysquares(rel_path='plots/p01e03_modified-loss-xysquares', show=True)  # was: exp_overlap-loss

    main()


# ========================================================================= #
# DONE                                                                      #
# ========================================================================= #
