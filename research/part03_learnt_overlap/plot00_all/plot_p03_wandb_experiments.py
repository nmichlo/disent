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

import logging
import os
from typing import Optional
from typing import Sequence

import pandas as pd
import seaborn as sns
from cachier import cachier as _cachier
from matplotlib import pyplot as plt

import research.code.util as H
from disent.util.function import wrapped_partial
from disent.util.profiling import Timer
from research.code.util._wandb_plots import drop_non_unique_cols
from research.code.util._wandb_plots import drop_unhashable_cols
from research.code.util._wandb_plots import load_runs


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


# cachier instance
_CACHIER: _cachier = wrapped_partial(_cachier, cache_dir=os.path.join(os.path.dirname(__file__), 'plots/.cache'))


def clear_plots_cache(clear_wandb=False, clear_processed=True):
    from research.code.util._wandb_plots import clear_runs_cache
    if clear_wandb:
        clear_runs_cache()
    if clear_processed:
        load_general_data.clear_cache()


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

@_CACHIER()
def load_general_data(
    project: str,
    include_history: bool = False,
    keep_cols: Sequence[str] = None,
    drop_unhashable: bool = False,
    drop_non_unique: bool = False,
):
    # keep columns
    if keep_cols is None:
        keep_cols = []
    keep_cols = list(keep_cols)
    if include_history:
        keep_cols = ['history'] + keep_cols
    # load data
    df = load_runs(project, include_history=include_history)
    # process data
    with Timer('processing data'):
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
            # triplet experiments
            'framework/cfg/triplet_margin_max': K_TRIPLET_MARGIN,
            'framework/cfg/triplet_scale':      K_TRIPLET_SCALE,
            'framework/cfg/triplet_p':          K_TRIPLET_P,
            'framework/cfg/detach_decoder':     K_DETACH,
            'framework/cfg/triplet_loss':       K_TRIPLET_MODE,
        })
        # filter out unneeded columns
        if drop_unhashable:
            df, dropped_hash = drop_unhashable_cols(df, skip=keep_cols)
        if drop_non_unique:
            df, dropped_diverse = drop_non_unique_cols(df, skip=keep_cols)
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
# Experiment 3                                                              #
# ========================================================================= #


def plot_e03_different_gt_representations(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    color_entangled_data: str = LGREEN,
    color_disentangled_data: str = LBLUE2,
    metrics: Sequence[str] = (K_MIG, K_DCI, K_RCORR_GT_F, K_RCORR_DATA_F, K_AXIS, K_LINE),
):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df = load_general_data(f'{os.environ["WANDB_USER"]}/MSC-p03e03_different-gt-representations', keep_cols=(K_GROUP, K_Z_SIZE))
    # select run groups
    #     +DUMMY.repeat=1,2,3 \
    #     settings.framework.beta=0.001,0.00316,0.01,0.0316 \
    #     framework=betavae,adavae_os \
    #     settings.model.z_size=9 \
    #     dataset=xyobject,xyobject_shaded \
    df = df[df[K_GROUP].isin(['sweep_different-gt-repr_basic-vaes'])]
    df = df.sort_values([K_DATASET, K_FRAMEWORK, K_BETA, K_REPEAT])
    # print common key values
    print('K_GROUP:    ', list(df[K_GROUP].unique()))
    print('K_DATASET:  ', list(df[K_DATASET].unique()))
    print('K_FRAMEWORK:', list(df[K_FRAMEWORK].unique()))
    print('K_BETA:     ', list(df[K_BETA].unique()))
    print('K_Z_SIZE:   ', list(df[K_Z_SIZE].unique()))
    print('K_REPEAT:   ', list(df[K_REPEAT].unique()))
    print('K_STATE:    ', list(df[K_STATE].unique()))
    # rename more stuff
    df = rename_entries(df)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    orig = df
    # select runs
    # df = df[df[K_STATE] == 'finished']
    # [1.0, 0.316, 0.1, 0.0316, 0.01, 0.00316, 0.001, 0.000316]
    # df = df[(0.000316 < df[K_BETA]) & (df[K_BETA] < 1.0)]
    print('NUM', len(orig), '->', len(df))
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    df[K_DATASET].replace('xysquares_minimal', 'XYSquares', inplace=True)
    df[K_DATASET].replace('smallnorb', 'NORB', inplace=True)
    df[K_DATASET].replace('cars3d', 'Cars3D', inplace=True)
    df[K_DATASET].replace('3dshapes', 'Shapes3D', inplace=True)
    df[K_DATASET].replace('dsprites', 'dSprites', inplace=True)
    df[K_DATASET].replace('xyobject', 'XYObject', inplace=True)
    df[K_DATASET].replace('xyobject_shaded', 'XYObject (Shades)', inplace=True)
    df[K_FRAMEWORK].replace('adavae_os', 'Ada-GVAE', inplace=True)
    df[K_FRAMEWORK].replace('betavae', 'Beta-VAE', inplace=True)

    PALLETTE = {
        'XYObject': color_entangled_data,
        'XYObject (Shades)': color_disentangled_data,
    }

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    fig, axs = plt.subplots(2, len(metrics) // 2, figsize=(len(metrics)//2*3.75, 2*2.7))
    axs = axs.flatten()
    # PLOT
    for i, (key, ax) in enumerate(zip(metrics, axs)):
        assert key in df.columns, f'{repr(key)} not in {sorted(df.columns)}'
        sns.violinplot(data=df, ax=ax, x=K_FRAMEWORK, y=key, hue=K_DATASET, palette=PALLETTE, split=True, cut=0, width=0.75, scale='width', inner='quartile')
        ax.set_ylim([-0.1, 1.1])
        ax.set_ylim([0, None])
        if i == 0:
            ax.legend(bbox_to_anchor=(0, 1.0), fontsize=12, loc='upper left', labelspacing=0.1)
            ax.set_xlabel(None)
            # ax.set_ylabel('Minimum Recon. Loss')
        else:
            if ax.get_legend():
                ax.get_legend().remove()
            ax.set_xlabel(None)
            # ax.set_ylabel(None)
    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=300)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    return fig, axs


# ========================================================================= #
# Entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':

    assert 'WANDB_USER' in os.environ, 'specify "WANDB_USER" environment variable'

    logging.basicConfig(level=logging.INFO)

    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../../code/util/gadfly.mplstyle'))

    # clear_plots_cache(clear_wandb=True, clear_processed=True)
    # clear_plots_cache(clear_wandb=False, clear_processed=True)

    def main():
        plot_e03_different_gt_representations(rel_path='plots/p03e03_different-gt-representations', show=True)

    main()


# ========================================================================= #
# DONE                                                                      #
# ========================================================================= #
