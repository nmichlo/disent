import os
from typing import List
from typing import Optional

import pandas as pd
import seaborn as sns
import wandb
from cachier import cachier as _cachier
from matplotlib import pyplot as plt
from tqdm import tqdm

import research.util as H
from disent.util.function import wrapped_partial


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


cachier = wrapped_partial(_cachier, cache_dir='./cache')
DF = pd.DataFrame


def clear_cache():
    load_runs.clear_cache()


# ========================================================================= #
# Load WANDB Data                                                           #
# ========================================================================= #


@cachier()
def load_runs(project: str) -> pd.DataFrame:
    api = wandb.Api()

    runs = api.runs(project)

    info_list, summary_list, config_list, name_list = [], [], [], []
    for run in tqdm(runs, desc=f'loading: {project}'):
        info_list.append({
            'id': run.id,
            'name': run.name,
            'state': run.state,
            'storage_id': run.storage_id,
            'url': run.url,
        })
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith('_')})
        name_list.append(run.name)

    return pd.DataFrame({
        "info": info_list,
        "summary": summary_list,
        "config": config_list,
        "name": name_list
    })


def load_expanded_runs(project: str) -> pd.DataFrame:
    # load the data
    df_runs: DF = load_runs(project)
    # expand the dictionaries
    df_info: DF = df_runs['info'].apply(pd.Series)
    df_summary: DF = df_runs['summary'].apply(pd.Series)
    df_config: DF = df_runs['config'].apply(pd.Series)
    # merge the data
    df: DF = df_config.join(df_summary).join(df_info)
    assert len(df.columns) == len(df_info.columns) + len(df_summary.columns) + len(df_config.columns)
    # done!
    return df


def drop_unhashable(df: pd.DataFrame, inplace: bool = False) -> (pd.DataFrame, List[str]):
    dropped = []
    for col in df.columns:
        try:
            df[col].unique()
        except:
            dropped.append(col)
            df = df.drop(col, inplace=inplace, axis=1)
    return df, dropped


def drop_non_diverse_cols(df: pd.DataFrame, inplace: bool = False) -> (pd.DataFrame, List[str]):
    dropped = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            dropped.append(col)
            df = df.drop(col, inplace=inplace, axis=1)
    return df, dropped


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
K_MIG       = 'MIG Score'
K_DCI       = 'DCI Score'


def load_general_data(project: str):
    # load data
    df = load_expanded_runs(project)
    # filter out unneeded columns
    df, dropped_hash = drop_unhashable(df)
    df, dropped_diverse = drop_non_diverse_cols(df)
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
        'final_metric/mig.discrete_score.max':  K_MIG,
        'final_metric/dci.disentanglement.max': K_DCI,
    })


# ========================================================================= #
# Plot Experiments                                                          #
# ========================================================================= #

PINK = '#FE375F'
PURPLE = '#5E5BE5'
BLUE = '#0A83FE'
LBLUE = '#63D2FE'
ORANGE = '#FE9F0A'
GREEN = '#2FD157'


def plot_incr_overlap_exp(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    reg_order: int = 4,
    color_betavae: str = PINK,
    color_adavae: str = ORANGE,
    titles: bool = False,
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
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    (ax0, ax1) = axs
    # PLOT: MIG
    sns.regplot(ax=ax0, x=K_SPACING, y=K_MIG, data=data_adavae,  seed=777, order=reg_order, robust=False, color=color_adavae,  marker='o')
    sns.regplot(ax=ax0, x=K_SPACING, y=K_MIG, data=data_betavae, seed=777, order=reg_order, robust=False, color=color_betavae, marker='x', line_kws=dict(linestyle='dashed'))
    ax0.legend(labels=["Ada-GVAE", "Beta-VAE"], fontsize=14)
    ax0.set_ylim([-0.1, 1.1])
    ax0.set_xlim([0.8, 8.2])
    if titles: ax0.set_title('Framework Mig Scores')
    # PLOT: DCI
    sns.regplot(ax=ax1, x=K_SPACING, y=K_DCI, data=data_adavae,  seed=777, order=reg_order, robust=False, color=color_adavae,  marker='o')
    sns.regplot(ax=ax1, x=K_SPACING, y=K_DCI, data=data_betavae, seed=777, order=reg_order, robust=False, color=color_betavae, marker='x', line_kws=dict(linestyle='dashed'))
    ax1.legend(labels=["Ada-GVAE", "Beta-VAE"], fontsize=14)
    ax1.set_ylim([-0.1, 1.1])
    ax1.set_xlim([0.8, 8.2])
    if titles: ax1.set_title('Framework DCI Scores')
    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    return fig, axs





def plot_hparams_exp(
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

    df = df[[K_DATASET, K_FRAMEWORK, K_MIG, K_DCI]]
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
    sns.violinplot(x=K_DATASET, y=K_MIG, hue=K_FRAMEWORK, palette=PALLETTE, split=True, cut=0, width=0.75, data=df, ax=ax0, scale='width', inner='quartile')
    ax0.set_ylim([-0.1, 1.1])
    ax0.legend(bbox_to_anchor=(0.425, 0.9), fontsize=13)
    sns.violinplot(x=K_DATASET, y=K_DCI, hue=K_FRAMEWORK, palette=PALLETTE, split=True, cut=0, width=0.75, data=df, ax=ax1, scale='width', inner='quartile')
    ax1.set_ylim([-0.1, 1.1])
    ax1.get_legend().remove()
    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=300)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    return fig, axs



def plot_overlap_loss_exp(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    color_betavae: str = PINK,
    color_adavae: str = ORANGE,
    color_mse: str = '#9FD911',
    color_mse_overlap: str = '#36CFC8',
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

    df = df[[K_DATASET, K_FRAMEWORK, K_LOSS, K_BETA, K_MIG, K_DCI]]
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
    sns.violinplot(x=K_FRAMEWORK, y=K_MIG, hue=K_LOSS, palette=PALLETTE, split=True, cut=0, width=0.5, data=df, ax=ax0, scale='width', inner='quartile')
    ax0.set_ylim([-0.1, 1.1])
    ax0.legend(fontsize=13)
    # ax0.legend(bbox_to_anchor=(0.425, 0.9), fontsize=13)
    sns.violinplot(x=K_FRAMEWORK, y=K_DCI, hue=K_LOSS, palette=PALLETTE, split=True, cut=0, width=0.5, data=df, ax=ax1, scale='width', inner='quartile')
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
    plt.style.use(os.path.join(os.path.dirname(__file__), '../gadfly.mplstyle'))

    # clear_cache()

    def main():
        # plot_hparams_exp(rel_path='plots/exp_hparams-exp', show=True)
        plot_overlap_loss_exp(rel_path='plots/exp_overlap-loss', show=True)
        # plot_incr_overlap_exp(rel_path='plots/exp_incr-overlap', show=True)

    main()


# ========================================================================= #
# DONE                                                                      #
# ========================================================================= #
