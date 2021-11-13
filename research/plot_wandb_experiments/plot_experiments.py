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
K_FRAMEWORK = 'Framework'
K_SPACING   = 'Spacing'
K_BETA      = 'Beta'
K_REPEAT    = 'Repeat'
K_STATE     = 'State'
K_MIG       = 'MIG Score'
K_DCI       = 'DCI Score'


def print_common(df: pd.DataFrame):
    # print common key values
    print('K_GROUP:    ', list(df[K_GROUP].unique()))
    print('K_FRAMEWORK:', list(df[K_FRAMEWORK].unique()))
    print('K_SPACING:  ', list(df[K_SPACING].unique()))
    print('K_BETA:     ', list(df[K_BETA].unique()))
    print('K_REPEAT:   ', list(df[K_REPEAT].unique()))
    print('K_STATE:    ', list(df[K_STATE].unique()))


def load_general_data(project: str):
    # load data
    df = load_expanded_runs(project)
    # filter out unneeded columns
    df, dropped_hash = drop_unhashable(df)
    df, dropped_diverse = drop_non_diverse_cols(df)
    # rename columns
    return df.rename(columns={
        'EXTRA/tags':                           K_GROUP,
        'framework/name':                       K_FRAMEWORK,
        'dataset/data/grid_spacing':            K_SPACING,
        'settings/framework/beta':              K_BETA,
        'DUMMY/repeat':                         K_REPEAT,
        'state':                                K_STATE,
        'final_metric/mig.discrete_score.max':  K_MIG,
        'final_metric/dci.disentanglement.max': K_DCI,
    })


# ========================================================================= #
# Plot Experiments                                                          #
# ========================================================================= #


def plot_incr_overlap_exp(rel_path: Optional[str] = None, save: bool = True, show: bool = True):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df = load_general_data(f'{os.environ["WANDB_USER"]}/CVPR-01__incr_overlap')
    # select run groups
    df = df[df[K_GROUP] == 'sweep_xy_squares_overlap']
    # print common key values
    print_common(df)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    BETA = 0.00316   # if grid_spacing <  6
    BETA = 0.001     # if grid_spacing >= 6

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    orig = df
    # select runs
    # df = df[df[K_STATE] == 'finished']
    # df = df[df[K_REPEAT].isin([1, 2, 3])]

    adavae_selector    = (df[K_FRAMEWORK] == 'adavae_os') & (df[K_BETA] == 0.001)
    data_adavae  = df[adavae_selector]

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
    sns.regplot(ax=ax0, x=K_SPACING, y=K_MIG, data=data_betavae,  order=1, robust=False, color='pink')
    sns.regplot(ax=ax0, x=K_SPACING, y=K_MIG, data=data_adavae,  order=1, robust=False, color='lightblue')
    ax0.legend(labels=["Beta-VAE", "Ada-GVAE"])
    ax0.set_ylim([-0.1, 1.1])
    ax0.set_xlim([0.8, 8.2])
    ax0.set_title('Framework Mig Scores')
    # PLOT: DCI
    sns.regplot(ax=ax1, x=K_SPACING, y=K_DCI, data=data_betavae,  order=1, robust=False, color='pink')
    sns.regplot(ax=ax1, x=K_SPACING, y=K_DCI, data=data_adavae,  order=1, robust=False, color='lightblue')
    ax1.legend(labels=["Beta-VAE", "Ada-GVAE"])
    ax1.set_ylim([-0.1, 1.1])
    ax1.set_xlim([0.8, 8.2])
    ax1.set_title('Framework DCI Scores')
    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    return fig, axs


# ========================================================================= #
# Entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':

    assert 'WANDB_USER' in os.environ, 'specify "WANDB_USER" environment variable'

    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../gadfly.mplstyle'))

    # clear_cache()

    def main():
        plot_incr_overlap_exp(rel_path='plots/exp_incr-overlap', show=True)

    main()


# ========================================================================= #
# DONE                                                                      #
# ========================================================================= #
