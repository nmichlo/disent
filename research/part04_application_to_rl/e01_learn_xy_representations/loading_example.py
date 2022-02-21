import os.path
from pathlib import Path
from pprint import pprint

import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rich.console import Console
from rich.table import Table
from natsort import natsorted


# same as part01_data_overlap/plot03_wandb/plot_all_experiments.py
PINK = '#FE375F'     # usually: Beta-VAE
PURPLE = '#5E5BE5'   # maybe:   TVAE
BLUE = '#1A93FE'     # maybe:   Ada-TVAE
ORANGE = '#FE9F0A'   # usually: Ada-VAE


if __name__ == '__main__':

    # matplotlib style
    plt.style.use(Path(__file__).parents[2].joinpath('code/util/gadfly.mplstyle'))

    # load data storage
    tables = []
    df = pd.DataFrame(columns=['exp', 'name', 'i', 'rcorr', 'axis', 'linear'])

    # iterate through all the experiment roots
    for root in [
        # Path(__file__).parent.joinpath('exp/00001_xy8_1.5_10000'),
        Path(__file__).parent.joinpath('exp/00002_xy8_1.25_10000'),
        # Path(__file__).parent.joinpath('exp/00003_xy8_1.5_5000'),
        # Path(__file__).parent.joinpath('exp/00004_xy8_1.25_5000'),
    ]:
        # make table
        table = Table('run', title=root.name)
        table.add_column('rcorr_ground_latent', max_width=7, justify='center')
        table.add_column('axis_ratio',          max_width=7, justify='center')
        table.add_column('linear_ratio',        max_width=7, justify='center')
        # get paths
        paths = [str(s) for s in Path(root).glob('**/rl_data.npz')]
        paths = natsorted(paths, key=lambda s: ('_'.join(Path(s).parent.name.split('_')[2:]), s))  # strip: eg. 0x0_xy8 from names when sorting
        # iterate through all the runs in an experiment
        for i, path in enumerate(paths):
            print(f'LOADING: {path}')
            data = np.load(path, allow_pickle=True)['data'].tolist()
            # pprint({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in data.items()}, width=200, sort_dicts=False, compact=True, depth=2)
            # get stats
            name = Path(path).parent.name
            rcorr = data["metrics"]["distances.rcorr_ground_latent.global"]
            axisr = data["metrics"]["linearity.axis_ratio.var"]
            linea = data["metrics"]["linearity.linear_ratio.var"]
            # append stats
            table.add_row(name, f'{rcorr:5.3f}', f'{axisr:5.3f}', f'{linea:5.3f}')
            df = df.append({'exp': root.name, 'name': name, 'i': i, 'rcorr': rcorr, 'axis': axisr, 'linear': linea}, ignore_index=True)
        # store
        tables.append(table)

    # make table
    console = Console(width=200)
    console.print(*tables, new_line_start=True)

    # filter
    df = df[~df['name'].isin([
        '0x1_xy8_adavae',           # ok, but adavae_os is the original!
        '0x7_xy8_triplet_soft_B',   # bad, sampling is not good
        '0x5_xy8_triplet_soft_A1',  # ok, but A2 is a stronger case!
        '0x8_xy8_triplet_soft_C',   # ok, but manhat (A) sampling is better
        '0x9_xy8_adatvae_soft_A1',  # ok, but A2 is a stronger case!
        '0x11_xy8_adatvae_soft_B',  # bad, sampling is not good
        '0x12_xy8_adatvae_soft_C',  # ok, but manhat (A) sampling is better
        # '0x4_xy8_adatvae_soft_A',
        # '0x10_xy8_adatvae_soft_A2',
        # '0x2_xy8_adavae_os',
        # '0x0_xy8_betavae',
        # '0x3_xy8_triplet_soft_A',
        # '0x6_xy8_triplet_soft_A2',
    ])]
    df = df[~df['exp'].isin([
        '00001_xy8_1.5_10000',  # ada not strong enough
        '00003_xy8_1.5_5000',   # ada not strong enough
        '00004_xy8_1.25_5000',  # might as well use longer runs
    ])]

    # make plots
    # for col in ['rcorr', 'axis', 'linear']:
    #     ax = sns.lineplot('name', col, hue='exp', data=df)
    #     plt.xticks(rotation=90)
    #     plt.tight_layout()
    #     plt.ylim([-0.05, 1.05])
    #     plt.show()

    # rename the runs
    df.loc[df['name'] == '0x4_xy8_adatvae_soft_A',   'name'] = 'Ada-TVAE'
    df.loc[df['name'] == '0x10_xy8_adatvae_soft_A2', 'name'] = 'Ada-TVAE (80%)'
    df.loc[df['name'] == '0x2_xy8_adavae_os',        'name'] = 'Ada-GVAE'
    df.loc[df['name'] == '0x0_xy8_betavae',          'name'] = 'Beta-VAE'
    df.loc[df['name'] == '0x3_xy8_triplet_soft_A',   'name'] = 'TVAE'
    df.loc[df['name'] == '0x6_xy8_triplet_soft_A2',  'name'] = 'TVAE (80%)'

    # make new df
    df1 = df.rename(columns={'rcorr': 'score'})
    df1['Metric'] = 'Ground-Latent Correlation'
    df2 = df.rename(columns={'axis': 'score'})
    df2['Metric'] = 'Axis Ratio'
    df3 = df.rename(columns={'linear': 'score'})
    df3['Metric'] = 'Linear Ratio'
    df_all = pd.concat([df1, df2, df3], keys=['name', 'metric', 'score'])

    # make save dir
    save_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(save_dir, exist_ok=True)

    # print everything!
    sns.barplot('name', 'score', hue='Metric',  data=df_all)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scores.png'))
    plt.show()
