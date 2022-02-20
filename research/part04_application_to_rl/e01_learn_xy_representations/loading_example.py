from pathlib import Path

import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rich.console import Console
from rich.table import Table
from natsort import natsorted


if __name__ == '__main__':

    table = Table(show_header=True, header_style="bold magenta")

    tables = []
    df = pd.DataFrame(columns=['exp', 'name', 'i', 'rcorr', 'axis', 'linear'])

    # iterate through all the experiment roots
    for root in [
        Path(__file__).parent.joinpath('exp/00001_xy8_1.5_10000'),
        Path(__file__).parent.joinpath('exp/00002_xy8_1.25_10000'),
        Path(__file__).parent.joinpath('exp/00003_xy8_1.5_5000'),
        Path(__file__).parent.joinpath('exp/00004_xy8_1.25_5000'),
    ]:
        # make table
        table = Table('run', title=root.name)
        table.add_column('rcorr_ground_latent', max_width=7, justify='center')
        table.add_column('axis_ratio',          max_width=7, justify='center')
        table.add_column('linear_ratio',        max_width=7, justify='center')
        # iterate through all the runs in an experiment
        for i, path in enumerate(natsorted(str(s) for s in Path(root).glob('**/rl_data.npz'))):
            print(f'LOADING: {path}')
            data = np.load(path, allow_pickle=True)['data'].tolist()
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

    # make plots
    for col in ['rcorr', 'axis', 'linear']:
        ax = sns.lineplot('name', col, hue='exp', data=df)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
