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

import os.path
import warnings
from typing import List
from typing import Sequence

import pandas as pd
import wandb
from cachier import cachier as _cachier
from tqdm import tqdm

from disent.util.function import wrapped_partial


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


# `research/code/util/.cache`
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')

# cachier instance
CACHIER: _cachier = wrapped_partial(_cachier, cache_dir=CACHE_DIR)


def clear_runs_cache():
    _load_runs_data.clear_cache()


# ========================================================================= #
# Load WANDB Data                                                           #
# ========================================================================= #


@CACHIER()
def _load_runs_data(project: str, include_history: bool = False) -> pd.DataFrame:
    api = wandb.Api()

    runs = api.runs(project)

    info_list, summary_list, config_list, name_list, history_list = [], [], [], [], []
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
        if include_history:
            history_list.append(run.history())

    data = {
        "info": info_list,
        "summary": summary_list,
        "config": config_list,
        "name": name_list
    }

    if include_history:
        data['history'] = history_list

    return pd.DataFrame(data)


def load_runs(project: str, include_history: bool = False) -> pd.DataFrame:
    # load the data
    df_runs_data: pd.DataFrame = _load_runs_data(project, include_history=include_history)
    # expand the dictionaries
    df_info: pd.DataFrame = df_runs_data['info'].apply(pd.Series)
    df_summary: pd.DataFrame = df_runs_data['summary'].apply(pd.Series)
    df_config: pd.DataFrame = df_runs_data['config'].apply(pd.Series)
    # merge the data
    df: pd.DataFrame = df_config.join(df_summary).join(df_info)
    assert len(df.columns) == len(df_info.columns) + len(df_summary.columns) + len(df_config.columns)
    # add history
    if include_history:
        assert 'history' not in df.columns
        df = df.join(df_runs_data['history'])  # history is already a series
        assert 'history' in df.columns
        assert len(df.columns) == len(df_info.columns) + len(df_summary.columns) + len(df_config.columns) + 1
    # done!
    return df


# ========================================================================= #
# Run Filtering                                                             #
# ========================================================================= #


def drop_unhashable_cols(df: pd.DataFrame, inplace: bool = False, skip: Sequence[str] = None) -> (pd.DataFrame, List[str]):
    """
    Drop all the columns of a dataframe that cannot be hashed
    -- this will remove media or other usually unnecessary content from the wandb api
    """
    # keep these items
    if skip is None:
        skip = []
    skip = set(skip)
    if skip - set(df.columns):
        warnings.warn(f'some skipped column names are not contained in the df: {skip - set(df.columns)}')
    # drop columns
    dropped = []
    for col in df.columns:
        if col in skip:
            continue
        try:
            df[col].unique()
        except:
            dropped.append(col)
            df = df.drop(col, inplace=inplace, axis=1)
    return df, dropped


def drop_non_unique_cols(df: pd.DataFrame, inplace: bool = False, skip: Sequence[str] = None) -> (pd.DataFrame, List[str]):
    """
    Drop all the columns of a dataframe where all the values are the same!
    """
    # keep these items
    if skip is None:
        skip = []
    skip = set(skip)
    if skip - set(df.columns):
        warnings.warn(f'some skipped column names are not contained in the df: {skip - set(df.columns)}')
    # drop columns
    dropped = []
    for col in df.columns:
        if col in skip:
            continue
        if len(df[col].unique()) == 1:
            dropped.append(col)
            df = df.drop(col, inplace=inplace, axis=1)
    return df, dropped


# ========================================================================= #
# Prepare Data                                                              #
# ========================================================================= #
