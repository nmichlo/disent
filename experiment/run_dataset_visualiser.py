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

import logging
import time
from collections import defaultdict

import hydra
import hydra.experimental
import numpy as np
import streamlit as st
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from disent.util.strings import make_box_str
from disent.visualize.visualize_util import make_image_grid
from experiment.exp.util import to_imgs
from experiment.run import hydra_check_datadir
from experiment.run import HydraDataModule
from experiment.util.hydra_utils import make_non_strict
from experiment.util.hydra_utils import merge_specializations
from experiment.util.streamlit_util import run_streamlit


log = logging.getLogger(__name__)


# ========================================================================= #
# GETTERS                                                                   #
# ========================================================================= #


@st.cache
def get_dataset(cfg):
    # check data preparation
    prepare_data_per_node = cfg.trainer.get('prepare_data_per_node', True)
    hydra_check_datadir(prepare_data_per_node, cfg)

    datamodule = HydraDataModule(cfg)
    datamodule.setup()
    return datamodule.dataset_train_aug  # GroundTruthDataset


@st.cache(allow_output_mutation=True)
def get_config():
    cfg = None
    @hydra.main(config_path='config', config_name="config")
    def main(config: DictConfig):
        nonlocal cfg
        cfg = make_non_strict(config)
        # hydra config does not support variables in defaults lists, we handle this manually
        cfg = merge_specializations(cfg, 'config', visualise)
        # print the config
        log.info(f'Final Config Is:\n{make_box_str(OmegaConf.to_yaml(cfg))}')
    main()
    return cfg


# ========================================================================= #
# RUNNER                                                                    #
# ========================================================================= #

def visualise(cfg: DictConfig):
    dataset = get_dataset(cfg)

    st.title(f'Visualising: {cfg.dataset.name}')
    ST = defaultdict(st.empty)

    PADDING = 8
    SCALE = st.sidebar.slider('Image Scale', 0.1, 10.0, value=1.5)
    WAIT = st.sidebar.slider('Refresh Rate', 0.25, 10.0, value=0.75)

    while True:
        # get random images
        idx = np.random.randint(len(dataset))
        obs = dataset[idx]

        # convert augmented images to observations
        x      = [to_imgs(torch.clamp(obs, 0, 1)).numpy() for obs in obs['x']]
        x_targ = [to_imgs(torch.clamp(obs, 0, 1)).numpy() for obs in obs['x_targ']]
        img = make_image_grid(x + x_targ, pad=PADDING, border=False, bg_color=255, num_cols=len(x))

        # send images to server
        width = ((PADDING + x[0].shape[1]) * len(x) - PADDING) * SCALE
        ST['x'].image(img, 'x', width=int(width))

        # wait
        time.sleep(WAIT)


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    # wrap this program with streamlit
    run_streamlit(__file__)
    # run the visualiser
    visualise(get_config())


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
