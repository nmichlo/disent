
# ========================================================================= #
# STREAMLIT RUNNER                                                          #
# ========================================================================= #

if __name__ == '__main__':
    from experiment.util.streamlit_util import run_streamlit
    run_streamlit(__file__)

# ========================================================================= #
# IMPORTS                                                                   #
# ========================================================================= #


import logging
from collections import defaultdict
import time
import copy

import numpy as np
import kornia
import torch

from omegaconf import DictConfig
import hydra
import hydra.experimental
import streamlit as st

from disent.visualize.visualize_util import make_image_grid
from experiment.run import HydraDataModule, hydra_check_datadir


log = logging.getLogger(__name__)


# ========================================================================= #
# RUNNER                                                                    #
# ========================================================================= #


def visualise(cfg: DictConfig):
    # check data preparation
    prepare_data_per_node = cfg.trainer.get('prepare_data_per_node', True)
    hydra_check_datadir(prepare_data_per_node, cfg)

    datamodule = HydraDataModule(cfg)
    datamodule.setup()
    dataset = datamodule.dataset_train_aug

    st.title(f'Visualising: {cfg.dataset.name}')
    ST = defaultdict(st.empty)

    PADDING = 8
    SCALE = st.sidebar.slider('Image Scale', 0.1, 10.0, value=1.5)
    WAIT = st.sidebar.slider('Refresh Rate', 0.5, 10.0, value=2.0)

    while True:
        # get random images
        idx = np.random.randint(len(dataset))
        obs = dataset[idx]

        # convert augmented images to observations
        x = [(kornia.tensor_to_image(torch.clamp(obs, 0, 1))*255).astype('uint8') for obs in obs['x']]
        x_targ = [(kornia.tensor_to_image(torch.clamp(obs, 0, 1))*255).astype('uint8') for obs in obs['x_targ']]
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

    @st.cache()
    def get_config():
        with hydra.experimental.initialize(config_path='config'):
            return hydra.experimental.compose(config_name='config')

    visualise(copy.deepcopy(get_config()))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
