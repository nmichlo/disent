import logging
from collections import defaultdict
import time

import numpy as np
import kornia
import torch

from omegaconf import DictConfig
import hydra
import hydra.experimental
import streamlit as st

from disent.visualize.visualize_util import make_image_grid
from experiment.util.streamlit_util import run_streamlit
from experiment.run import HydraDataModule, hydra_check_datadir


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
        cfg = config
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
    # wrap this program with streamlit
    run_streamlit(__file__)
    # run the visualiser
    visualise(get_config())


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
