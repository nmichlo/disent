import logging
import os
import time

import numpy as np
import kornia

from omegaconf import DictConfig, OmegaConf
import hydra

from disent.util import make_box_str
from experiment.hydra_system import HydraDataModule, hydra_check_datadir
from experiment.img_dashboard.img_server import send_images, REFRESH_MS

log = logging.getLogger(__name__)


# ========================================================================= #
# ENTRY POINT                                                               #
# ========================================================================= #


@hydra.main(config_path='../config', config_name="config")
def main(cfg: DictConfig):
    # print useful info
    log.info(make_box_str(OmegaConf.to_yaml(cfg)))
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    # check data preparation
    prepare_data_per_node = cfg.trainer.get('prepare_data_per_node', True)
    hydra_check_datadir(prepare_data_per_node, cfg)

    datamodule = HydraDataModule(cfg)
    datamodule.setup()

    while True:
        # get random images
        idx = np.random.randint(len(datamodule.dataset))
        obs = datamodule.dataset_train[idx]
        x, x_targ = obs['x'], obs['x_targ']

        # convert augmented images to observations
        if not isinstance(x, (tuple, list)):
            x, x_targ = [x], [x_targ]
        x = [kornia.tensor_to_image(obs) for obs in x]
        x_targ = [kornia.tensor_to_image(obs) for obs in x_targ]

        # send images to server
        try:
            send_images({
                **{f'x{i}': obs for i, obs in enumerate(x)},
                **{f'x{i}_targ': obs for i, obs in enumerate(x_targ)},
            }, format='png')
        except Exception as e:
            pass

        # wait
        time.sleep(REFRESH_MS/1000)


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        log.warning('Interrupted - Exited early!')
    except:
        log.error('A critical error occurred! Exiting safely...', exc_info=True)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
