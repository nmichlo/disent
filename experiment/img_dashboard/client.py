import logging
import os
import time

import numpy as np
import kornia

from omegaconf import DictConfig
import hydra

from disent.util import make_box_str
from experiment.hydra_system import HydraDataModule, hydra_check_datadir
from experiment.img_dashboard.server import send_images

log = logging.getLogger(__name__)


@hydra.main(config_path='../config', config_name="config")
def main(cfg: DictConfig):
    # print useful info
    log.info(make_box_str(cfg.pretty()))
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
        raw_obs, aug_obs = datamodule.data[idx], datamodule.dataset_train[idx]

        # convert augmented images to observations
        if not isinstance(aug_obs, (tuple, list)):
            aug_obs = [aug_obs]
        aug_obs = [kornia.tensor_to_image(obs) for obs in aug_obs]

        # send images to server
        try:
            send_images({
                'ground_truth': raw_obs,
                **{
                    f'obs{i}': obs
                    for i, obs in enumerate(aug_obs)
                }
            }, format='png')
        except Exception as e:
            pass

        # wait
        time.sleep(1)


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