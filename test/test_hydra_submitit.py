import logging
import os
import time
import hydra


# ========================================================================= #
# test_hydra_submitit                                                       #
# ========================================================================= #

log = logging.getLogger(__name__)

@hydra.main(config_name="config")
def my_app(cfg):
    log.info(f"Process ID {os.getpid()} executing task {cfg.task} ...")
    time.sleep(10)


if __name__ == "__main__":
    my_app()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
