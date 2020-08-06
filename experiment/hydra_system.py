import os
import logging

from omegaconf import DictConfig
import hydra
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchvision
from pytorch_lightning.loggers import WandbLogger

from disent.dataset.single import GroundTruthDataset
from disent.metrics import compute_dci, compute_factor_vae
from disent.model import GaussianAutoEncoder
from disent.util import make_box_str

from experiment.util.callbacks import DisentanglementLoggingCallback, LatentCycleLoggingCallback, LoggerProgressCallback

log = logging.getLogger(__name__)


# ========================================================================= #
# DATASET                                                                   #
# ========================================================================= #


class HydraDataModule(pl.LightningDataModule):

    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.hparams = hparams
        self.dataset = None
        self.dataset_train = None

    def prepare_data(self) -> None:
        # *NB* Do not set model parameters here.
        # - Instantiate data once to download and prepare if needed.
        # - trainer.prepare_data_per_node affects this functions behavior per node.
        if 'in_memory' in self.hparams.dataset.cls.params:
            hydra.utils.instantiate(self.hparams.dataset.cls, in_memory=False)
        else:
            hydra.utils.instantiate(self.hparams.dataset.cls)

    def setup(self, stage=None) -> None:
        # single observations
        self.dataset = GroundTruthDataset(
            ground_truth_data=hydra.utils.instantiate(self.hparams.dataset.cls),
            transform=torchvision.transforms.Compose([
                hydra.utils.instantiate(transform_cls)
                for transform_cls in self.hparams.dataset.transforms
            ])
        )
        # augment dataset if the framework requires
        self.dataset_train = self.dataset
        if 'augment_dataset' in self.hparams.framework:
            self.dataset_train = hydra.utils.instantiate(self.hparams.framework.augment_dataset.cls, self.dataset)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Training Dataset:
    #     The sample of data used to fit the model.
    # Validation Dataset:
    #     Data used to provide an unbiased evaluation of a model fit on the
    #     training dataset while tuning model hyperparameters. The
    #     evaluation becomes more biased as skill on the validation
    #     dataset is incorporated into the model configuration.
    # Test Dataset:
    #     The sample of data used to provide an unbiased evaluation of a
    #     final model fit on the training dataset.
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def train_dataloader(self):
        """Training Dataset: Sample of data used to fit the model"""
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.hparams.dataset.batch_size,
            num_workers=self.hparams.dataset.num_workers,
            shuffle=True
        )


# ========================================================================= #
# HYDRA CONFIG HELPERS                                                      #
# ========================================================================= #


def hydra_check_cuda(cuda):
    if not torch.cuda.is_available():
        if cuda:
            log.error('trainer.cuda=True but CUDA is not available on this machine!')
            exit()
        else:
            log.warning('CUDA is not available on this machine!')
    else:
        if not cuda:
            log.warning('CUDA is available but is not being used!')

def hydra_check_datadir(prepare_data_per_node, cfg):
    if not os.path.isabs(cfg.dataset.data_dir):
        log.warning(
            f'A relative path was specified for dataset.data_dir={repr(cfg.dataset.data_dir)}.'
            f' This is probably an error! Using relative paths can have unintended consequences'
            f' and performance drawbacks if the current working directory is on a shared/network drive.'
            f' Hydra config also uses a new working directory for each run of the program, meaning'
            f' data will be repeatedly downloaded.'
        )
        if prepare_data_per_node:
            log.error(
                f'trainer.prepare_data_per_node={repr(prepare_data_per_node)} but  dataset.data_dir='
                f'{repr(cfg.dataset.data_dir)} is a relative path which may be an error! Try specifying an'
                f' absolute path that is guaranteed to be unique from each node, eg. dataset.data_dir=/tmp/dataset'
            )
        raise RuntimeError('dataset.data_dir={repr(cfg.dataset.data_dir)} is a relative path!')

def hydra_append_metric_callback(callbacks, cfg):
    if cfg.callbacks.metrics.enabled:
        callbacks.append(DisentanglementLoggingCallback(
            every_n_epochs=cfg.callbacks.metrics.every_n_epochs,
            begin_first_epoch=False,
            epoch_end_metrics=[
                lambda dat, fn: compute_dci(dat, fn, 1000, 500, boost_mode='sklearn'),
                lambda dat, fn: compute_factor_vae(dat, fn, num_train=1000, num_eval=500, num_variance_estimate=1000),
            ],
            train_end_metrics=[
                compute_dci,
                compute_factor_vae
            ],
        ))

def hydra_make_logger(cfg):
    if cfg.logging.wandb.enabled:
        log.info(f'wandb log directory: {os.path.abspath("wandb")}')
        # TODO: this should be moved into configs, instantiated from a class & target
        return WandbLogger(
            offline=cfg.logging.wandb.get('offline', False),
            name=cfg.logging.wandb.name,
            project=cfg.logging.wandb.project,
            group=cfg.logging.wandb.get('group', None),
            tags=cfg.logging.wandb.get('tags', None),
            entity=cfg.logging.get('entity', None),
            save_dir=hydra.utils.to_absolute_path(cfg.logging.logs_dir),  # relative to hydra's original cwd
        )
    return None

def hydra_append_progress_callback(callbacks, cfg):
    if cfg.callbacks.progress.enabled:
        callbacks.append(LoggerProgressCallback(
            time_step=cfg.callbacks.progress.step_time
        ))

def hydra_append_latent_cycle_logger_callback(callbacks, cfg):
    # this currently only supports WANDB logger
    if cfg.logging.wandb.enabled:
        if cfg.callbacks.latent_cycle.enabled:
            # Log the latent cycle visualisations to wandb
            callbacks.append(LatentCycleLoggingCallback(
                seed=cfg.callbacks.latent_cycle.seed,
                every_n_epochs=cfg.callbacks.latent_cycle.every_n_epochs,
                begin_first_epoch=False,
            ))


# ========================================================================= #
# RUNNER                                                                    #
# ========================================================================= #


@hydra.main(config_path='hydra_config', config_name="config")
def main(cfg: DictConfig):
    # print useful info
    log.info(make_box_str(cfg.pretty()))
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    # check CUDA setting
    cuda = cfg.trainer.get('cuda', torch.cuda.is_available())
    hydra_check_cuda(cuda)

    # check data preparation
    prepare_data_per_node = cfg.trainer.get('prepare_data_per_node', True)
    hydra_check_datadir(prepare_data_per_node, cfg)

    # create trainer loggers & callbacks
    logger = hydra_make_logger(cfg)

    # TRAINER CALLBACKS
    callbacks = []
    hydra_append_latent_cycle_logger_callback(callbacks, cfg)
    hydra_append_metric_callback(callbacks, cfg)
    hydra_append_progress_callback(callbacks, cfg)
    
    # FRAMEWORK
    framework = hydra.utils.instantiate(
        cfg.framework.cls,
        make_optimizer_fn=lambda params: hydra.utils.instantiate(cfg.optimizer.cls, params),
        make_model_fn=lambda: GaussianAutoEncoder(
            encoder=hydra.utils.instantiate(cfg.model.encoder.cls),
            decoder=hydra.utils.instantiate(cfg.model.decoder.cls)
        ),
    )

    # LOG ALL HYPER-PARAMETERS
    framework.hparams = cfg

    # TRAIN
    trainer = pl.Trainer(
        row_log_interval=cfg.logging.get('log_interval', 50),
        log_save_interval=cfg.logging.get('save_log_interval', 100),
        logger=logger,
        callbacks=callbacks,
        gpus=1 if cuda else 0,
        max_epochs=cfg.trainer.get('epochs', 100),
        max_steps=cfg.trainer.get('steps', None),
        prepare_data_per_node=prepare_data_per_node,
        progress_bar_refresh_rate=0,  # ptl 0.9
    )
    trainer.fit(framework, datamodule=HydraDataModule(cfg))


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #

if __name__ == '__main__':
    try:
        main()
    except:
        log.error('A critical error occurred! Exiting safely...', exc_info=True)

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
