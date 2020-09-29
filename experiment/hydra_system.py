import os
import logging

from omegaconf import DictConfig, OmegaConf
import hydra
import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.loggers import WandbLogger, CometLogger

from disent.dataset.groundtruth import GroundTruthDataset
from disent.metrics import compute_dci, compute_factor_vae
from disent.model import GaussianAutoEncoder
from disent.transform.groundtruth import GroundTruthDatasetBatchAugment
from disent.util import make_box_str

from experiment.util.callbacks import VaeDisentanglementLoggingCallback, VaeLatentCycleLoggingCallback, LoggerProgressCallback
from experiment.util.callbacks.callbacks_vae import VaeLatentCorrelationLoggingCallback
from experiment.util.hydra_utils import instantiate_recursive

log = logging.getLogger(__name__)


# ========================================================================= #
# DATASET                                                                   #
# ========================================================================= #


class HydraDataModule(pl.LightningDataModule):

    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.hparams = hparams
        # transform: prepares data from datasets | augment: augments transformed data for inputs
        self._transform = instantiate_recursive(self.hparams.dataset.transform)
        self._augment = instantiate_recursive(self.hparams.augment.transform)
        # batch_augment: augments transformed data for inputs, should be applied across a batch, same as self.augment
        self.batch_augment = GroundTruthDatasetBatchAugment(transform=self._augment) if (self._augment is not None) else None
        # datasets
        self.dataset_train: GroundTruthDataset = None
        self.dataset_train_aug: GroundTruthDataset = None

    def prepare_data(self) -> None:
        # *NB* Do not set model parameters here.
        # - Instantiate data once to download and prepare if needed.
        # - trainer.prepare_data_per_node affects this functions behavior per node.
        data = self.hparams.dataset.data.copy()
        if 'in_memory' in data:
            del data['in_memory']
        hydra.utils.instantiate(data)

    def setup(self, stage=None) -> None:
        # ground truth data
        data = hydra.utils.instantiate(self.hparams.dataset.data)
        # Wrap the data for the framework some datasets need triplets, pairs, etc.
        # Augmentation is done inside the frameworks so that it can be done on the GPU, otherwise things are very slow.
        self.dataset_train = hydra.utils.instantiate(self.hparams.framework.data_wrapper, ground_truth_data=data, transform=self._transform, augment=None)
        self.dataset_train_aug = hydra.utils.instantiate(self.hparams.framework.data_wrapper, ground_truth_data=data, transform=self._transform, augment=self._augment)
        assert isinstance(self.dataset_train, GroundTruthDataset)
        assert isinstance(self.dataset_train_aug, GroundTruthDataset)

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


def hydra_make_logger(cfg):
    loggers = []
    if ('wandb' in cfg.logging) and cfg.logging.wandb.get('enabled', True):
        loggers.append(WandbLogger(
            offline=cfg.logging.wandb.get('offline', False),
            entity=cfg.logging.wandb.get('entity', None),  # cometml: workspace
            project=cfg.logging.wandb.project,             # cometml: project_name
            name=cfg.logging.wandb.name,                   # cometml: experiment_name
            group=cfg.logging.wandb.get('group', None),    # experiment group
            tags=cfg.logging.wandb.get('tags', None),      # experiment tags
            save_dir=hydra.utils.to_absolute_path(cfg.logging.logs_dir),  # relative to hydra's original cwd
        ))
    if ('cometml' in cfg.logging) and cfg.logging.cometml.get('enabled', True):
        loggers.append(CometLogger(
            offline=cfg.logging.cometml.get('offline', False),
            workspace=cfg.logging.cometml.get('workspace', None),  # wandb: entity
            project_name=cfg.logging.cometml.project,              # wandb: project
            experiment_name=cfg.logging.cometml.name,              # wandb: name
            api_key=os.environ['COMET_API_KEY'],                   # TODO: use dotenv
            save_dir=hydra.utils.to_absolute_path(cfg.logging.logs_dir),  # relative to hydra's original cwd
        ))
    return loggers if loggers else None  # lists are turned into a LoggerCollection by pl


def hydra_append_progress_callback(callbacks, cfg):
    if 'progress' in cfg.callbacks:
        callbacks.append(LoggerProgressCallback(
            interval=cfg.callbacks.progress.interval
        ))


def hydra_append_latent_cycle_logger_callback(callbacks, cfg):
    if 'latent_cycle' in cfg.callbacks:
        if cfg.logging.wandb.enabled:
            # this currently only supports WANDB logger
            callbacks.append(VaeLatentCycleLoggingCallback(
                seed=cfg.callbacks.latent_cycle.seed,
                every_n_steps=cfg.callbacks.latent_cycle.every_n_steps,
                begin_first_step=False,
            ))
        else:
            log.warning('latent_cycle callback is not being used because wandb is not enabled!')


def hydra_append_metric_callback(callbacks, cfg):
    if 'metrics' in cfg.callbacks:
        callbacks.append(VaeDisentanglementLoggingCallback(
            every_n_steps=cfg.callbacks.metrics.every_n_steps,
            begin_first_step=False,
            step_end_metrics=[
                lambda dat, fn: compute_dci(dat, fn, 1000, 500, boost_mode='sklearn'),
                lambda dat, fn: compute_factor_vae(dat, fn, num_train=1000, num_eval=500, num_variance_estimate=1000),
            ],
            train_end_metrics=[
                compute_dci,
                compute_factor_vae
            ],
        ))


def hydra_append_correlation_callback(callbacks, cfg):
    if 'correlation' in cfg.callbacks:
        callbacks.append(VaeLatentCorrelationLoggingCallback(
            repeats_per_factor=cfg.callbacks.correlation.repeats_per_factor,
            every_n_steps=cfg.callbacks.correlation.every_n_steps,
            begin_first_step=False,
        ))


# ========================================================================= #
# RUNNER                                                                    #
# ========================================================================= #


@hydra.main(config_path='config', config_name="config")
def main(cfg: DictConfig):
    # print useful info
    log.info(make_box_str(OmegaConf.to_yaml(cfg)))
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
    hydra_append_progress_callback(callbacks, cfg)
    hydra_append_latent_cycle_logger_callback(callbacks, cfg)
    hydra_append_metric_callback(callbacks, cfg)
    hydra_append_correlation_callback(callbacks, cfg)

    # DATA
    datamodule = HydraDataModule(cfg)

    # FRAMEWORK
    framework = hydra.utils.instantiate(
        cfg.framework.module,
        make_optimizer_fn=lambda params: hydra.utils.instantiate(cfg.optimizer.cls, params),
        make_model_fn=lambda: GaussianAutoEncoder(
            encoder=hydra.utils.instantiate(cfg.model.encoder),
            decoder=hydra.utils.instantiate(cfg.model.decoder)
        ),
        # apply augmentations to batch on GPU which is faster than on the dataloader
        batch_augment=datamodule.batch_augment
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
    trainer.fit(framework, datamodule=datamodule)


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
