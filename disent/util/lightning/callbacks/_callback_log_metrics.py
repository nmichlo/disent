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

import logging
import warnings
from typing import Optional
from typing import Sequence

import pytorch_lightning as pl

from disent import registry as R
from disent.dataset.data import GroundTruthData
from disent.util.lightning.callbacks._callbacks_base import BaseCallbackPeriodic
from disent.util.lightning.callbacks._helper import _get_dataset_and_ae_like
from disent.util.lightning.logger_util import log_metrics
from disent.util.lightning.logger_util import wb_log_reduced_summaries
from disent.util.profiling import Timer
from disent.util.strings import colors as c


log = logging.getLogger(__name__)


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def _normalized_numeric_metrics(items: dict):
    results = {}
    for k, v in items.items():
        if isinstance(v, (float, int)):
            results[k] = v
        else:
            try:
                results[k] = float(v)
            except:
                log.warning(f'SKIPPED: metric with key: {repr(k)}, result has invalid type: {type(v)} with value: {repr(v)}')
    return results


# ========================================================================= #
# Metrics Callback                                                          #
# ========================================================================= #


class VaeMetricLoggingCallback(BaseCallbackPeriodic):

    def __init__(
        self,
        step_end_metrics: Optional[Sequence[str]] = None,
        train_end_metrics: Optional[Sequence[str]] = None,
        every_n_steps: Optional[int] = None,
        begin_first_step: bool = False,
    ):
        super().__init__(every_n_steps, begin_first_step)
        self.step_end_metrics = step_end_metrics if step_end_metrics else []
        self.train_end_metrics = train_end_metrics if train_end_metrics else []
        assert isinstance(self.step_end_metrics, list)
        assert isinstance(self.train_end_metrics, list)
        assert self.step_end_metrics or self.train_end_metrics, 'No metrics given to step_end_metrics or train_end_metrics'

    def _compute_metrics_and_log(self, trainer: pl.Trainer, pl_module: pl.LightningModule, metrics: list, is_final=False):
        # get dataset and vae framework from trainer and module
        dataset, vae = _get_dataset_and_ae_like(trainer, pl_module, unwrap_groundtruth=True)
        # check if we need to skip
        # TODO: dataset needs to be able to handle wrapped datasets!
        if not dataset.is_ground_truth:
            warnings.warn(f'{dataset.__class__.__name__} is not an instance of {GroundTruthData.__name__}. Skipping callback: {self.__class__.__name__}!')
            return
        # get padding amount
        pad = max(7+len(k) for k in R.METRICS)  # I know this is a magic variable... im just OCD
        # compute all metrics
        for metric in metrics:
            if is_final:
                log.info(f'| {metric.__name__:<{pad}} - computing...')
            with Timer() as timer:
                scores = metric(dataset, lambda x: vae.encode(x.to(vae.device)))
            metric_results = ' '.join(f'{k}{c.GRY}={c.lMGT}{v:.3f}{c.RST}' for k, v in scores.items())
            log.info(f'| {metric.__name__:<{pad}} - time{c.GRY}={c.lYLW}{timer.pretty:<9}{c.RST} - {metric_results}')

            # log to trainer
            prefix = 'final_metric' if is_final else 'epoch_metric'
            prefixed_scores = {f'{prefix}/{k}': v for k, v in scores.items()}
            log_metrics(trainer.logger, _normalized_numeric_metrics(prefixed_scores))

            # log summary for WANDB
            # this is kinda hacky... the above should work for parallel coordinate plots
            wb_log_reduced_summaries(trainer.logger, prefixed_scores, reduction='max')

    def do_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.step_end_metrics:
            log.debug('Computing Epoch Metrics:')
            with Timer() as timer:
                self._compute_metrics_and_log(trainer, pl_module, metrics=self.step_end_metrics, is_final=False)
            log.debug(f'Computed Epoch Metrics! {timer.pretty}')

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.train_end_metrics:
            log.debug('Computing Final Metrics...')
            with Timer() as timer:
                self._compute_metrics_and_log(trainer, pl_module, metrics=self.train_end_metrics, is_final=True)
            log.debug(f'Computed Final Metrics! {timer.pretty}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
