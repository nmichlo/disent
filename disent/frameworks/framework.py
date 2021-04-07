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

import warnings
from dataclasses import dataclass
from dataclasses import fields
from numbers import Number
from typing import Any
from typing import Dict
from typing import final
from typing import Tuple
from typing import Union

import logging
import torch

from disent.schedule import Schedule
from disent.util import DisentConfigurable
from disent.util import DisentLightningModule

log = logging.getLogger(__name__)


# ========================================================================= #
# framework                                                                 #
# ========================================================================= #


class BaseFramework(DisentConfigurable, DisentLightningModule):

    @dataclass
    class cfg(DisentConfigurable.cfg):
        pass

    def __init__(self, make_optimizer_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(cfg=cfg)
        # optimiser
        assert callable(make_optimizer_fn)
        self._make_optimiser_fn = make_optimizer_fn
        # batch augmentations: not implemented as dataset transforms because we want to apply these on the GPU
        assert (batch_augment is None) or callable(batch_augment)
        self._batch_augment = batch_augment
        # schedules
        self._registered_schedules = set()
        self._active_schedules: Dict[str, Tuple[Any, Schedule]] = {}

    @final
    def configure_optimizers(self):
        # return optimizers
        return self._make_optimiser_fn(self.parameters())

    @final
    def training_step(self, batch, batch_idx):
        """This is a pytorch-lightning function that should return the computed loss"""
        try:
            # augment batch with GPU support
            if self._batch_augment is not None:
                batch = self._batch_augment(batch)
            # update the config values based on registered schedules
            self._update_config_from_schedules()
            # compute loss
            loss, logs_dict = self.do_training_step(batch, batch_idx)
            # check returned values
            assert 'loss' not in logs_dict
            self._assert_valid_loss(loss)
            # log returned values
            logs_dict['loss'] = loss
            self.log_dict(logs_dict)
            # return loss
            return loss
        except Exception as e:  # pragma: no cover
            # call in all the child processes for the best chance of clearing this...
            # remove callbacks from trainer so we aren't stuck running forever!
            # TODO: this is a hack... there must be a better way to do this... could it be a pl bug?
            #       this logic is duplicated in the run_utils
            if self.trainer and self.trainer.callbacks:
                self.trainer.callbacks.clear()
            # continue propagating errors
            raise e

    @final
    def _assert_valid_loss(self, loss):
        if self.trainer.terminate_on_nan:
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError('The returned loss is nan or inf')
        if loss > 1e+12:
            raise ValueError(f'The returned loss: {loss:.2e} is out of bounds: > {1e+12:.0e}')

    def forward(self, batch) -> torch.Tensor:  # pragma: no cover
        """this function should return the single final output of the model, including the final activation"""
        raise NotImplementedError

    def do_training_step(self, batch, batch_idx) -> Tuple[torch.Tensor, Dict[str, Union[Number, torch.Tensor]]]:  # pragma: no cover
        """
        should return a dictionary of items to log with the key 'train_loss'
        as the variable to minimize
        """
        raise NotImplementedError

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Schedules                                                             #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @final
    def has_schedule(self, target: str):
        return target in self._registered_schedules

    @final
    def remove_schedule(self, target):
        if self.has_schedule(target):
            self._registered_schedules.remove(target)
            self._active_schedules.pop(target)
        else:
            raise KeyError(f'Cannot remove schedule for target {repr(target)} that does not exist!')

    @final
    def register_schedule(self, target: str, schedule: Schedule, logging=True) -> bool:
        """
        returns True if schedule has been activated, False if the key is
        not in the config and it has not be activated!
        """
        assert isinstance(target, str)
        assert isinstance(schedule, Schedule)
        assert type(schedule) is not Schedule
        # handle the case where a schedule for the target already exists!
        if self.has_schedule(target):
            raise KeyError(f'A schedule for target {repr(target)} has already been registered!')
        # check the target exists!
        possible_targets = {f.name for f in fields(self.cfg)}
        # register this target
        self._registered_schedules.add(target)
        # activate this schedule
        if target in possible_targets:
            initial_val = getattr(self.cfg, target)
            self._active_schedules[target] = (initial_val, schedule)
            if logging:
                log.info(f'Activating schedule for target {repr(target)} on {repr(self.__class__.__name__)}.')
            return True
        else:
            if logging:
                log.warning(f'Unactivated schedule for target {repr(target)} on {repr(self.__class__.__name__)} because the key was not found in the config.')
            return False

    @final
    def _update_config_from_schedules(self):
        if not self._active_schedules:
            return
        # log the step value
        self.log('scheduler/step', self.trainer.global_step)
        # update the values
        for target, (initial_val, scheduler) in self._active_schedules.items():
            # get the scheduled value
            new_value = scheduler.compute_value(step=self.trainer.global_step, value=initial_val)
            # update it on the config
            setattr(self.cfg, target, new_value)
            # log that things changed
            self.log(f'scheduled/{target}', new_value)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
