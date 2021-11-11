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

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from numbers import Number
from pprint import pformat
from typing import Any
from typing import Dict
from typing import final
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import logging
import torch

from disent import registry
from disent.schedule import Schedule
from disent.nn.modules import DisentLightningModule


log = logging.getLogger(__name__)


# ========================================================================= #
# framework config                                                          #
# ========================================================================= #


class DisentConfigurable(object):

    @dataclass
    class cfg(object):
        def get_keys(self) -> list:
            return list(self.to_dict().keys())

        def to_dict(self) -> dict:
            return asdict(self)

        def __str__(self):
            return pformat(self.to_dict(), sort_dicts=False)

    def __init__(self, cfg: cfg = cfg()):
        if cfg is None:
            cfg = self.__class__.cfg()
            log.info(f'Initialised default config {cfg=} for {self.__class__.__name__}')
        super().__init__()
        assert isinstance(cfg, self.__class__.cfg), f'{cfg=} ({type(cfg)}) is not an instance of {self.__class__.cfg}'
        self.cfg = cfg


# ========================================================================= #
# framework                                                                 #
# ========================================================================= #


class DisentFramework(DisentConfigurable, DisentLightningModule):

    @dataclass
    class cfg(DisentConfigurable.cfg):
        # optimizer config
        optimizer: Union[str, Type[torch.optim.Optimizer]] = 'adam'
        optimizer_kwargs: Optional[Dict[str, Union[str, float, int]]] = None

    def __init__(
        self,
        cfg: cfg = None,
        # apply the batch augmentations on the GPU instead
        batch_augment: callable = None,
    ):
        # save the config values to the class
        super().__init__(cfg=cfg)
        # get the optimizer
        if isinstance(self.cfg.optimizer, str):
            if self.cfg.optimizer not in registry.OPTIMIZERS:
                raise KeyError(f'invalid optimizer: {repr(self.cfg.optimizer)}, valid optimizers are: {sorted(registry.OPTIMIZER)}, otherwise pass a torch.optim.Optimizer class instead.')
            self.cfg.optimizer = registry.OPTIMIZERS[self.cfg.optimizer]
        # check the optimizer values
        assert callable(self.cfg.optimizer)
        assert isinstance(self.cfg.optimizer_kwargs, dict) or (self.cfg.optimizer_kwargs is None), f'invalid optimizer_kwargs type, got: {type(self.cfg.optimizer_kwargs)}'
        # set default values for optimizer
        if self.cfg.optimizer_kwargs is None:
            self.cfg.optimizer_kwargs = dict()
        if 'lr' not in self.cfg.optimizer_kwargs:
            self.cfg.optimizer_kwargs['lr'] = 1e-3
            log.info('lr not specified in `optimizer_kwargs`, setting to default value of `1e-3`')
        # batch augmentations may not be implemented as dataset
        # transforms so we can apply these on the GPU instead
        assert callable(batch_augment) or (batch_augment is None)
        self._batch_augment = batch_augment
        # schedules
        # - maybe add support for schedules in the config?
        self._registered_schedules = set()
        self._active_schedules: Dict[str, Tuple[Any, Schedule]] = {}

    @final
    def configure_optimizers(self):
        optimizer_cls = self.cfg.optimizer
        # check that we can call the optimizer
        if not callable(optimizer_cls):
            raise TypeError(f'unsupported optimizer type: {type(optimizer_cls)}')
        # instantiate class
        optimizer_instance = optimizer_cls(self.parameters(), **self.cfg.optimizer_kwargs)
        # check instance
        if not isinstance(optimizer_instance, torch.optim.Optimizer):
            raise TypeError(f'returned object is not an instance of torch.optim.Optimizer, got: {type(optimizer_instance)}')
        # return the optimizer
        return optimizer_instance

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
        if loss > 1e+20:
            raise ValueError(f'The returned loss: {loss:.2e} is out of bounds: > {1e+20:.0e}')

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
